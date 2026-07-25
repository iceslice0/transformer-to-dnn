import glob
import json
import math
import os
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageSmokeTests(unittest.TestCase):
    def test_imports(self) -> None:
        import transformer_surgery
        from transformer_surgery import distill, models, ops, ptq, surgery
        from transformer_surgery.internal import util
        from transformer_surgery.internal import reporting
        from transformer_surgery.models import pet
        from transformer_surgery.models.adapters import get_model_adapter

        self.assertTrue(transformer_surgery.__all__)
        self.assertTrue(hasattr(models, "DeiTTinySurgeryModel"))
        self.assertEqual(get_model_adapter("deit_tiny_pet").patient_name, "DeiT-Tiny")
        self.assertTrue(hasattr(ops, "AffineContract"))
        self.assertTrue(hasattr(ops, "SurgeryAttention"))
        self.assertFalse(hasattr(ops, "RoutingMax"))
        self.assertEqual(pet.PET_NUM_CLASSES, 37)
        self.assertTrue(hasattr(reporting, "traceable_artifact_path"))
        self.assertTrue(hasattr(surgery, "surgery"))
        self.assertTrue(hasattr(distill, "run_distill"))
        self.assertTrue(hasattr(ptq, "run_ptq"))
        self.assertIsNone(importlib.util.find_spec("transformer_surgery.pet"))
        self.assertIsNone(importlib.util.find_spec(".".join(("transformer_surgery", "pipe" + "line"))))

    def test_configs_load(self) -> None:
        from transformer_surgery.cli.distill_config import (
            CLI_JEFFREYS_CONFIG_DEFAULT,
            JeffreysDistillConfig,
        )
        from transformer_surgery.cli.pretrain_config import (
            CLI_PRETRAIN_CONFIG_DEFAULT,
            PretrainPetConfig,
        )
        from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
        from transformer_surgery.cli.surgery_config import (
            CLI_SURGERY_CONFIG_DEFAULT,
            SurgeryConfig,
        )

        defaults = [
            CLI_PRETRAIN_CONFIG_DEFAULT,
            CLI_SURGERY_CONFIG_DEFAULT,
            CLI_JEFFREYS_CONFIG_DEFAULT,
            "configs/ptq/64_fast_jeffreys_8bit.json",
        ]
        for path in defaults:
            self.assertTrue((ROOT / path).is_file(), path)

        for path in glob.glob(str(ROOT / "configs/pretrain/*.json")):
            PretrainPetConfig.load(path)
        for path in glob.glob(str(ROOT / "configs/surgery/*.json")):
            cfg = SurgeryConfig.load(path)
            lim = cfg.gibbs_tail_calibration_batches
            self.assertTrue(lim is None or lim >= 1)
        for path in glob.glob(str(ROOT / "configs/distill/*.json")):
            cfg = JeffreysDistillConfig.load(path)
            self.assertEqual(cfg.base_seed, 42)
            self.assertGreaterEqual(cfg.num_trainings, 1)
        for path in glob.glob(str(ROOT / "configs/ptq/*.json")):
            PTQSurgeryConfig.load(path)
        self.assertFalse(SurgeryConfig.load(str(ROOT / CLI_SURGERY_CONFIG_DEFAULT)).disable_calib_gibbs_tail_prob)
        forbidden_metadata_field = "_".join(("meta", "json"))
        for path in glob.glob(str(ROOT / "configs/**/*.json"), recursive=True):
            with open(path, encoding="utf-8") as f:
                self.assertNotIn(forbidden_metadata_field, json.load(f), path)

    def test_gibbs_tail_calibration_helpers(self) -> None:
        import torch

        from transformer_surgery.models.adapters import sample_topk_scores, topk_tail_mass_stats

        scores = torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7)
        teacher, vals, idx, nk, k_top = sample_topk_scores(scores, top_k=3, max_rows=2)
        self.assertEqual(tuple(teacher.shape), (2, 7))
        self.assertEqual(tuple(vals.shape), (2, 3))
        stats = topk_tail_mass_stats(teacher, idx, nk, k_top)
        self.assertEqual(stats["count"], 2)
        self.assertGreaterEqual(stats["mean"], 0.0)

    def test_gibbs_topk_stabilizes_from_ordered_topk(self) -> None:
        import torch

        from transformer_surgery.ops import GibbsTopKSoftmax

        scores = torch.tensor([[[[1.0, 5.0, 3.0, -2.0, 4.0]]]])
        gibbs = GibbsTopKSoftmax(
            seq_len=5,
            top_k=3,
            eps=1e-6,
            gibbs_tail_prob_eps=0.0,
            allow_matmul=False,
        )
        probs, idx, q_tail = gibbs(scores)
        raw_vals, expected_idx = torch.topk(scores, k=3, dim=-1, largest=True, sorted=True)
        stable = raw_vals - raw_vals[..., :1]
        expected = torch.exp(stable) / (torch.exp(stable).sum(dim=-1, keepdim=True) + 1e-6)
        self.assertTrue(torch.equal(idx, expected_idx))
        self.assertTrue(torch.allclose(probs, expected, atol=1e-6))
        self.assertTrue(torch.equal(q_tail, torch.zeros_like(q_tail)))

    def test_ptq_calibration_samples_random_batches(self) -> None:
        import torch
        import torch.nn as nn
        from types import SimpleNamespace

        from transformer_surgery.internal.calibration import (
            build_ptq_node_setup,
            build_ptq_wrapper,
            gather_ptq_dequant_moments,
            gather_ptq_range_moments,
            make_ptq_wrapper,
            ptq_wrapper_from_reload_config,
            sample_calibration_batch_indices,
        )

        gen = torch.Generator().manual_seed(123)
        expected_gen = torch.Generator().manual_seed(123)
        indices = sample_calibration_batch_indices(10, 4, generator=gen)
        expected = sorted(torch.randperm(10, generator=expected_gen)[:4].tolist())
        self.assertEqual(indices, expected)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(set(indices)), 4)
        self.assertEqual(sample_calibration_batch_indices(10, None), list(range(10)))
        self.assertEqual(sample_calibration_batch_indices(0, None), [])

        model = nn.Sequential(nn.Linear(3, 2, bias=False))
        selected = {"0": "linear"}
        batch_indices = [1, 3]
        loader = [(torch.full((4, 3), float(i)), torch.zeros(4, dtype=torch.long)) for i in range(5)]
        stats = gather_ptq_range_moments(model, loader, selected, batch_indices)
        data = stats["0"]
        self.assertEqual(data.examples_total, 8)
        self.assertGreaterEqual(data.input_max_abs[0].item(), 3.0)

        cfg = SimpleNamespace(
            weight_bits=8,
            activation_bits=8,
            affine_activation_bits=None,
            matmul_activation_bits=None,
            per_output_channel=True,
            dequant_var_eps=1e-8,
        )
        setup = build_ptq_node_setup(
            "0",
            model[0],
            data,
            weight_bits=cfg.weight_bits,
            activation_bits=cfg.activation_bits,
            affine_activation_bits=cfg.affine_activation_bits,
            matmul_activation_bits=cfg.matmul_activation_bits,
            per_output_channel=cfg.per_output_channel,
        )
        gather_ptq_dequant_moments(
            model,
            loader,
            selected,
            stats,
            {"0": setup},
            batch_indices,
        )
        self.assertEqual(data.dequant_examples_total, 8)
        self.assertGreater(data.bias_fit.count, 0)
        wrapper = make_ptq_wrapper(model[0], setup, data, dequant_var_eps=cfg.dequant_var_eps)
        self.assertEqual(wrapper.quantizer.input_scale.numel(), 1)
        self.assertIsInstance(wrapper.accumulator, nn.Linear)
        self.assertEqual(setup.activation_bits, 8)
        built = build_ptq_wrapper(model[0], setup, data, dequant_var_eps=cfg.dequant_var_eps)
        skeleton = ptq_wrapper_from_reload_config(built.reload_config, nn.Linear(3, 2, bias=False))
        skeleton.load_state_dict(built.module.state_dict(), strict=True)
        self.assertEqual(tuple(skeleton(torch.ones(2, 3)).shape), (2, 2))

        from transformer_surgery.internal.calibration import (
            ptq_activation_bits_for_kind,
            ptq_activation_group,
        )

        self.assertEqual(ptq_activation_group("linear"), "linear")
        self.assertEqual(ptq_activation_group("conv2d"), "linear")
        self.assertEqual(ptq_activation_group("affine_scale"), "affine")
        self.assertEqual(ptq_activation_group("matmul"), "matmul")
        # Linear/Conv keep activation_bits even when affine override is set.
        self.assertEqual(
            ptq_activation_bits_for_kind(
                "linear", 8, affine_activation_bits=12, matmul_activation_bits=8
            ),
            8,
        )
        self.assertEqual(
            ptq_activation_bits_for_kind(
                "conv2d", 8, affine_activation_bits=12, matmul_activation_bits=8
            ),
            8,
        )
        self.assertEqual(
            ptq_activation_bits_for_kind(
                "affine_scale", 8, affine_activation_bits=12, matmul_activation_bits=8
            ),
            12,
        )
        self.assertEqual(
            ptq_activation_bits_for_kind(
                "matmul", 8, affine_activation_bits=12, matmul_activation_bits=8
            ),
            8,
        )
        self.assertEqual(
            ptq_activation_bits_for_kind(
                "affine_scale", 8, affine_activation_bits=None, matmul_activation_bits=None
            ),
            8,
        )

    def test_traceable_artifact_names(self) -> None:
        from transformer_surgery.internal.reporting import (
            metadata_path_for_checkpoint,
            traceable_artifact_path,
            traceable_log_path,
        )

        pretrain_cfg = ROOT / "configs/pretrain/pet_deit_tiny.json"
        surgery_cfg = ROOT / "configs/surgery/topk64_fast.json"
        distill_cfg = ROOT / "configs/distill/64_fast_jeffreys.json"
        ptq_cfg = ROOT / "configs/ptq/64_fast_jeffreys_8bit.json"
        self.assertTrue(
            traceable_artifact_path(
                "artifacts/checkpoints/anything.pt",
                str(pretrain_cfg),
                "ts-pretrain-pet",
            ).endswith("artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt")
        )
        surgery_pt = traceable_artifact_path(
            "artifacts/checkpoints/anything.pt",
            str(surgery_cfg),
            "ts-surgery",
            extension=".pt",
        )
        self.assertTrue(
            surgery_pt.endswith("artifacts/checkpoints/ts_surgery_topk64_fast.pt")
        )
        self.assertTrue(
            traceable_artifact_path(
                "artifacts/checkpoints/anything.pt",
                str(distill_cfg),
                "ts-distill",
                extension=".pt",
            ).endswith("artifacts/checkpoints/ts_distill_64_fast_jeffreys.pt")
        )
        self.assertTrue(
            traceable_artifact_path(
                "artifacts/checkpoints/anything.pt",
                str(ptq_cfg),
                "ts-ptq",
                "",
                ".pt",
            ).endswith("artifacts/checkpoints/ts_ptq_64_fast_jeffreys_8bit.pt")
        )
        self.assertTrue(
            metadata_path_for_checkpoint(surgery_pt).endswith(
                "artifacts/metadata/ts_surgery_topk64_fast.json"
            )
        )
        self.assertTrue(
            traceable_log_path("artifacts/logs", str(surgery_cfg), "ts-surgery", "model_after_surgery").endswith(
                "artifacts/logs/ts_surgery_topk64_fast_model_after_surgery.txt"
            )
        )
        self.assertTrue(
            traceable_log_path("artifacts/logs", str(ptq_cfg), "ts-ptq", "model_after_ptq").endswith(
                "artifacts/logs/ts_ptq_64_fast_jeffreys_8bit_model_after_ptq.txt"
            )
        )

    def test_distill_repeat_metadata_helpers(self) -> None:
        from transformer_surgery.distill import _validation_accuracy_summary, merge_post_distill_into_surgery_meta

        runs = [
            {"run_index": 0, "seed": 42, "val_acc": 0.60, "val_ce_mean": 1.0, "val_jeffreys_mean": 0.3},
            {"run_index": 1, "seed": 43, "val_acc": 0.80, "val_ce_mean": 0.8, "val_jeffreys_mean": 0.2},
        ]
        summary = _validation_accuracy_summary(runs)
        self.assertAlmostEqual(summary["val_acc_mean"], 0.70)
        self.assertAlmostEqual(summary["val_acc_std"], math.sqrt(0.02))
        with tempfile.TemporaryDirectory() as td:
            meta_path = os.path.join(td, "distill.json")
            merge_post_distill_into_surgery_meta(
                meta_path,
                0.80,
                0.8,
                0.2,
                run_results=runs,
                accuracy_summary=summary,
                best_run=runs[1],
            )
            with open(meta_path, encoding="utf-8") as f:
                cal = json.load(f)["calibration"]
        self.assertEqual(cal["student_post_distill_runs"], runs)
        self.assertEqual(cal["student_post_distill_best_seed"], 43)
        self.assertAlmostEqual(cal["student_post_distill_val_acc_mean"], 0.70)

    def test_cli_help(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        modules = [
            "transformer_surgery.cli.pretrain_pet",
            "transformer_surgery.cli.surgery",
            "transformer_surgery.cli.distill",
            "transformer_surgery.cli.ptq",
        ]
        for module in modules:
            with self.subTest(module=module):
                proc = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("--config", proc.stdout)

    def test_model_instantiates_on_cpu(self) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from transformer_surgery.models import DeiTTinySurgeryModel
        from transformer_surgery.models import adapters
        from transformer_surgery.internal import calibration as calibration_mod
        from transformer_surgery.models.adapters import get_model_adapter
        from transformer_surgery.ops import SurgeryAttention

        model = DeiTTinySurgeryModel(num_classes=37).cpu()
        self.assertEqual(model.num_classes, 37)
        self.assertEqual(model.seq_len, 197)
        self.assertIsInstance(model.blocks[0].attn, SurgeryAttention)
        self.assertEqual(sum(p.numel() for p in model.blocks[0].mlp.act.parameters()), 0)
        x = torch.linspace(-3.0, 3.0, 9)
        self.assertTrue(torch.equal(model.blocks[0].mlp.act(x), F.gelu(x)))
        tail_prob = model.blocks[0].attn.gibbs.gibbs_tail_prob_eps
        self.assertIsInstance(tail_prob, nn.Parameter)
        self.assertEqual(tuple(tail_prob.shape), ())
        self.assertTrue(hasattr(model.blocks[0].attn.gibbs, "scale_top_probs_by_tail"))

        adapter = get_model_adapter("deit_tiny_pet")
        for helper_name in ("_attention_scores", "_sample_topk_scores", "_topk_tail_mass_stats"):
            self.assertFalse(hasattr(adapter, helper_name), helper_name)
        self.assertTrue(hasattr(adapters, "sample_topk_scores"))
        self.assertTrue(hasattr(adapters, "topk_tail_mass_stats"))
        self.assertTrue(hasattr(adapters, "apply_gibbs_tail_calibration"))
        self.assertTrue(hasattr(calibration_mod, "calibrate_vit_reference"))
        self.assertTrue(hasattr(calibration_mod, "fused_qkv_attention_qk_scores"))

        disabled_calibration = DeiTTinySurgeryModel(
            num_classes=37,
            depth=1,
            gibbs_tail_prob_eps=0.1,
        ).cpu()
        skipped = adapter.apply_calibration(
            disabled_calibration,
            {"disable_calib_gibbs_tail_prob": True, "gibbs_tail_prob_eps_calibrated_by_block": [0.02]},
        )
        self.assertEqual(skipped, {})
        self.assertAlmostEqual(
            float(disabled_calibration.blocks[0].attn.gibbs.gibbs_tail_prob_eps.detach()), 0.1, places=2
        )

        calibrated = DeiTTinySurgeryModel(
            num_classes=37,
            depth=2,
            gibbs_tail_prob_eps=0.1,
        ).cpu()
        applied = adapter.apply_calibration(
            calibrated,
            {"gibbs_tail_prob_eps_calibrated_by_block": [0.02, 0.03]},
        )
        self.assertEqual(applied["gibbs_tail_prob_eps_applied_by_block"], [0.02, 0.03])
        self.assertAlmostEqual(float(calibrated.blocks[0].attn.gibbs.gibbs_tail_prob_eps.detach()), 0.02, places=3)
        self.assertTrue(calibrated.blocks[0].attn.gibbs.gibbs_tail_prob_eps.requires_grad)


if __name__ == "__main__":
    unittest.main()
