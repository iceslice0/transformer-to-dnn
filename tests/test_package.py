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
        self.assertEqual(get_model_adapter("pythia_70m_wikitext2").patient_name, "Pythia-70M")
        self.assertTrue(hasattr(ops, "AffineContract"))
        self.assertTrue(hasattr(ops, "SurgeryAttention"))
        self.assertTrue(hasattr(ops, "SurgeryCausalAttention"))
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

        from transformer_surgery.internal.calibration import sample_topk_scores, topk_tail_mass_stats

        scores = torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7)
        teacher, vals, idx, nk, k_top = sample_topk_scores(scores, top_k=3, max_rows=2)
        self.assertEqual(tuple(teacher.shape), (2, 7))
        self.assertEqual(tuple(vals.shape), (2, 3))
        stats = topk_tail_mass_stats(teacher, idx, nk, k_top)
        self.assertEqual(stats["count"], 2)
        self.assertGreaterEqual(stats["mean"], 0.0)
        self.assertIn("std", stats)
        self.assertGreaterEqual(stats["std"], 0.0)
        self.assertLessEqual(stats["min"], stats["max"])

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

    def test_routed_support_selection(self) -> None:
        import torch

        from transformer_surgery.internal.routing import kmeans_per_head
        from transformer_surgery.ops import GibbsTopKSoftmax, QueryCentroidRouter

        torch.manual_seed(0)
        q = torch.randn(1, 2, 6, 4)
        k = torch.randn(1, 2, 6, 4)
        scores = q @ k.transpose(-2, -1) / 2.0
        for exact in (False, True):
            gibbs = GibbsTopKSoftmax(
                seq_len=6, top_k=3, eps=1e-6, gibbs_tail_prob_eps=0.2, use_exact_tail_mass=exact
            )
            # Exact scores as selection scores reproduce the default top-k path.
            for a, b in zip(gibbs(scores), gibbs(scores, scores)):
                self.assertTrue(torch.allclose(a, b, atol=1e-6))
            # Centroids equal to the queries route every query to its own exact top-k.
            router = QueryCentroidRouter(q[0])
            for a, b in zip(gibbs(scores), gibbs(scores, router(q, k))):
                self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertEqual(tuple(router(q, k).shape), (1, 2, 6, 6))
        # Sparse scoring on the routed support equals the dense routed path (strict and fast).
        from transformer_surgery.ops import PairwiseDotBySquare

        router = QueryCentroidRouter(torch.randn(2, 3, 4))
        gibbs = GibbsTopKSoftmax(seq_len=6, top_k=3, eps=1e-6, gibbs_tail_prob_eps=0.2)
        for allow_matmul in (False, True):
            dot = PairwiseDotBySquare(4, allow_matmul=allow_matmul)
            dense = gibbs(dot(q, k), router(q, k))
            support = router.support(q, k, 3)
            sparse = gibbs.forward_selected(dot.selected(q, k, support), support, 6)
            for a, b in zip(dense, sparse):
                self.assertTrue(torch.allclose(a.float(), b.float(), atol=1e-5))
        x = torch.cat((torch.randn(2, 50, 3) * 0.01 + 5.0, torch.randn(2, 50, 3) * 0.01 - 5.0), dim=1)
        km = kmeans_per_head(x, 2, 10, torch.Generator().manual_seed(0))
        self.assertTrue(torch.all(km["relative_inertia"] < 1e-3))
        self.assertTrue(torch.all(km["empty_clusters"] == 0))

    def test_strict_sigmoid_rsqrt_matches_reference(self) -> None:
        import torch

        from transformer_surgery.internal.util import set_surgery_dtype
        from transformer_surgery.ops import GibbsTopKSoftmax, RewrittenLayerNorm, SigmoidGELU, SurgeryAttention

        set_surgery_dtype(torch.float64)
        try:
            torch.manual_seed(0)
            x = torch.randn(2, 7, 12, dtype=torch.float64) * 3 + 1
            ref = torch.nn.LayerNorm(12, eps=1e-3).double()
            ln = RewrittenLayerNorm(12, eps=1e-3, allow_matmul=False, strict_unary="sigmoid_rsqrt").double()
            self.assertTrue(torch.allclose(ln(x), ref(x), atol=1e-10))
            scores = torch.randn(1, 2, 5, 9, dtype=torch.float64) * 4
            for exact in (False, True):
                kw = dict(seq_len=9, top_k=4, eps=0.0, gibbs_tail_prob_eps=0.3, use_exact_tail_mass=exact)
                fast = GibbsTopKSoftmax(allow_matmul=True, **kw).double()
                strict = GibbsTopKSoftmax(allow_matmul=False, strict_unary="sigmoid_rsqrt", **kw).double()
                for a, b in zip(fast(scores), strict(scores)):
                    self.assertTrue(torch.allclose(a.double(), b.double(), atol=1e-10))
            g = SigmoidGELU(allow_matmul=False).double()
            self.assertTrue(torch.allclose(g(x), x * torch.sigmoid(1.702 * x), atol=1e-10))
            for exact in (False, True):
                torch.manual_seed(1)
                kw = dict(dim=12, num_heads=2, seq_len=7, top_k=3, gibbs_tail_prob_eps=0.2, use_exact_tail_mass=exact)
                fast = SurgeryAttention(allow_matmul=True, **kw).double()
                strict = SurgeryAttention(allow_matmul=False, strict_unary="sigmoid_rsqrt", **kw).double()
                strict.load_state_dict(fast.state_dict(), strict=False)
                self.assertTrue(torch.allclose(fast(x), strict(x), atol=1e-9))
        finally:
            set_surgery_dtype(torch.bfloat16)

    def test_hadamard_graph_matches_reference_with_two_unary_maps(self) -> None:
        import torch

        from transformer_surgery.internal.util import set_surgery_dtype
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
        from transformer_surgery.ops import (
            GibbsTopKSoftmax,
            PairwiseDotBySquare,
            RewrittenLayerNorm,
            SigmoidGELU,
            SurgeryAttention,
        )

        set_surgery_dtype(torch.float64)
        try:
            torch.manual_seed(0)
            x = torch.randn(2, 7, 12, dtype=torch.float64) * 3 + 1
            ref = torch.nn.LayerNorm(12, eps=1e-3).double()
            ln = RewrittenLayerNorm(12, eps=1e-3, hadamard=True).double()
            self.assertTrue(torch.allclose(ln(x), ref(x), atol=1e-10))
            scores = torch.randn(1, 2, 5, 9, dtype=torch.float64) * 4
            for exact in (False, True):
                kw = dict(seq_len=9, top_k=4, eps=0.0, gibbs_tail_prob_eps=0.3, use_exact_tail_mass=exact)
                fast = GibbsTopKSoftmax(allow_matmul=True, **kw).double()
                had = GibbsTopKSoftmax(hadamard=True, **kw).double()
                for a, b in zip(fast(scores), had(scores)):
                    self.assertTrue(torch.allclose(a.double(), b.double(), atol=1e-10))
            g = SigmoidGELU(hadamard=True).double()
            self.assertTrue(torch.allclose(g(x), x * torch.sigmoid(1.702 * x), atol=1e-10))
            q, k = torch.randn(2, 2, 5, 6, dtype=torch.float64), torch.randn(2, 2, 7, 6, dtype=torch.float64)
            dot_f, dot_h = PairwiseDotBySquare(6, allow_matmul=True), PairwiseDotBySquare(6, hadamard=True)
            self.assertTrue(torch.allclose(dot_f(q, k), dot_h(q, k), atol=1e-10))
            idx = torch.randint(0, 7, (2, 2, 5, 3))
            self.assertTrue(torch.allclose(dot_f.selected(q, k, idx), dot_h.selected(q, k, idx), atol=1e-10))
            for exact in (False, True):
                for top_k in (3, 7):
                    torch.manual_seed(1)
                    kw = dict(dim=12, num_heads=2, seq_len=7, top_k=top_k, gibbs_tail_prob_eps=0.2, use_exact_tail_mass=exact)
                    fast = SurgeryAttention(allow_matmul=True, **kw).double()
                    had = SurgeryAttention(hadamard=True, **kw).double()
                    had.load_state_dict(fast.state_dict(), strict=False)
                    self.assertTrue(torch.allclose(fast(x), had(x), atol=1e-9))
            with self.assertRaises(ValueError):
                RewrittenLayerNorm(8, eps=1e-6, allow_matmul=True, hadamard=True)
            # Unary maps of the whole Hadamard graph: sigmoid and inverse square root, plus GELU unless replaced.
            for gelu, expected in (("exact", {"NLGELU", "NLRsqrtPlusEps", "NLSigmoid"}), ("sigmoid", {"NLRsqrtPlusEps", "NLSigmoid"})):
                model = DeiTTinySurgeryModel(
                    num_classes=3, img_size=32, patch_size=16, embed_dim=12, depth=1, num_heads=2, top_k=3,
                    hadamard=True, gelu=gelu, drop_path_rate=0.0,
                ).double().eval()
                seen = set()
                for m in model.modules():
                    if type(m).__name__.startswith("NL"):
                        m.register_forward_hook(lambda mod, _i, _o: seen.add(type(mod).__name__))
                model(torch.randn(2, 3, 32, 32, dtype=torch.float64))
                self.assertEqual(seen, expected)
        finally:
            set_surgery_dtype(torch.bfloat16)

    def test_gibbs_exact_tail_mass_matches_dense_omitted(self) -> None:
        import torch

        from transformer_surgery.ops import GibbsTopKSoftmax

        scores = torch.tensor([[[[1.0, 5.0, 3.0, -2.0, 4.0], [0.5, -1.0, 2.0, 2.5, 0.0]]]])
        for allow_matmul in (False, True):
            gibbs = GibbsTopKSoftmax(
                seq_len=5,
                top_k=2,
                eps=1e-6,
                gibbs_tail_prob_eps=0.5,
                allow_matmul=allow_matmul,
                use_exact_tail_mass=True,
            )
            probs, idx, q_tail = gibbs(scores)
            flat = scores.reshape(-1, 5)
            dense = torch.softmax(flat, dim=-1)
            top_mass = dense.gather(1, idx.reshape(-1, 2)).sum(dim=-1)
            expected_tail = (1.0 - top_mass).reshape_as(q_tail.squeeze(-1))
            self.assertTrue(torch.allclose(q_tail.squeeze(-1), expected_tail, atol=1e-5))
            self.assertTrue(
                torch.allclose(
                    probs.sum(dim=-1) + q_tail.squeeze(-1),
                    torch.ones_like(expected_tail),
                    atol=1e-5,
                )
            )
            kept = dense.gather(1, idx.reshape(-1, 2)).reshape_as(probs)
            self.assertTrue(torch.allclose(probs, kept, atol=1e-5))

    def test_strict_unary_default_and_log_exp_deprecation(self) -> None:
        import warnings

        from transformer_surgery.cli.surgery_config import SurgeryConfig
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
        from transformer_surgery.ops import RewrittenLayerNorm

        self.assertEqual(SurgeryConfig().strict_unary, "sigmoid_rsqrt")
        self.assertEqual(RewrittenLayerNorm(8, eps=1e-6).strict_unary, "sigmoid_rsqrt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RewrittenLayerNorm(8, eps=1e-6, strict_unary="log_exp")
            RewrittenLayerNorm(8, eps=1e-6, allow_matmul=True, strict_unary="log_exp")
        self.assertEqual(sum(issubclass(w.category, FutureWarning) for w in caught), 1)
        # Checkpoints written before ``strict_unary`` existed are log-domain.
        extra = {
            "top_k": 4, "eps_ln": 1e-6, "gibbs_tail_prob_eps": 0.1, "disable_layernorm_replacement": False,
            "disable_attention_surgery": False, "disable_softmax_replacement": False, "allow_matmul": False,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = DeiTTinySurgeryModel.from_pretrained_extra(extra, num_classes=2)
        self.assertEqual(model.fc_norm.strict_unary, "log_exp")

    def test_integer_polarization_is_exact_for_int8(self) -> None:
        import torch

        a, b = torch.meshgrid(torch.arange(-128, 128), torch.arange(-128, 128), indexing="ij")
        plus, minus = (a + b) ** 2, (a - b) ** 2
        self.assertTrue(torch.equal((plus - minus) // 4, a * b))
        self.assertLess(int(torch.maximum(plus, minus).max()), 2**17)
        self.assertLess(int((plus - minus).abs().max()), 2**17)
        self.assertEqual(int((a * b).abs().max()), 2**14)
        self.assertLessEqual((2**17 - 1) * 2**14, 2**31 - 1)

    def test_sparse_mix_full_support_matches_gather(self) -> None:
        import torch

        from transformer_surgery.ops import SparseWeightedSumBySquare

        torch.manual_seed(0)
        v = torch.randn(2, 3, 6, 4, dtype=torch.float64)
        probs = torch.softmax(torch.randn(2, 3, 5, 6, dtype=torch.float64), dim=-1)
        idx = torch.argsort(torch.rand(2, 3, 5, 6), dim=-1)
        expected = torch.einsum("bhqk,bhqkd->bhqd", probs, v.unsqueeze(2).expand(-1, -1, 5, -1, -1).gather(
            3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, 4)))
        for allow_matmul in (False, True):
            out = SparseWeightedSumBySquare(allow_matmul=allow_matmul)(probs, idx, v)
            self.assertTrue(torch.allclose(out, expected, atol=1e-10))

    def test_ptq_per_tensor_weights_keep_per_channel_bias(self) -> None:
        import torch
        import torch.nn as nn

        from transformer_surgery.internal.calibration import (
            build_ptq_node_setup,
            build_ptq_wrapper,
            gather_ptq_dequant_moments,
            gather_ptq_range_moments,
        )

        torch.manual_seed(0)
        cases = (
            ("conv2d", nn.Conv2d(2, 3, kernel_size=1, bias=True), (4, 2, 5, 5)),
            ("linear", nn.Linear(4, 3, bias=True), (4, 6, 4)),
        )
        for kind, layer, shape in cases:
            with torch.no_grad():
                layer.bias.copy_(torch.tensor([-2.0, 0.5, 3.0]))
            model = nn.Sequential(layer)
            selected = {"0": kind}
            loader = [(torch.randn(*shape), torch.zeros(shape[0], dtype=torch.long)) for _ in range(4)]
            batch_indices = list(range(len(loader)))
            x = loader[0][0]
            with torch.no_grad():
                ref = layer(x)
            stats = gather_ptq_range_moments(model, loader, selected, batch_indices)
            setup = build_ptq_node_setup(
                "0",
                layer,
                stats["0"],
                weight_bits=8,
                activation_bits=8,
                affine_activation_bits=None,
                matmul_activation_bits=None,
                per_output_channel=False,
            )
            self.assertEqual(setup.weight_scale.ndim, 0)
            gather_ptq_dequant_moments(model, loader, selected, stats, {"0": setup}, batch_indices)
            built = build_ptq_wrapper(layer, setup, stats["0"], dequant_var_eps=1e-8)
            # Per-tensor weights still keep one output bias per channel.
            self.assertEqual(built.module.out_bias.numel(), 3)
            with torch.no_grad():
                out = built.module(x)
            self.assertLess((out - ref).abs().max().item(), 0.1 * ref.abs().max().item(), msg=kind)

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
        self.assertTrue(hasattr(calibration_mod, "sample_topk_scores"))
        self.assertTrue(hasattr(calibration_mod, "topk_tail_mass_stats"))
        self.assertTrue(hasattr(calibration_mod, "apply_gibbs_tail_calibration"))
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

    def test_pythia_adapter_and_causal_attention(self) -> None:
        import torch
        from types import SimpleNamespace

        from transformer_surgery.models.adapters import get_model_adapter
        from transformer_surgery.models.pythia.surgery_model import (
            PYTHIA_70M_DEFAULTS,
            PythiaSurgeryModel,
            arch_dict_from_hf_config,
        )
        from transformer_surgery.ops import (
            SurgeryCausalAttention,
            build_causal_attn_mask,
            build_causal_valid_key_mean_matrix,
        )
        from transformer_surgery.cli.surgery_config import SurgeryConfig

        adapter = get_model_adapter("pythia_70m_wikitext2")
        self.assertEqual(adapter.patient_name, "Pythia-70M")
        self.assertEqual(adapter.primary_metric_name(), "neg_nll")
        self.assertIn("Surgery-only", adapter.next_stage_hint())

        mask = build_causal_attn_mask(4)
        self.assertTrue(torch.isneginf(mask[0, 0, 0, 3]))
        self.assertEqual(float(mask[0, 0, 3, 0]), 0.0)
        L = build_causal_valid_key_mean_matrix(3)
        self.assertTrue(torch.allclose(L[2], torch.tensor([1 / 3, 1 / 3, 1 / 3])))

        attn = SurgeryCausalAttention(
            32, 4, 8, 8, rotary_ndims=8, allow_matmul=True, gibbs_tail_prob_eps=0.0
        )
        x = torch.randn(2, 8, 32)
        y1 = attn(x)
        x2 = x.clone()
        x2[:, -1] += 5.0
        y2 = attn(x2)
        self.assertTrue(torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5))

        # Sparse exact-tail mass: top probs + q_tail ~= 1; early rows have ~0 tail.
        sparse = SurgeryCausalAttention(
            32, 4, 16, 4, rotary_ndims=8, allow_matmul=True, use_exact_tail_mass=True, gibbs_tail_prob_eps=0.5
        )
        xs = torch.randn(1, 16, 32)
        q, k, v = sparse._split_qkv(xs)
        q, k = sparse._apply_rotary(q, k)
        scores = sparse.dot(q, k) + sparse.causal_mask
        probs, idx, q_tail = sparse.gibbs(scores)
        query_pos = torch.arange(16).view(1, 1, 16, 1)
        valid = idx <= query_pos
        probs = probs * valid.to(probs.dtype)
        self.assertTrue(torch.allclose(probs.sum(-1) + q_tail.squeeze(-1), torch.ones(1, 4, 16), atol=1e-4))
        self.assertLess(float(q_tail[0, 0, :4].detach().abs().max()), 1e-3)
        # After zeroing, no mass on future slots.
        future_mass = (probs * (~valid).to(probs.dtype)).detach().abs().sum()
        self.assertEqual(float(future_mass), 0.0)
        _ = sparse(xs)  # full forward

        # Synthetic reconstruct from extra (no HF download).
        cfg = SimpleNamespace(
            top_k=16,
            eps=1e-5,
            gibbs_tail_prob_eps=0.0,
            disable_layernorm_replacement=False,
            disable_attention_surgery=False,
            disable_softmax_replacement=False,
            allow_matmul=True,
            use_exact_tail_mass=False,
            context_length=16,
        )
        arch = dict(PYTHIA_70M_DEFAULTS)
        arch.update(
            {
                "vocab_size": 128,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
            }
        )
        model = PythiaSurgeryModel.from_arch_and_surgery(arch, cfg)
        ids = torch.randint(0, 128, (2, 16))
        logits = model(ids)
        self.assertEqual(tuple(logits.shape), (2, 16, 128))
        extra = {
            "top_k": 16,
            "gibbs_tail_prob_eps": 0.0,
            "disable_layernorm_replacement": False,
            "disable_attention_surgery": False,
            "disable_softmax_replacement": False,
            "allow_matmul": True,
            "use_exact_tail_mass": False,
            "context_length": 16,
            "eps_ln": 1e-5,
            "arch": {**arch, "seq_len": 16, "context_length": 16},
        }
        rebuilt = PythiaSurgeryModel.from_pretrained_extra(extra, cfg)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(ids), rebuilt(ids), atol=1e-6))

        # Optional HF parity when transformers is installed.
        try:
            from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        except ImportError:
            return
        hf_cfg = GPTNeoXConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            rope_scaling={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25},
            use_parallel_residual=True,
            attention_bias=True,
            hidden_act="gelu",
        )
        ref = GPTNeoXForCausalLM(hf_cfg).eval()
        scfg = SimpleNamespace(
            top_k=16,
            eps=1e-5,
            gibbs_tail_prob_eps=0.0,
            disable_layernorm_replacement=True,
            disable_attention_surgery=True,
            disable_softmax_replacement=True,
            allow_matmul=True,
            use_exact_tail_mass=False,
            context_length=16,
        )
        student = PythiaSurgeryModel.from_arch_and_surgery(arch_dict_from_hf_config(hf_cfg), scfg).eval()
        student.load_from_reference(ref)
        with torch.no_grad():
            a = ref(input_ids=ids, use_cache=False).logits
            b = student(ids)
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))

        # Configs load.
        dense = ROOT / "configs/surgery/pythia_70m_topk128_fast_tailmass_exact.json"
        sparse = ROOT / "configs/surgery/pythia_70m_topk32_fast_tailmass_exact.json"
        self.assertTrue(dense.is_file())
        self.assertTrue(sparse.is_file())
        dcfg = SurgeryConfig.load(str(dense))
        sc = SurgeryConfig.load(str(sparse))
        self.assertEqual(dcfg.model_key, "pythia_70m_wikitext2")
        self.assertEqual(dcfg.top_k, 128)
        self.assertTrue(dcfg.use_exact_tail_mass)
        self.assertTrue(sc.use_exact_tail_mass)
        self.assertEqual(sc.top_k, 32)

    def test_tokenize_and_window_helpers(self) -> None:
        import torch

        from transformer_surgery.models.pythia.loaders import tokenize_and_window

        class _Tok:
            def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
                # One id per character for determinism.
                return {"input_ids": list(range(len(text)))}

        x, y = tokenize_and_window("abcdefghij", tokenizer=_Tok(), context_length=4)
        self.assertEqual(tuple(x.shape), (2, 4))  # tokens 0..9 -> windows at 0 and 4
        self.assertTrue(torch.equal(x[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(y[0], torch.tensor([1, 2, 3, 4])))
        self.assertTrue(torch.equal(x[1], torch.tensor([4, 5, 6, 7])))
        self.assertTrue(torch.equal(y[1], torch.tensor([5, 6, 7, 8])))


if __name__ == "__main__":
    unittest.main()
