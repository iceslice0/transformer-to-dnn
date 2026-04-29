import glob
import json
import os
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageSmokeTests(unittest.TestCase):
    def test_imports(self) -> None:
        import transformer_surgery
        from transformer_surgery import distill, models, ops, ptq, surgery, util
        from transformer_surgery.models import pet
        from transformer_surgery.models.adapters import get_model_adapter

        self.assertTrue(transformer_surgery.__all__)
        self.assertTrue(hasattr(models, "DeiTTinySurgeryModel"))
        self.assertEqual(get_model_adapter("deit_tiny_pet").patient_name, "DeiT-Tiny")
        self.assertTrue(hasattr(ops, "AffineContract"))
        self.assertEqual(pet.PET_NUM_CLASSES, 37)
        self.assertTrue(hasattr(util, "traceable_artifact_path"))
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
            SurgeryConfig.load(path)
        for path in glob.glob(str(ROOT / "configs/distill/*.json")):
            JeffreysDistillConfig.load(path)
        for path in glob.glob(str(ROOT / "configs/ptq/*.json")):
            PTQSurgeryConfig.load(path)
        forbidden_metadata_field = "_".join(("meta", "json"))
        for path in glob.glob(str(ROOT / "configs/**/*.json"), recursive=True):
            with open(path, encoding="utf-8") as f:
                self.assertNotIn(forbidden_metadata_field, json.load(f), path)

    def test_traceable_artifact_names(self) -> None:
        from transformer_surgery.util import metadata_path_for_checkpoint, traceable_artifact_path, traceable_log_path

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
                "wrapped",
                ".pt",
            ).endswith("artifacts/checkpoints/ts_ptq_64_fast_jeffreys_8bit_wrapped.pt")
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
        from transformer_surgery.models import DeiTTinySurgeryModel

        model = DeiTTinySurgeryModel(num_classes=37).cpu()
        self.assertEqual(model.num_classes, 37)
        self.assertEqual(model.seq_len, 197)


if __name__ == "__main__":
    unittest.main()
