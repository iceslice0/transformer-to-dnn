import glob
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageSmokeTests(unittest.TestCase):
    def test_imports(self) -> None:
        import transformer_surgery
        from transformer_surgery import models, ops, pet, pipeline, ptq
        from transformer_surgery.models.adapters import get_model_adapter

        self.assertTrue(transformer_surgery.__all__)
        self.assertTrue(hasattr(models, "DeiTTinySurgeryModel"))
        self.assertEqual(get_model_adapter("deit_tiny_pet").patient_name, "DeiT-Tiny")
        self.assertTrue(hasattr(ops, "AffineContract"))
        self.assertTrue(hasattr(pet, "PretrainPetConfig"))
        self.assertTrue(hasattr(pipeline, "parse_cli_config"))
        self.assertTrue(hasattr(ptq, "PTQSurgeryConfig"))

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
            CLI_SURGERY_RUN_CONFIG_DEFAULT,
            SurgeryRunConfig,
        )

        defaults = [
            CLI_PRETRAIN_CONFIG_DEFAULT,
            CLI_SURGERY_RUN_CONFIG_DEFAULT,
            CLI_JEFFREYS_CONFIG_DEFAULT,
            "configs/ptq/full_8bit.json",
        ]
        for path in defaults:
            self.assertTrue((ROOT / path).is_file(), path)

        for path in glob.glob(str(ROOT / "configs/pretrain/*.json")):
            PretrainPetConfig.load(path)
        for path in glob.glob(str(ROOT / "configs/surgery/*.json")):
            SurgeryRunConfig.load(path)
        for path in glob.glob(str(ROOT / "configs/distill/*.json")):
            JeffreysDistillConfig.load(path)
        for path in glob.glob(str(ROOT / "configs/ptq/*.json")):
            PTQSurgeryConfig.load(path)

    def test_cli_help(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        modules = [
            "transformer_surgery.cli.pretrain_pet",
            "transformer_surgery.cli.run_surgery",
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
