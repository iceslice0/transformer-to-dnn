#!/usr/bin/env python3
"""CLI wrapper for model-adapter CE + Jeffreys distillation."""

from __future__ import annotations

from transformer_surgery.cli.common import apply_device_from_config
from transformer_surgery.cli.distill_config import parse_distill_config
from transformer_surgery.distill import run_distill


def main() -> None:
    cfg = parse_distill_config()
    apply_device_from_config(cfg)
    run_distill(cfg)


if __name__ == "__main__":
    main()
