#!/usr/bin/env python3
"""CLI wrapper for the surgery transform."""

from __future__ import annotations

from transformer_surgery.cli.surgery_config import parse_surgery_config
from transformer_surgery.cli.common import apply_device_from_config, apply_dtype_from_config
from transformer_surgery.surgery import surgery


def main() -> None:
    cfg = parse_surgery_config()
    apply_device_from_config(cfg)
    apply_dtype_from_config(cfg)
    surgery(cfg)


if __name__ == "__main__":
    main()
