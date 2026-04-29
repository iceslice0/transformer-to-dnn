#!/usr/bin/env python3
"""CLI wrapper for post-training quantization."""

from transformer_surgery.cli.common import apply_device_from_config, apply_dtype_from_config
from transformer_surgery.cli.ptq_config import parse_ptq_config
from transformer_surgery.ptq import run_ptq


def main() -> None:
    cfg = parse_ptq_config()
    device = apply_device_from_config(cfg)
    dtype = apply_dtype_from_config(cfg)
    run_ptq(cfg, device=device, dtype=dtype)


if __name__ == "__main__":
    main()
