#!/usr/bin/env python3
"""
CLI for Jeffreys distillation. All defaults and semantics live in
:class:`pet_reference_utils.JeffreysDistillConfig` and :func:`jeffreys_distill_pipeline`.
"""

from __future__ import annotations

from pet_reference_utils import run_jeffreys_distill_cli


def main() -> None:
    run_jeffreys_distill_cli()


if __name__ == "__main__":
    main()
