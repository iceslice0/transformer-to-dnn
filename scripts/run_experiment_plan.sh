#!/usr/bin/env bash
set -euo pipefail

# Run the NeurIPS experiment plan end-to-end through make targets.
# Usage:
#   bash scripts/run_experiment_plan.sh [all|a|b|c|d]

PART="${1:-all}"

case "$PART" in
  all)
    make all
    ;;
  a|A)
    make a
    ;;
  b|B)
    make b
    ;;
  c|C)
    make c
    ;;
  d|D)
    make d
    ;;
  *)
    echo "Unknown part: $PART"
    echo "Usage: bash scripts/run_experiment_plan.sh [all|a|b|c|d]"
    exit 2
    ;;
esac
