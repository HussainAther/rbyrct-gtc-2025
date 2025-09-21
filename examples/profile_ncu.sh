#!/usr/bin/env bash
set -euo pipefail
ncu --set full --target-processes all --export assets/ncu_report --export-format csv python examples/bench.py
echo "Wrote assets/ncu_report*"

