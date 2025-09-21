#!/usr/bin/env bash
set -euo pipefail
nsys profile -o assets/mart_demo --trace=cuda,osrt python examples/bench.py
nsys stats assets/mart_demo.qdrep > assets/nsys_stats.txt
echo "Wrote assets/mart_demo.qdrep and assets/nsys_stats.txt"

