#!/usr/bin/env bash
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  python3 run_local.py
else
  python run_local.py
fi
