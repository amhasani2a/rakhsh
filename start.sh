#!/usr/bin/env bash
# Rakhsh one-click start (macOS / Linux): creates a venv on first run, then opens the web UI.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q -e ".[formats]"
fi
exec ./.venv/bin/rakhsh serve "$@"
