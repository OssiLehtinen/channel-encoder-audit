"""Rebuild results_full/summary.txt from the JSON outputs without
re-running anything. Useful when a late stage failed or you want to
regenerate tables after editing the summary format.

Usage:
    uv run python -m fft_encode.rebuild_summary --dir results_full
"""

from __future__ import annotations

import argparse
import json
import os

from .run_all import write_summary


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_full")
    args = ap.parse_args()

    main = load(os.path.join(args.dir, "main.json"))
    dmodel = load(os.path.join(args.dir, "dmodel.json"))
    carrier = load(os.path.join(args.dir, "carrier.json"))
    probe = load(os.path.join(args.dir, "probe.json"))

    main_runs = main.get("runs", []) if main else []
    mask_results = main.get("mask", []) if main else []
    dmodel_runs = dmodel.get("runs", []) if dmodel else []
    carrier_runs = carrier.get("runs", []) if carrier else []
    probe_results = probe if probe else []

    write_summary(args.dir, main_runs, dmodel_runs, carrier_runs,
                  probe_results, mask_results)
    print(f"\nrebuilt {args.dir}/summary.txt")


if __name__ == "__main__":
    main()
