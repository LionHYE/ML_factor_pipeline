"""Overnight batch mining: repeat miner.py runs with different seeds on a frozen cache.

Workflow:
  1. First run refreshes the local data cache once.
  2. Every later run uses --no-refresh, so all runs see the exact same data.
  3. Each run mines top-k IS candidates and verifies them through run_pipeline.py
     (G1-G12); every candidate is appended to runs/registry.jsonl by G12.
  4. Next morning: python harvest.py

Usage:
  python mine_batch.py --symbols-file symbols_l1.txt --hours 8
  python mine_batch.py --symbols-file symbols_l1.txt --runs 5 --top-k 10

Stops before starting a new run once the time budget is used up (a run in
progress is never killed). Seeds are seed0, seed0+1, ... so every run explores
a different region and everything stays reproducible.
"""
import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="Loop miner.py over seeds within a time budget")
    ap.add_argument("--symbols-file", required=True)
    ap.add_argument("--timeframe", default="1d", choices=["1d", "4h", "1h", "15m"])
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--hours", type=float, default=8.0, help="time budget (default 8h)")
    ap.add_argument("--runs", type=int, default=0,
                    help="max number of runs (0 = unlimited within time budget)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="top-k per run to verify (default 10; verification is the slow part)")
    ap.add_argument("--seed0", type=int, default=100, help="first seed (default 100)")
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    run_i = 0
    while True:
        if args.runs and run_i >= args.runs:
            print(f"[batch] reached max runs ({args.runs}); stopping")
            break
        if time.time() >= deadline:
            print("[batch] time budget exhausted; stopping")
            break
        seed = args.seed0 + run_i
        cmd = [sys.executable, "-u", "miner.py",
               "--symbols-file", args.symbols_file,
               "--timeframe", args.timeframe,
               "--data-dir", args.data_dir,
               "--config", args.config,
               "--top-k", str(args.top_k),
               "--seed", str(seed)]
        if run_i > 0:
            cmd.append("--no-refresh")  # freeze the cache after the first run
        remaining = (deadline - time.time()) / 3600
        print(f"\n[batch] ===== run {run_i + 1} seed={seed} "
              f"(budget left {remaining:.1f}h) =====", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        print(f"[batch] run {run_i + 1} finished in {(time.time() - t0) / 60:.0f} min "
              f"exit={r.returncode}", flush=True)
        if r.returncode:
            print("[batch] run failed; stopping to avoid burning the night on errors")
            break
        run_i += 1

    print(f"[batch] done: {run_i} completed runs. Next: python harvest.py")


if __name__ == "__main__":
    main()
"""
"""
