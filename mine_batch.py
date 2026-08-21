"""Overnight batch mining: repeat miner.py runs with different seeds on a frozen cache.

Workflow:
  1. The first run refreshes the local data cache once (serial, alone).
  2. Every later run uses --no-refresh, so all runs see the exact same data.
  3. With --n-jobs N, up to N miner runs execute in parallel after the warm-up.
  4. Each run mines top-k IS candidates and verifies them through run_pipeline.py
     (G1-G12); every candidate is appended to runs/registry.jsonl by G12.
  5. Next morning: python harvest.py

Usage:
  python mine_batch.py --symbols-file symbols_l1.txt --hours 8
  python mine_batch.py --symbols-file symbols_l1.txt --hours 8 --n-jobs 3
  python mine_batch.py --symbols-file symbols_l1.txt --runs 5 --top-k 10

Notes on --n-jobs:
  - Each job is a full miner.py process (GP search + per-candidate verification),
    so keep n_jobs <= physical cores / 2; verification is numpy-heavy.
  - Parallel jobs append to runs/registry.jsonl concurrently. Line-level appends
    are small and practically atomic, but keep n_jobs modest (<= 4) to be safe.
  - Job starts are staggered by 2 seconds so runs/miner_<timestamp>.json names
    cannot collide.

Progress lines (machine-readable, for the LionAlgo platform status parser):
  [batch] progress runs_done=<n> active=<k> elapsed_h=<x.xx> budget_h=<y.y>
"""
import argparse
import subprocess
import sys
import threading
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
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel miner runs after the warm-up run (default 1; keep <= 4)")
    args = ap.parse_args()

    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be >= 1")

    t_start = time.time()
    deadline = t_start + args.hours * 3600
    lock = threading.Lock()
    state = {"next": 0, "done": 0, "active": 0, "stop": False}

    def progress():
        elapsed = (time.time() - t_start) / 3600
        print(f"[batch] progress runs_done={state['done']} active={state['active']} "
              f"elapsed_h={elapsed:.2f} budget_h={args.hours:.1f}", flush=True)

    def claim_seed():
        """Reserve the next seed, or None if budget/limit reached."""
        with lock:
            if state["stop"] or time.time() >= deadline:
                return None
            if args.runs and state["next"] >= args.runs:
                return None
            seed = args.seed0 + state["next"]
            state["next"] += 1
            state["active"] += 1
            return seed

    def release(ok):
        with lock:
            state["active"] -= 1
            if ok:
                state["done"] += 1
            else:
                state["stop"] = True
        progress()

    def run_once(seed, refresh):
        cmd = [sys.executable, "-u", "miner.py",
               "--symbols-file", args.symbols_file,
               "--timeframe", args.timeframe,
               "--data-dir", args.data_dir,
               "--config", args.config,
               "--top-k", str(args.top_k),
               "--seed", str(seed)]
        if not refresh:
            cmd.append("--no-refresh")
        print(f"[batch] ===== run seed={seed} start "
              f"(budget left {(deadline - time.time()) / 3600:.1f}h) =====", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        print(f"[batch] run seed={seed} finished in {(time.time() - t0) / 60:.0f} min "
              f"exit={r.returncode}", flush=True)
        return r.returncode == 0

    # --- Phase 1: warm-up run, serial, refreshes the cache once ---
    seed = claim_seed()
    if seed is None:
        raise SystemExit("[batch] nothing to do (budget or run limit is zero)")
    print("[batch] warm-up run: refreshing data cache once...", flush=True)
    ok = run_once(seed, refresh=True)
    release(ok)
    if not ok:
        raise SystemExit("[batch] warm-up run failed; aborting before burning the night on errors")

    # --- Phase 2: parallel workers on the frozen cache ---
    def worker(index):
        time.sleep(index * 2)  # stagger: avoid miner_<timestamp>.json collisions
        while True:
            s = claim_seed()
            if s is None:
                return
            release(run_once(s, refresh=False))

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(args.n_jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if state["stop"]:
        print("[batch] stopped early because a run failed", flush=True)
    print(f"[batch] done: {state['done']} completed runs. Next: python harvest.py", flush=True)


if __name__ == "__main__":
    main()
