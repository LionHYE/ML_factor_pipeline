#!/usr/bin/env python3
"""Harvest runs/registry.jsonl after a mining batch.

Records are grouped by config_hash: results produced under different gate
parameters or gate order are different experiments. M is never pooled across
groups, pass rates are never compared across groups.

Usage:
  python harvest.py
  python harvest.py --registry runs/registry.jsonl --since 2026-08-22
"""
import argparse
import json
import os
from collections import Counter, defaultdict


def load(path, since):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and r.get("time", "") < since:
                continue
            records.append(r)
    return records


def summarize(rs):
    latest = {}
    for r in rs:  # dedupe: keep the latest attempt per expression
        latest[r.get("expr", "?")] = r
    fails = Counter()
    for r in rs:
        for gate, ok in (r.get("gates") or {}).items():
            if not ok:
                fails[gate] += 1
    passes = [r for r in latest.values() if r.get("verdict")]
    return latest, fails, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=os.path.join("runs", "registry.jsonl"))
    ap.add_argument("--since", default="", help="ISO time prefix, e.g. 2026-08-22")
    args = ap.parse_args()
    if not os.path.exists(args.registry):
        raise SystemExit(f"registry not found: {args.registry}")

    records = load(args.registry, args.since)
    groups = defaultdict(list)
    for r in records:
        groups[r.get("config_hash", "(legacy-no-hash)")].append(r)

    print(f"records: {len(records)}   config groups: {len(groups)}")
    if len(groups) > 1:
        print("WARNING: multiple config_hash groups. Each group is a separate")
        print("experiment: do not pool M, compare pass rates, or mix PASS lists.")

    for h, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        latest, fails, passes = summarize(rs)
        notes = sorted({r["config_note"] for r in rs if r.get("config_note")})
        print("=" * 72)
        print(f"config_hash: {h}   M = {len(rs)} attempts   unique exprs = {len(latest)}")
        if notes:
            print(f"config notes: {notes}")
        if fails:
            print("gate failure counts (all attempts):")
            for gate, cnt in fails.most_common():
                print(f"  {gate:<24} {cnt}")
        print(f"PASS (latest attempt per expr): {len(passes)}")
        for r in sorted(passes, key=lambda r: r.get("time", "")):
            print(f"  [{r.get('time', '?')}] dir={r.get('direction', 0):+.0f}  {r.get('expr')}")
        if passes:
            print(f"NOTE: with M = {len(rs)} searches behind them, re-check any")
            print("marginal G5 result with g5_n_placebo >= M before trusting it.")

    if not records:
        print("no records matched the filter")


if __name__ == "__main__":
    main()
