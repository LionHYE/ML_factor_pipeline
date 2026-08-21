"""Summarize runs/registry.jsonl: how many candidates were tested (M), which
gates kill most candidates, and which expressions passed everything.

Usage:
  python harvest.py
  python harvest.py --registry runs/registry.jsonl --since 2026-08-22

Honesty note: M (total lines) is the real number of tries behind any PASS.
Before trusting a PASS, compare its |t| against a placebo null whose breadth
matches the actual search size (see G5 note in the README/config).
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="runs/registry.jsonl")
    ap.add_argument("--since", default="", help="only records with time >= this (ISO prefix)")
    args = ap.parse_args()

    records = []
    try:
        with open(args.registry, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        raise SystemExit(f"no registry at {args.registry} -- nothing has been tested yet")

    if args.since:
        records = [r for r in records if r.get("time", "") >= args.since]
    if not records:
        raise SystemExit("no records in the selected window")

    m_total = len(records)
    passes = [r for r in records if r.get("verdict")]

    # first-failing-gate distribution (gate order preserved from the record)
    first_fail = Counter()
    fail_any = Counter()
    for r in records:
        gates = r.get("gates", {})
        failed = [g for g, ok in gates.items() if not ok]
        for g in failed:
            fail_any[g] += 1
        if failed:
            first_fail[failed[0]] += 1

    # dedupe expressions (same expr may be re-discovered across seeds)
    uniq = {}
    for r in records:
        uniq.setdefault(r.get("expr"), r)

    print("=" * 72)
    print(f"registry : {args.registry}")
    print(f"M (tries): {m_total}   unique exprs: {len(uniq)}   PASS: {len(passes)}")
    print("-" * 72)
    print("gate failure counts (candidate can fail several gates):")
    for g, c in sorted(fail_any.items(), key=lambda x: -x[1]):
        print(f"  {g:<24} {c:>5}  ({c / m_total:.0%})")
    print("-" * 72)
    if passes:
        print("PASS list (eligible for the correlation stage, nothing more):")
        seen = set()
        for r in passes:
            e = r.get("expr")
            if e in seen:
                continue
            seen.add(e)
            print(f"  [{r.get('time')}] dir={r.get('direction'):+.0f}  {e}")
        print(f"\nreminder: these survived a search of M={m_total}. Re-check G5 with a")
        print("null at least as broad as the actual search before believing any of them.")
    else:
        print("PASS list: (empty) -- the gates are doing their job.")
    print("=" * 72)


if __name__ == "__main__":
    main()
