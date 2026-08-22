#!/usr/bin/env python3
"""Test ONE factor expression through gates G1-G12.

Passing every gate means: this single factor, before any correlation check
against your existing factor pool, is eligible. Nothing more.

Usage:
  python3 run_pipeline.py --expr "reverse(ts_delta(close, 5))" --synthetic --synthetic-effect 0.12
  python3 run_pipeline.py --expr "rank(ts_std_dev(returns, 20))" --symbols-file symbols_example.txt
  python3 run_pipeline.py --expr "..." --symbols BTCUSDT,ETHUSDT --timeframe 1d --no-refresh
"""
import argparse
import json
import os
import time

import yaml

import expression as ex
import gates
import registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True, help='e.g. "reverse(ts_delta(close, 5))"')
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-effect", type=float, default=0.0)
    ap.add_argument("--symbols", default="", help="comma separated Bybit symbols")
    ap.add_argument("--symbols-file", default="", help="one symbol per line")
    ap.add_argument("--timeframe", default="1d", choices=["1d", "4h", "1h", "15m"])
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--no-refresh", action="store_true", help="local cache only, never hit network")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate gate order up front; a bad config must fail before any work.
    gate_order = gates.resolve_gate_order(cfg)
    cfg_hash = registry.config_fingerprint(cfg, gate_order)

    if args.synthetic:
        from synthetic import make_synthetic
        fields = make_synthetic(effect=args.synthetic_effect, seed=args.seed)
        dataset = f"synthetic(effect={args.synthetic_effect})"
    else:
        import data as dataio
        if args.symbols_file:
            symbols = [s.strip() for s in open(args.symbols_file)
                       if s.strip() and not s.startswith("#")]
        elif args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            raise SystemExit("provide --symbols / --symbols-file, or use --synthetic")
        fields = dataio.build_panel(symbols, args.timeframe, args.data_dir,
                                    refresh=not args.no_refresh)
        dataset = f"bybit:{args.timeframe}:{len(fields['close'].columns)}syms"

    tree = ex.parse(args.expr)
    t0 = time.time()
    report = gates.run_gates(tree, fields, cfg, seed=args.seed)
    elapsed = time.time() - t0

    print("=" * 76)
    print(f"expr    : {tree.to_str()}")
    print(f"config  : {args.config}   hash={cfg_hash}   short_circuit={bool(cfg.get('short_circuit', False))}")
    print(f"dataset : {dataset}   elapsed: {elapsed:.1f}s   direction: {report['direction']:+.0f}")
    print("-" * 76)
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['gate']:<22} {json.dumps(r['detail'], default=str)[:130]}")
    for name in report.get("skipped", []):
        print(f"[SKIP] {name:<22} short-circuited after first failure")
    print("-" * 76)
    print(f"VERDICT : {'PASS -- eligible for the correlation stage' if report['verdict'] else 'REJECT'}")

    # G12: registry -- every candidate is logged, pass or fail
    reg_path = os.path.join("runs", "registry.jsonl")
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "expr": tree.to_str(),
        "dataset": dataset,
        "verdict": report["verdict"],
        "direction": report["direction"],
        "config_hash": cfg_hash,
        "gates": {r["gate"]: r["passed"] for r in report["results"]},
    }
    if report.get("skipped"):
        record["skipped"] = report["skipped"]
    note = str(cfg.get("note") or "").strip()
    if note:
        record["config_note"] = note
    registry.append(reg_path, record)
    print(f"registry: M = {registry.count(reg_path)} candidates tested so far ({reg_path})")

    out = os.path.join("runs", f"report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump({"record": record, "full": report}, f, default=str, indent=1)
    print(f"report  : {out}")


if __name__ == "__main__":
    main()
