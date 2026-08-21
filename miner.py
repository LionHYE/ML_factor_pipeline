"""Minimal GP factor miner (no external GP dependency).

Anti-overfit design:
- fitness = |IS ICIR| - parsimony * nodes   (never raw returns, never full-sample)
- hard caps: depth <= max_depth, nodes <= max_nodes (invalid offspring rejected)
- small operator set, hoist mutation against bloat, early stop on stagnation

Usage:
  python miner.py                         # synthetic demo
  python miner.py --symbols-file symbols_l1.txt
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import yaml

import expression as ex
import metrics as mx
import ops


class Miner:
    def __init__(self, cfg, fields, is_idx, seed=0):
        self.cfg = cfg
        self.m = cfg["miner"]
        self.fields = fields
        self.is_idx = is_idx
        self.fwd = mx.forward_returns(fields["close"], cfg["horizon"], cfg["lag"]).loc[is_idx]
        self.rng = random.Random(seed)
        self.evaluated = 0
        self.cache = {}

    def _valid(self, t):
        try:
            ex.validate(t)
        except Exception:
            return False
        return ex.complexity(t) <= self.m["max_nodes"] and ex.depth(t) <= self.m["max_depth"]

    def _random(self):
        while True:
            t = ex.random_expr(self.rng, self.m["max_depth"])
            if self._valid(t):
                return t

    def fitness(self, t):
        key = t.to_str()
        if key in self.cache:
            return self.cache[key]
        self.evaluated += 1
        try:
            f = ex.evaluate(t, self.fields).loc[self.is_idx]
            f = f.replace([np.inf, -np.inf], np.nan)
            st = mx.ic_stats(mx.ic_series(f, self.fwd, self.cfg["min_names"]))
            if np.isfinite(st["icir"]):
                fit = abs(st["icir"]) - self.m["parsimony"] * ex.complexity(t)
            else:
                fit = -1e9
        except Exception:
            fit = -1e9
        self.cache[key] = fit
        return fit

    def _tournament(self, pop, fits):
        best, bf = None, -1e18
        for _ in range(self.m["tournament"]):
            i = self.rng.randrange(len(pop))
            if fits[i] > bf:
                bf, best = fits[i], pop[i]
        return best

    def _crossover(self, a, b):
        a = a.copy()
        na = [x for x in ex.nodes(a) if x.kind != "window"]
        nb = [x for x in ex.nodes(b) if x.kind != "window"]
        ta = self.rng.choice(na)
        tb = self.rng.choice(nb).copy()
        ta.kind, ta.name, ta.children, ta.value = tb.kind, tb.name, tb.children, tb.value
        return a

    def _subtree_mut(self, a):
        return self._crossover(a, self._random())

    def _hoist_mut(self, a):
        a = a.copy()
        calls = [x for x in ex.nodes(a) if x.kind == "call"]
        if not calls:
            return a
        t = self.rng.choice(calls)
        sub = [x for x in ex.nodes(t) if x.kind != "window" and x is not t]
        if not sub:
            return a
        s = self.rng.choice(sub)
        t.kind, t.name, t.children, t.value = s.kind, s.name, s.children, s.value
        return a

    def _point_mut(self, a):
        a = a.copy()
        t = self.rng.choice(ex.nodes(a))
        if t.kind == "window":
            t.value = self.rng.choice(ops.WINDOWS)
        elif t.kind == "terminal":
            t.name = self.rng.choice(ops.TERMINALS)
        elif t.name in ops.BINARY:
            t.name = self.rng.choice(ops.BINARY)
        elif t.name in ops.UNARY:
            t.name = self.rng.choice(ops.UNARY)
        else:
            t.name = self.rng.choice(ops.TS)
        return a

    def run(self, top_k=20, verbose=True):
        m = self.m
        pop = [self._random() for _ in range(m["population"])]
        best_hist = []
        for gen in range(m["generations"]):
            fits = [self.fitness(t) for t in pop]
            order = np.argsort(fits)[::-1]
            best_hist.append(fits[order[0]])
            if verbose:
                print(f"[miner] gen {gen}: best={fits[order[0]]:.4f} "
                      f"expr={pop[order[0]].to_str()} evaluated={self.evaluated}",
                      flush=True)
            if len(best_hist) >= 4 and best_hist[-1] <= best_hist[-4] + 1e-6:
                if verbose:
                    print("[miner] early stop: no improvement for 3 generations",
                          flush=True)
                break
            new = [pop[i].copy() for i in order[:max(2, m["population"] // 100)]]
            while len(new) < m["population"]:
                r = self.rng.random()
                if r < m["p_crossover"]:
                    child = self._crossover(self._tournament(pop, fits), self._tournament(pop, fits))
                elif r < m["p_crossover"] + m["p_subtree"]:
                    child = self._subtree_mut(self._tournament(pop, fits))
                elif r < m["p_crossover"] + m["p_subtree"] + m["p_hoist"]:
                    child = self._hoist_mut(self._tournament(pop, fits))
                elif r < m["p_crossover"] + m["p_subtree"] + m["p_hoist"] + m["p_point"]:
                    child = self._point_mut(self._tournament(pop, fits))
                else:
                    child = self._random()
                if self._valid(child):
                    new.append(child)
            pop = new
        fits = [self.fitness(t) for t in pop]
        seen = {}
        for t, f in sorted(zip(pop, fits), key=lambda x: -x[1]):
            s = t.to_str()
            if s not in seen:
                seen[s] = f
            if len(seen) >= top_k:
                break
        return [{"expr": s, "fitness": float(f)} for s, f in seen.items()], self.evaluated


def _read_symbols(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def _run_real(args):
    import data as dataio

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = _read_symbols(args.symbols_file)
    if not symbols:
        raise SystemExit(f"no symbols found in {args.symbols_file}")

    fields = dataio.build_panel(
        symbols, args.timeframe, args.data_dir, refresh=not args.no_refresh
    )
    n = len(fields["close"])
    is_n = int(n * cfg["is_fraction"])
    if is_n < 2:
        raise SystemExit(f"not enough bars for IS split: {n}")
    is_idx = fields["close"].index[:is_n]
    print(f"[miner] dataset=bybit:{args.timeframe}:{len(fields['close'].columns)}syms "
          f"bars={n} IS={is_n} ({cfg['is_fraction']:.1%})", flush=True)

    top, evaluated = Miner(cfg, fields, is_idx, seed=args.seed).run(
        top_k=args.top_k
    )
    print(f"\n[miner] evaluated {evaluated} candidates; top-{len(top)} IS candidates:",
          flush=True)
    for i, row in enumerate(top, 1):
        print(f"  {i:>2}. {row['fitness']:.4f}  {row['expr']}", flush=True)

    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join("runs", f"miner_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": f"bybit:{args.timeframe}:{len(fields['close'].columns)}syms",
            "bars": n,
            "is_bars": is_n,
            "is_fraction": cfg["is_fraction"],
            "symbols": list(fields["close"].columns),
            "evaluated": evaluated,
            "top": top,
        }, f, indent=1)
    print(f"[miner] top candidates saved: {out}", flush=True)

    if args.no_verify:
        return

    print("\n[miner] verifying top candidates through run_pipeline.py "
          "(cache fixed with --no-refresh)", flush=True)
    for i, row in enumerate(top, 1):
        print(f"\n===== candidate {i}/{len(top)}: {row['expr']} =====", flush=True)
        cmd = [
            sys.executable, "-u", "run_pipeline.py",
            "--config", args.config,
            "--expr", row["expr"],
            "--symbols-file", args.symbols_file,
            "--timeframe", args.timeframe,
            "--data-dir", args.data_dir,
            "--no-refresh",
            "--seed", str(args.seed),
        ]
        result = subprocess.run(cmd)
        if result.returncode:
            raise SystemExit(
                f"[miner] verification process failed with exit code "
                f"{result.returncode}"
            )


def _run_synthetic_demo():
    # small demo on synthetic data
    from synthetic import make_synthetic

    cfg = yaml.safe_load(open("config.yaml"))
    cfg["miner"]["population"] = 200
    cfg["miner"]["generations"] = 5
    fields = make_synthetic(effect=0.12, seed=1)
    n = len(fields["close"])
    is_idx = fields["close"].index[:int(n * cfg["is_fraction"])]
    top, m_evals = Miner(cfg, fields, is_idx, seed=0).run(top_k=10)
    print(f"\nevaluated {m_evals} candidates; top:")
    for t in top:
        print(f"  {t['fitness']:.4f}  {t['expr']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mine GP factors on IS data")
    ap.add_argument("--symbols-file", default="",
                    help="real-data symbol list; omit for synthetic demo")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--timeframe", default="1d", choices=["1d", "4h", "1h", "15m"])
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--no-refresh", action="store_true",
                    help="local cache only, never hit network")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip run_pipeline.py verification of top candidates")
    args = ap.parse_args()

    if args.symbols_file:
        _run_real(args)
    else:
        _run_synthetic_demo()
