"""Minimal GP factor miner (no external GP dependency).

Anti-overfit design:
- fitness = |IS ICIR| - parsimony * nodes   (never raw returns, never full-sample)
- hard caps: depth <= max_depth, nodes <= max_nodes (invalid offspring rejected)
- small operator set, hoist mutation against bloat, early stop on stagnation
"""
import random

import numpy as np

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
                      f"expr={pop[order[0]].to_str()} evaluated={self.evaluated}")
            if len(best_hist) >= 4 and best_hist[-1] <= best_hist[-4] + 1e-6:
                if verbose:
                    print("[miner] early stop: no improvement for 3 generations")
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


if __name__ == "__main__":
    # small demo on synthetic data
    import yaml
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
