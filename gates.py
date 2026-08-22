"""Single-factor validation gates G1-G11 (G12 = registry, handled by the runner).

Thresholds live in config.yaml and must be frozen before running any candidate.

Gate execution order is config-driven (config.yaml: gate_order), subject to the
hard dependency constraints in _MUST_PRECEDE. Changing any parameter OR the
order changes the config fingerprint (registry.config_fingerprint); records
produced under different fingerprints are different experiments and must never
be pooled or compared.
"""
import random

import numpy as np
import pandas as pd

import expression as ex
import metrics as mx

DEFAULT_GATE_ORDER = [
    "G1_data", "G2_lookahead", "G3_complexity", "G4_is_ic", "G5_placebo",
    "G6_decay_tradability", "G7_neutralization", "G8_after_cost",
    "G9_robustness", "G10_oos", "G11_bootstrap",
]

# hard dependency constraints: gate -> gates that must run before it
_MUST_PRECEDE = {
    "G1_data": set(),
    "G2_lookahead": {"G1_data"},
    "G3_complexity": {"G1_data"},
    "G4_is_ic": {"G1_data", "G2_lookahead"},
    "G5_placebo": {"G4_is_ic"},
    "G6_decay_tradability": {"G4_is_ic"},
    "G7_neutralization": {"G4_is_ic"},
    "G8_after_cost": {"G4_is_ic"},
    "G9_robustness": {"G4_is_ic"},
    "G10_oos": {"G4_is_ic"},
    "G11_bootstrap": {"G4_is_ic"},
}


def resolve_gate_order(cfg):
    """Validate and return the gate execution order from config.

    Raises ValueError on missing/unknown gates or dependency violations.
    """
    order = [str(x) for x in (cfg.get("gate_order") or DEFAULT_GATE_ORDER)]
    if sorted(order) != sorted(DEFAULT_GATE_ORDER):
        missing = sorted(set(DEFAULT_GATE_ORDER) - set(order))
        extra = sorted(set(order) - set(DEFAULT_GATE_ORDER))
        raise ValueError(
            f"gate_order must contain exactly G1-G11 once each "
            f"(missing={missing}, unknown={extra})")
    pos = {name: i for i, name in enumerate(order)}
    for gate, deps in _MUST_PRECEDE.items():
        for dep in deps:
            if pos[dep] > pos[gate]:
                raise ValueError(
                    f"gate_order invalid: {dep} must run before {gate}")
    return order


def _num(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


def _res(gate, passed, **detail):
    return {"gate": gate, "passed": bool(passed),
            "detail": {k: _num(v) for k, v in detail.items()}}


# ---------------------------------------------------------------- gate funcs

def _g1_data(ctx):
    factor, close = ctx["factor"], ctx["close"]
    g, min_names = ctx["g"], ctx["min_names"]
    row_valid = factor.notna().sum(axis=1)
    warm = row_valid >= min_names
    if warm.sum() < 100:
        r = _res("G1_data", False, usable_rows=int(warm.sum()))
        r["_fatal"] = True
        return r
    first = warm.idxmax()
    live = factor.loc[first:]
    denom = close.loc[first:].notna().sum().sum()
    coverage = live.notna().sum().sum() / max(denom, 1)
    cs_std = float(live.std(axis=1).mean())
    return _res("G1_data", coverage >= g["g1_min_coverage"] and cs_std > 0,
                coverage=coverage, mean_cs_std=cs_std)


def _g2_lookahead(ctx):
    tree, fields, factor = ctx["tree"], ctx["fields"], ctx["factor"]
    idx, n, seed = ctx["idx"], ctx["n"], ctx["seed"]
    nprng = np.random.default_rng(seed)
    cuts = sorted(nprng.choice(np.arange(int(n * 0.5), n - 2), size=5, replace=False))
    leak = False
    for c in cuts:
        cut_date = idx[int(c)]
        trunc = {k: v.loc[:cut_date] for k, v in fields.items()}
        try:
            f2 = ex.evaluate(tree, trunc).replace([np.inf, -np.inf], np.nan)
            a = f2.iloc[-1].values.astype(float)
            b = factor.loc[cut_date].values.astype(float)
            if not np.allclose(a, b, equal_nan=True, rtol=1e-8, atol=1e-10):
                leak = True
                break
        except Exception:
            leak = True
            break
    return _res("G2_lookahead", not leak, checked_cuts=len(cuts))


def _g3_complexity(ctx):
    tree, g = ctx["tree"], ctx["g"]
    nodes_n, dep = ex.complexity(tree), ex.depth(tree)
    return _res("G3_complexity",
                nodes_n <= g["g3_max_nodes"] and dep <= g["g3_max_depth"],
                nodes=nodes_n, depth=dep)


def _g4_is_ic(ctx):
    st_is, g = ctx["st_is"], ctx["g"]
    return _res("G4_is_ic", abs(st_is["t"]) >= g["g4_min_abs_t"],
                n=st_is["n"], mean=st_is["mean"], t=st_is["t"],
                icir=st_is["icir"], direction=ctx["direction"])


def _g5_placebo(ctx):
    g, fields, seed = ctx["g"], ctx["fields"], ctx["seed"]
    is_idx, fwd, min_names = ctx["is_idx"], ctx["fwd"], ctx["min_names"]
    st_is = ctx["st_is"]
    prng = random.Random(seed + 1)
    null_ts = []
    trials = 0
    max_trials = int(g["g5_n_placebo"]) * 4
    while len(null_ts) < int(g["g5_n_placebo"]) and trials < max_trials:
        trials += 1
        cand = ex.random_expr(prng, max_depth=int(g["g3_max_depth"]))
        if ex.complexity(cand) > int(g["g3_max_nodes"]):
            continue
        try:
            pf = ex.evaluate(cand, fields).replace([np.inf, -np.inf], np.nan)
            s = mx.ic_stats(mx.ic_series(pf.loc[is_idx], fwd.loc[is_idx], min_names))
            if np.isfinite(s["t"]) and s["n"] >= 50:
                null_ts.append(abs(s["t"]))
        except Exception:
            continue
    if len(null_ts) >= 50:
        thr = float(np.percentile(null_ts, g["g5_percentile"]))
        return _res("G5_placebo", abs(st_is["t"]) > thr,
                    null_n=len(null_ts), null_pctile=thr,
                    candidate_abs_t=abs(st_is["t"]))
    return _res("G5_placebo", False, null_n=len(null_ts),
                note="not enough valid placebo factors")


def _g6_decay_tradability(ctx):
    g, close, lag = ctx["g"], ctx["close"], ctx["lag"]
    factor, is_idx, min_names = ctx["factor"], ctx["is_idx"], ctx["min_names"]
    base_sign = ctx["base_sign"]
    hz = {}
    for h in g["g6_horizons"]:
        fh = mx.forward_returns(close, int(h), lag)
        hz[int(h)] = mx.ic_stats(mx.ic_series(factor.loc[is_idx], fh.loc[is_idx], min_names))["mean"]
    agree = sum(1 for v in hz.values() if np.isfinite(v) and np.sign(v) == base_sign and base_sign != 0)
    fac_ac = float(np.nanmean(mx.ic_series(factor, factor.shift(1), min_names)))
    return _res("G6_decay_tradability",
                agree >= max(2, len(hz) - 1) and fac_ac >= g["g6_min_factor_autocorr"],
                factor_autocorr=fac_ac, sign_agree=agree,
                ic_by_horizon={str(k): _num(v) for k, v in hz.items()})


def _g7_neutralization(ctx):
    g, fields, factor = ctx["g"], ctx["fields"], ctx["factor"]
    idx, is_idx, fwd, min_names = ctx["idx"], ctx["is_idx"], ctx["fwd"], ctx["min_names"]
    m_is = ctx["m_is"]
    rets = fields["returns"]
    mkt = rets.mean(axis=1)
    cov = rets.mul(mkt, axis=0).rolling(60).mean() - rets.rolling(60).mean().mul(mkt.rolling(60).mean(), axis=0)
    var = (mkt ** 2).rolling(60).mean() - mkt.rolling(60).mean() ** 2
    beta = cov.div(var + 1e-12, axis=0)
    size = np.log(fields["dollar_volume"].rolling(20).mean() + 1.0)
    mom = rets.rolling(20).sum()
    resid = pd.DataFrame(np.nan, index=idx, columns=factor.columns)
    for d in is_idx:
        y = factor.loc[d].values.astype(float)
        X = np.column_stack([np.ones(len(y)), beta.loc[d].values,
                             size.loc[d].values, mom.loc[d].values])
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < min_names + 4:
            continue
        coef, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        r = np.full(len(y), np.nan)
        r[m] = y[m] - X[m] @ coef
        resid.loc[d] = r
    st_res = mx.ic_stats(mx.ic_series(resid.loc[is_idx], fwd.loc[is_idx], min_names))
    if np.isfinite(m_is) and m_is != 0 and np.isfinite(st_res["mean"]):
        retained = abs(st_res["mean"]) / abs(m_is)
    else:
        retained = 0.0
    return _res("G7_neutralization",
                retained >= g["g7_min_retained"] and abs(st_res["t"]) >= g["g7_min_abs_t"],
                resid_ic_mean=st_res["mean"], resid_t=st_res["t"],
                retained_ratio=retained)


def _g8_after_cost(ctx):
    g, cfg, close, lag = ctx["g"], ctx["cfg"], ctx["close"], ctx["lag"]
    factor, direction, is_idx = ctx["factor"], ctx["direction"], ctx["is_idx"]
    sfactor = factor * direction
    ranks = sfactor.rank(axis=1, pct=True)
    longs = (ranks >= 0.8).astype(float)
    shorts = (ranks <= 0.2).astype(float)
    nl = longs.sum(axis=1)
    ns = shorts.sum(axis=1)
    pos = longs.div(nl.where(nl > 0), axis=0).fillna(0.0) - shorts.div(ns.where(ns > 0), axis=0).fillna(0.0)
    fwd1 = mx.forward_returns(close, 1, lag)
    gross = (pos * fwd1).sum(axis=1)
    turn = (pos - pos.shift(1)).abs().sum(axis=1)
    net = (gross - turn * (float(cfg["cost_bps_per_side"]) / 1e4)).loc[is_idx]
    sharpe = mx.annualized_sharpe(net, int(cfg["periods_per_year"]))
    return _res("G8_after_cost", sharpe >= g["g8_min_after_cost_sharpe"],
                after_cost_sharpe=sharpe,
                mean_daily_turnover=float(turn.loc[is_idx].mean()))


def _g9_robustness(ctx):
    g, tree, fields = ctx["g"], ctx["tree"], ctx["fields"]
    is_idx, fwd, min_names = ctx["is_idx"], ctx["fwd"], ctx["min_names"]
    base_sign, ic_is, direction = ctx["base_sign"], ctx["ic_is"], ctx["direction"]
    pert_ok = True
    pert_detail = {}
    if ex.has_windows(tree):
        for k in g["g9_window_scales"]:
            t2 = ex.scale_windows(tree, float(k))
            try:
                pf = ex.evaluate(t2, fields).replace([np.inf, -np.inf], np.nan)
                s = mx.ic_stats(mx.ic_series(pf.loc[is_idx], fwd.loc[is_idx], min_names))
                pert_detail[str(k)] = _num(s["t"])
                same_sign = np.isfinite(s["mean"]) and np.sign(s["mean"]) == base_sign
                if not (same_sign and abs(s["t"]) >= g["g9_min_pert_abs_t"]):
                    pert_ok = False
            except Exception:
                pert_ok = False
    mkt = fields["returns"].mean(axis=1)
    vol = mkt.rolling(20).std()
    trend = mkt.rolling(60).mean()
    ic_adj = ic_is * direction
    vmed = vol.loc[is_idx].median()
    regs = {}
    for name, mask in {
        "hivol_up": (vol > vmed) & (trend > 0),
        "hivol_dn": (vol > vmed) & (trend <= 0),
        "lovol_up": (vol <= vmed) & (trend > 0),
        "lovol_dn": (vol <= vmed) & (trend <= 0),
    }.items():
        sel = ic_adj[mask.reindex(is_idx).fillna(False)].dropna()
        if len(sel) >= 30:
            regs[name] = float(sel.mean())
    agree_r = sum(1 for v in regs.values() if v > 0)
    need = min(int(g["g9_min_regime_agree"]), max(len(regs), 1))
    return _res("G9_robustness", pert_ok and agree_r >= need,
                window_t=pert_detail, regime_ic=regs, regimes_positive=agree_r)


def _g10_oos(ctx):
    g, factor, oos_idx = ctx["g"], ctx["factor"], ctx["oos_idx"]
    fwd, min_names = ctx["fwd"], ctx["min_names"]
    direction, m_is = ctx["direction"], ctx["m_is"]
    ic_oos = mx.ic_series(factor.loc[oos_idx], fwd.loc[oos_idx], min_names) * direction
    st_oos = mx.ic_stats(ic_oos)
    mean_is_adj = abs(m_is) if np.isfinite(m_is) else 0.0
    if mean_is_adj > 0 and np.isfinite(st_oos["mean"]):
        retention = st_oos["mean"] / mean_is_adj
    else:
        retention = 0.0
    return _res("G10_oos",
                st_oos["t"] >= g["g10_min_oos_t"] and retention >= g["g10_min_oos_retention"],
                oos_mean=st_oos["mean"], oos_t=st_oos["t"],
                retention=retention, oos_n=st_oos["n"])


def _g11_bootstrap(ctx):
    g, factor, fwd = ctx["g"], ctx["factor"], ctx["fwd"]
    min_names, direction, seed = ctx["min_names"], ctx["direction"], ctx["seed"]
    ic_full = mx.ic_series(factor, fwd, min_names) * direction
    lo, hi = mx.block_bootstrap_ci(ic_full, int(g["g11_block"]), int(g["g11_n_boot"]), seed=seed)
    return _res("G11_bootstrap", bool(np.isfinite(lo) and lo > 0), ci_lo=lo, ci_hi=hi)


_GATE_FUNCS = {
    "G1_data": _g1_data,
    "G2_lookahead": _g2_lookahead,
    "G3_complexity": _g3_complexity,
    "G4_is_ic": _g4_is_ic,
    "G5_placebo": _g5_placebo,
    "G6_decay_tradability": _g6_decay_tradability,
    "G7_neutralization": _g7_neutralization,
    "G8_after_cost": _g8_after_cost,
    "G9_robustness": _g9_robustness,
    "G10_oos": _g10_oos,
    "G11_bootstrap": _g11_bootstrap,
}


def run_gates(tree, fields, cfg, seed=0):
    g = cfg["gates"]
    order = resolve_gate_order(cfg)
    close = fields["close"]
    horizon, lag = int(cfg["horizon"]), int(cfg["lag"])
    min_names = int(cfg["min_names"])

    # ---- evaluate the factor (G0: not reorderable) ----
    try:
        ex.validate(tree)
        factor = ex.evaluate(tree, fields)
        factor = factor.replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        return {"verdict": False, "direction": 1.0, "skipped": list(order),
                "results": [_res("G0_eval", False, error=str(e))]}

    idx = factor.index
    n = len(idx)
    is_end = int(n * float(cfg["is_fraction"]))
    embargo = int(cfg["embargo_bars"])
    is_idx = idx[:is_end]
    oos_idx = idx[is_end + embargo:]
    fwd = mx.forward_returns(close, horizon, lag)

    # IS IC stats and direction are shared context; G4 only applies the
    # threshold. This keeps direction available regardless of gate order.
    ic_is = mx.ic_series(factor.loc[is_idx], fwd.loc[is_idx], min_names)
    st_is = mx.ic_stats(ic_is)
    m_is = st_is["mean"]
    direction = -1.0 if (np.isfinite(m_is) and m_is < 0) else 1.0
    base_sign = np.sign(m_is) if np.isfinite(m_is) else 0.0

    ctx = {
        "tree": tree, "fields": fields, "cfg": cfg, "g": g, "seed": seed,
        "factor": factor, "close": close, "idx": idx, "n": n,
        "is_idx": is_idx, "oos_idx": oos_idx, "fwd": fwd,
        "min_names": min_names, "lag": lag,
        "ic_is": ic_is, "st_is": st_is, "m_is": m_is,
        "direction": direction, "base_sign": base_sign,
    }

    short_circuit = bool(cfg.get("short_circuit", False))
    results, skipped = [], []
    stop = False
    for name in order:
        if stop:
            skipped.append(name)
            continue
        r = _GATE_FUNCS[name](ctx)
        fatal = r.pop("_fatal", False)
        results.append(r)
        if fatal or (short_circuit and not r["passed"]):
            stop = True

    verdict = (not skipped) and all(r["passed"] for r in results)
    return {"verdict": bool(verdict), "direction": float(direction),
            "results": results, "skipped": skipped}
