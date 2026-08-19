"""Single-factor validation gates G1-G11 (G12 = registry, handled by the runner).
Thresholds live in config.yaml and must be frozen before running any candidate.
"""
import random

import numpy as np
import pandas as pd

import expression as ex
import metrics as mx


def _num(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


def _res(gate, passed, **detail):
    return {"gate": gate, "passed": bool(passed),
            "detail": {k: _num(v) for k, v in detail.items()}}


def run_gates(tree, fields, cfg, seed=0):
    g = cfg["gates"]
    close = fields["close"]
    horizon, lag = int(cfg["horizon"]), int(cfg["lag"])
    min_names = int(cfg["min_names"])
    results = []

    # ---- evaluate the factor ----
    try:
        ex.validate(tree)
        factor = ex.evaluate(tree, fields)
        factor = factor.replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        return {"verdict": False, "direction": 1.0,
                "results": [_res("G0_eval", False, error=str(e))]}

    idx = factor.index
    n = len(idx)
    is_end = int(n * float(cfg["is_fraction"]))
    embargo = int(cfg["embargo_bars"])
    is_idx = idx[:is_end]
    oos_idx = idx[is_end + embargo:]
    fwd = mx.forward_returns(close, horizon, lag)

    # ---- G1 data quality ----
    row_valid = factor.notna().sum(axis=1)
    warm = row_valid >= min_names
    if warm.sum() < 100:
        results.append(_res("G1_data", False, usable_rows=int(warm.sum())))
        return {"verdict": False, "direction": 1.0, "results": results}
    first = warm.idxmax()
    live = factor.loc[first:]
    denom = close.loc[first:].notna().sum().sum()
    coverage = live.notna().sum().sum() / max(denom, 1)
    cs_std = float(live.std(axis=1).mean())
    results.append(_res("G1_data", coverage >= g["g1_min_coverage"] and cs_std > 0,
                        coverage=coverage, mean_cs_std=cs_std))

    # ---- G2 look-ahead detection ----
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
    results.append(_res("G2_lookahead", not leak, checked_cuts=len(cuts)))

    # ---- G3 complexity ----
    nodes_n, dep = ex.complexity(tree), ex.depth(tree)
    results.append(_res("G3_complexity",
                        nodes_n <= g["g3_max_nodes"] and dep <= g["g3_max_depth"],
                        nodes=nodes_n, depth=dep))

    # ---- G4 in-sample IC (direction decided on IS only) ----
    ic_is = mx.ic_series(factor.loc[is_idx], fwd.loc[is_idx], min_names)
    st_is = mx.ic_stats(ic_is)
    m_is = st_is["mean"]
    direction = -1.0 if (np.isfinite(m_is) and m_is < 0) else 1.0
    base_sign = np.sign(m_is) if np.isfinite(m_is) else 0.0
    results.append(_res("G4_is_ic", abs(st_is["t"]) >= g["g4_min_abs_t"],
                        n=st_is["n"], mean=st_is["mean"], t=st_is["t"],
                        icir=st_is["icir"], direction=direction))

    # ---- G5 placebo null distribution ----
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
        results.append(_res("G5_placebo", abs(st_is["t"]) > thr,
                            null_n=len(null_ts), null_pctile=thr,
                            candidate_abs_t=abs(st_is["t"])))
    else:
        results.append(_res("G5_placebo", False, null_n=len(null_ts),
                            note="not enough valid placebo factors"))

    # ---- G6 horizon decay + tradability ----
    hz = {}
    for h in g["g6_horizons"]:
        fh = mx.forward_returns(close, int(h), lag)
        hz[int(h)] = mx.ic_stats(mx.ic_series(factor.loc[is_idx], fh.loc[is_idx], min_names))["mean"]
    agree = sum(1 for v in hz.values() if np.isfinite(v) and np.sign(v) == base_sign and base_sign != 0)
    fac_ac = float(np.nanmean(mx.ic_series(factor, factor.shift(1), min_names)))
    results.append(_res("G6_decay_tradability",
                        agree >= max(2, len(hz) - 1) and fac_ac >= g["g6_min_factor_autocorr"],
                        factor_autocorr=fac_ac, sign_agree=agree,
                        ic_by_horizon={str(k): _num(v) for k, v in hz.items()}))

    # ---- G7 neutralization (beta / size / momentum) ----
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
    results.append(_res("G7_neutralization",
                        retained >= g["g7_min_retained"] and abs(st_res["t"]) >= g["g7_min_abs_t"],
                        resid_ic_mean=st_res["mean"], resid_t=st_res["t"],
                        retained_ratio=retained))

    # ---- G8 after-cost long-short backtest (IS) ----
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
    results.append(_res("G8_after_cost", sharpe >= g["g8_min_after_cost_sharpe"],
                        after_cost_sharpe=sharpe,
                        mean_daily_turnover=float(turn.loc[is_idx].mean())))

    # ---- G9 robustness: window perturbation + regimes ----
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
    results.append(_res("G9_robustness", pert_ok and agree_r >= need,
                        window_t=pert_detail, regime_ic=regs, regimes_positive=agree_r))

    # ---- G10 OOS (purged walk-forward split) ----
    ic_oos = mx.ic_series(factor.loc[oos_idx], fwd.loc[oos_idx], min_names) * direction
    st_oos = mx.ic_stats(ic_oos)
    mean_is_adj = abs(m_is) if np.isfinite(m_is) else 0.0
    if mean_is_adj > 0 and np.isfinite(st_oos["mean"]):
        retention = st_oos["mean"] / mean_is_adj
    else:
        retention = 0.0
    results.append(_res("G10_oos",
                        st_oos["t"] >= g["g10_min_oos_t"] and retention >= g["g10_min_oos_retention"],
                        oos_mean=st_oos["mean"], oos_t=st_oos["t"],
                        retention=retention, oos_n=st_oos["n"]))

    # ---- G11 block bootstrap CI on full-sample adjusted IC ----
    ic_full = mx.ic_series(factor, fwd, min_names) * direction
    lo, hi = mx.block_bootstrap_ci(ic_full, int(g["g11_block"]), int(g["g11_n_boot"]), seed=seed)
    results.append(_res("G11_bootstrap", bool(np.isfinite(lo) and lo > 0), ci_lo=lo, ci_hi=hi))

    verdict = all(r["passed"] for r in results)
    return {"verdict": bool(verdict), "direction": float(direction), "results": results}
