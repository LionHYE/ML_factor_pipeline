"""IC / return metrics. No look-ahead: signal at t -> enter close[t+lag] -> exit close[t+lag+horizon]."""
import numpy as np
import pandas as pd


def forward_returns(close, horizon=1, lag=1):
    entry = close.shift(-int(lag))
    exitp = close.shift(-(int(lag) + int(horizon)))
    return exitp / entry - 1.0


def _rank(a):
    order = a.argsort()
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    return ranks


def ic_series(factor, fwd, min_names=10):
    """Per-date cross-sectional Spearman IC."""
    idx = factor.index
    f = factor.values
    r = fwd.reindex(index=idx, columns=factor.columns).values
    out = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        a, b = f[i], r[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_names:
            continue
        ar = _rank(a[m])
        br = _rank(b[m])
        ar = ar - ar.mean()
        br = br - br.mean()
        d = np.sqrt((ar * ar).sum() * (br * br).sum())
        if d > 0:
            out[i] = float((ar * br).sum() / d)
    return pd.Series(out, index=idx)


def ic_stats(ic):
    v = ic.dropna()
    n = len(v)
    if n < 20:
        return {"n": int(n), "mean": float("nan"), "std": float("nan"), "t": 0.0, "icir": float("nan")}
    mean = float(v.mean())
    std = float(v.std())
    t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    icir = mean / std if std > 0 else float("nan")
    return {"n": int(n), "mean": mean, "std": std, "t": float(t), "icir": float(icir)}


def block_bootstrap_ci(series, block=10, n_boot=1000, alpha=0.05, seed=0):
    v = series.dropna().values
    n = len(v)
    if n < block * 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=k)
        sample = np.concatenate([v[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def annualized_sharpe(rets, periods_per_year=365):
    v = rets.dropna()
    if len(v) < 30 or v.std() == 0:
        return 0.0
    return float(v.mean() / v.std() * np.sqrt(periods_per_year))
