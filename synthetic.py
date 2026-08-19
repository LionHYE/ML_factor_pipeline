"""Synthetic OHLCV panel with an optional injected cross-sectional reversal effect.

The injected effect is timed so it is capturable only at the pipeline's
execution lag (signal at t -> enter close[t+1] -> exit close[t+2]).
"""
import numpy as np
import pandas as pd


def make_synthetic(n_symbols=30, n_days=750, effect=0.0, lookback=5, seed=0):
    rng = np.random.default_rng(seed)
    logp = np.zeros((n_days, n_symbols))
    logp[0] = rng.normal(4.5, 0.5, n_symbols)
    for t in range(1, n_days):
        common = rng.normal(0.0003, 0.015)
        idio = rng.normal(0, 0.02, n_symbols)
        drift = np.zeros(n_symbols)
        if effect > 0 and t - 2 - lookback >= 0:
            sig = logp[t - 2] - logp[t - 2 - lookback]
            z = (sig - sig.mean()) / (sig.std() + 1e-9)
            drift = -effect * 0.02 * z
        logp[t] = logp[t - 1] + common + idio + drift
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    cols = [f"SYM{i:02d}USDT" for i in range(n_symbols)]
    close = pd.DataFrame(np.exp(logp), index=idx, columns=cols)
    open_ = close.shift(1) * np.exp(rng.normal(0, 0.002, (n_days, n_symbols)))
    wick = np.abs(rng.normal(0, 0.004, (n_days, n_symbols)))
    high = pd.DataFrame(np.fmax(open_.values, close.values) * np.exp(wick), index=idx, columns=cols)
    low = pd.DataFrame(np.fmin(open_.values, close.values) * np.exp(-wick), index=idx, columns=cols)
    volume = pd.DataFrame(np.exp(rng.normal(10, 1, (n_days, n_symbols))), index=idx, columns=cols)
    fields = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    fields["returns"] = close.pct_change()
    fields["dollar_volume"] = volume * close
    return fields
