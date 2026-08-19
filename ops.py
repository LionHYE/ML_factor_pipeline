"""Causal operators for factor expressions.

Operator names and semantics follow phandas-modify
(https://github.com/LionHYE/phandas-modify), so any expression that passes
this pipeline can be pasted into phandas directly:

    from phandas import *
    factor = reverse(ts_delta(close, 5))

All time-series ops are strictly backward-looking (rolling / shift only).
All inputs/outputs are pandas DataFrames shaped (dates x symbols).
"""
import numpy as np
import pandas as pd

EPS = 1e-9

# ---- arithmetic (phandas: add / subtract / multiply / divide) ----
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    # phandas: division by 0 -> NaN
    y = y.where(y.abs() > EPS)
    return x / y

# ---- elementary math (phandas: reverse / sign / s_log_1p) ----
def reverse(x):
    return -x

def sign(x):
    return np.sign(x)

def s_log_1p(x):
    # phandas: sign(x) * ln(1 + |x|)
    return np.log1p(x.abs()) * np.sign(x)

# ---- cross-sectional (phandas: rank / zscore) ----
def rank(x):
    return x.rank(axis=1, pct=True)

def zscore(x):
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd + EPS, axis=0)

# ---- time-series, causal (phandas: ts_*) ----
def ts_delay(x, w):
    return x.shift(int(w))

def ts_delta(x, w):
    return x - x.shift(int(w))

def ts_mean(x, w):
    return x.rolling(int(w), min_periods=int(w)).mean()

def ts_std_dev(x, w):
    return x.rolling(int(w), min_periods=int(w)).std()

def ts_sum(x, w):
    return x.rolling(int(w), min_periods=int(w)).sum()

def ts_min(x, w):
    return x.rolling(int(w), min_periods=int(w)).min()

def ts_max(x, w):
    return x.rolling(int(w), min_periods=int(w)).max()

def ts_rank(x, w):
    w = int(w)
    def _r(a):
        return (a[:-1] < a[-1]).mean()
    return x.rolling(w, min_periods=w).apply(_r, raw=True)

def ts_zscore(x, w):
    w = int(w)
    mu = x.rolling(w, min_periods=w).mean()
    sd = x.rolling(w, min_periods=w).std()
    return (x - mu) / (sd + EPS)

def ts_av_diff(x, w):
    w = int(w)
    return x - x.rolling(w, min_periods=w).mean()

def ts_decay_linear(x, w):
    w = int(w)
    wts = np.arange(1, w + 1, dtype=float)
    wsum = wts.sum()
    def _d(a):
        return (a * wts).sum() / wsum
    return x.rolling(w, min_periods=w).apply(_d, raw=True)

BINARY = ["add", "subtract", "multiply", "divide"]
UNARY = ["reverse", "sign", "s_log_1p", "rank", "zscore"]
TS = ["ts_delay", "ts_delta", "ts_mean", "ts_std_dev", "ts_sum", "ts_min",
      "ts_max", "ts_rank", "ts_zscore", "ts_av_diff", "ts_decay_linear"]
TERMINALS = ["open", "high", "low", "close", "volume", "returns", "dollar_volume"]
WINDOWS = [3, 5, 10, 20, 60]

OPS = {name: obj for name, obj in list(globals().items())
       if callable(obj) and name in BINARY + UNARY + TS}
