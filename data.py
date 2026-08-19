"""Data layer: local CSV cache first; fetch Bybit v5 kline only for missing/stale data.

Cache layout: {data_dir}/{SYMBOL}_{timeframe}.csv  (UTC index; open,high,low,close,volume,turnover)
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

BYBIT_KLINE = "https://api.bybit.com/v5/market/kline"
INTERVALS = {"1d": "D", "4h": "240", "1h": "60", "15m": "15"}
BAR_MS = {"1d": 86400000, "4h": 14400000, "1h": 3600000, "15m": 900000}
COLS = ["open", "high", "low", "close", "volume", "turnover"]


def fetch_bybit_ohlcv(symbol, timeframe="1d", start_ms=None, end_ms=None,
                      category="linear", pause=0.15):
    interval = INTERVALS[timeframe]
    end_ms = int(end_ms or time.time() * 1000)
    start_ms = int(start_ms if start_ms is not None else end_ms - 1500 * BAR_MS[timeframe])
    rows = {}
    cur_end = end_ms
    while True:
        params = {"category": category, "symbol": symbol, "interval": interval,
                  "start": start_ms, "end": cur_end, "limit": 1000}
        url = BYBIT_KLINE + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit error for {symbol}: {payload.get('retMsg')}")
        batch = payload["result"]["list"]  # newest first
        if not batch:
            break
        for r in batch:
            rows[int(r[0])] = [float(r[1]), float(r[2]), float(r[3]),
                               float(r[4]), float(r[5]), float(r[6])]
        oldest = min(int(r[0]) for r in batch)
        if oldest <= start_ms or len(batch) < 1000:
            break
        cur_end = oldest - 1
        time.sleep(pause)
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=COLS).sort_index()
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df


def load_or_fetch(symbol, timeframe="1d", data_dir="data_cache", refresh=True,
                  lookback_bars=1500, category="linear"):
    """Local file first. Fetch from Bybit only when the file is missing or stale,
    then persist -- so the network is hit only on first run or when updating."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
    now_ms = int(time.time() * 1000)
    df = None
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df is None or df.empty:
        df = fetch_bybit_ohlcv(symbol, timeframe,
                               now_ms - lookback_bars * BAR_MS[timeframe], now_ms, category)
        if df.empty:
            raise RuntimeError(f"no data for {symbol}")
        df.to_csv(path)
    elif refresh:
        last_ms = int(df.index[-1].timestamp() * 1000)
        if now_ms - last_ms > 2 * BAR_MS[timeframe]:
            new = fetch_bybit_ohlcv(symbol, timeframe, last_ms + 1, now_ms, category)
            if not new.empty:
                df = pd.concat([df, new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.to_csv(path)
    # drop possibly-unfinished current bar
    if len(df):
        last_ms = int(df.index[-1].timestamp() * 1000)
        if now_ms - last_ms < BAR_MS[timeframe]:
            df = df.iloc[:-1]
    return df


def build_panel(symbols, timeframe="1d", data_dir="data_cache", refresh=True, **kw):
    per = {}
    for s in symbols:
        try:
            d = load_or_fetch(s, timeframe, data_dir, refresh, **kw)
            if len(d):
                per[s] = d
        except Exception as e:
            print(f"[data] skip {s}: {e}")
    if not per:
        raise RuntimeError("no symbols loaded")
    print(f"[data] loaded {len(per)}/{len(symbols)} symbols")
    fields = {}
    for f in ["open", "high", "low", "close", "volume"]:
        fields[f] = pd.DataFrame({s: d[f] for s, d in per.items()}).sort_index()
    close = fields["close"]
    fields["returns"] = close.pct_change()
    fields["dollar_volume"] = fields["volume"] * close
    return fields
