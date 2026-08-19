"""Build a tradable Bybit universe from a sector coin list.

Workflow:
  1. You edit a sector file under sectors/ (one base coin per line, e.g. SOL).
  2. This script queries Bybit v5 instruments-info + tickers (linear USDT perps),
     matches each coin to its actual perp symbol (handles 1000PEPE / SHIB1000
     style renames), drops anything not in 'Trading' status, sorts by 24h
     turnover, and writes symbols_<sector>.txt for run_pipeline.py / miner.py.

Usage:
  python3 make_universe.py --sector sectors/l1.txt
  python3 make_universe.py --sector sectors/meme.txt --min-turnover 5e6 --top 40

Notes:
  - Requires network access to api.bybit.com (run locally, not in a sandbox).
  - A universe below 25 symbols is flagged: cross-sectional stats get noisy.
"""
import argparse
import json
import os
import sys
import urllib.request

API = "https://api.bybit.com/v5/market"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "factor-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
    return data["result"]


def fetch_linear_usdt_instruments():
    """All linear USDT perps with status, keyed by symbol."""
    out, cursor = {}, ""
    while True:
        url = f"{API}/instruments-info?category=linear&limit=1000"
        if cursor:
            url += f"&cursor={cursor}"
        res = _get(url)
        for it in res["list"]:
            if it.get("quoteCoin") == "USDT" and it.get("contractType") == "LinearPerpetual":
                out[it["symbol"]] = it
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    return out


def fetch_turnover():
    """24h turnover (USDT) per symbol."""
    res = _get(f"{API}/tickers?category=linear")
    return {t["symbol"]: float(t.get("turnover24h") or 0.0) for t in res["list"]}


def match_symbol(coin, instruments):
    """Map base coin name to its actual Bybit perp symbol."""
    coin = coin.upper()
    for cand in (f"{coin}USDT", f"1000{coin}USDT", f"{coin}1000USDT",
                 f"10000{coin}USDT", f"1000000{coin}USDT"):
        if cand in instruments:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True, help="sector file, one base coin per line")
    ap.add_argument("--min-turnover", type=float, default=1e6,
                    help="min 24h turnover in USDT (default 1e6)")
    ap.add_argument("--top", type=int, default=0, help="keep only top N by turnover (0 = all)")
    ap.add_argument("--out", default="", help="output file (default symbols_<sector>.txt)")
    args = ap.parse_args()

    coins = [ln.strip().upper() for ln in open(args.sector)
             if ln.strip() and not ln.startswith("#")]
    print(f"[universe] {len(coins)} coins in {args.sector}")

    instruments = fetch_linear_usdt_instruments()
    turnover = fetch_turnover()

    rows, missing, halted, illiquid = [], [], [], []
    for c in coins:
        sym = match_symbol(c, instruments)
        if sym is None:
            missing.append(c)
            continue
        if instruments[sym].get("status") != "Trading":
            halted.append(sym)
            continue
        tv = turnover.get(sym, 0.0)
        if tv < args.min_turnover:
            illiquid.append(f"{sym}({tv/1e6:.1f}M)")
            continue
        rows.append((sym, tv))

    rows.sort(key=lambda r: -r[1])
    if args.top:
        rows = rows[: args.top]

    name = os.path.splitext(os.path.basename(args.sector))[0]
    out_path = args.out or f"symbols_{name}.txt"
    with open(out_path, "w") as f:
        for sym, _ in rows:
            f.write(sym + "\n")

    print(f"[universe] kept {len(rows)} -> {out_path}")
    for sym, tv in rows:
        print(f"  {sym:<16} 24h turnover {tv/1e6:>10.1f}M")
    if missing:
        print(f"[warn] no Bybit USDT perp found: {', '.join(missing)}")
    if halted:
        print(f"[warn] not in Trading status: {', '.join(halted)}")
    if illiquid:
        print(f"[warn] below turnover floor: {', '.join(illiquid)}")
    if len(rows) < 25:
        print(f"[warn] only {len(rows)} symbols (<25): cross-sectional IC will be noisy. "
              f"Add coins or lower --min-turnover.")


if __name__ == "__main__":
    main()
