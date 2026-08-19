# Factor Pipeline

單因子驗證流水線：GP 因子挖掘器丟出一個因子式 → G1–G12 門檻全部通過 →
才有資格進入下一階段（與現有因子池的相關性檢查，本專案刻意不包含）。

依賴：`numpy`、`pandas`、`PyYAML`（`pip install -r requirements.txt`）。無其他依賴，Bybit 抓取用標準庫。

## 快速開始

```bash
cd factor_pipeline

# 1) 框架自檢（必做）：有效應要抓到、純噪音要拒絕
python3 sanity_check.py

# 2) 合成資料上測單一因子
python3 run_pipeline.py --expr "reverse(ts_delta(close, 5))" --synthetic --synthetic-effect 0.12

# 3) 真實資料（本地快取優先，缺料才抓 Bybit 並寫回 data_cache/）
python3 run_pipeline.py --expr "rank(ts_std_dev(returns, 20))" --symbols-file symbols_example.txt

# 完全離線（只用本地快取，絕不碰網路）
python3 run_pipeline.py --expr "..." --symbols-file symbols_example.txt --no-refresh

# 4) 跑挖掘器 demo
python3 miner.py
```

## 資料行為

- `data_cache/{SYMBOL}_{tf}.csv` 存在 → 直接讀本地。
- 檔案缺失 → 抓 Bybit v5 kline（linear perp）存回本地。
- 檔案存在但落後兩根 bar 以上且未加 `--no-refresh` → 只補抓缺失段。
- 未收盤的當前 bar 一律丟棄。

## 因子式語法

終端：`open, high, low, close, volume, returns, dollar_volume`\
運算子命名與語義對齊 [phandas-modify](https://github.com/LionHYE/phandas-modify)，
通過流水線的式子可直接 `from phandas import *` 後使用：

- 元素級：`add subtract multiply divide reverse sign s_log_1p`
- 時序（第二參數為整數視窗，僅允許 3/5/10/20/60）：
  `ts_delay ts_delta ts_mean ts_std_dev ts_sum ts_min ts_max ts_rank ts_zscore ts_av_diff ts_decay_linear`
- 橫截面：`rank zscore`

舊名稱（`neg/sub/mul/div/delay/delta/ts_std/cs_rank/cs_zscore/log`）仍可解析，
會自動轉成 phandas 名稱輸出。

例：`rank(divide(ts_delta(close, 5), ts_std_dev(returns, 20)))`

## 門檻（詳見 config.yaml，跑之前鎖死）

| Gate | 檢驗 | 門檻 |
|---|---|---|
| G1 | 覆蓋率 / 非常數 | coverage >= 0.90 |
| G2 | 未來函數（截斷重算比對） | 逐 bar 完全相等 |
| G3 | 複雜度 | nodes <= 12, depth <= 4 |
| G4 | IS rank IC | \|t\| >= 3 |
| G5 | Placebo 零假設（200 隨機因子） | \|t\| > null 99 百分位 |
| G6 | 多視窗衰減 + 因子自相關 | 同號 >= 2/3，autocorr >= 0.7 |
| G7 | beta/size/momentum 中性化 | 保留 >= 50% 且 \|t\| >= 2 |
| G8 | 成本後 L-S 回測（單邊 12.5bps） | Sharpe >= 1 |
| G9 | 視窗 ±30% + 4 regime | 同號且 t 不崩；>= 3/4 regime 同號 |
| G10 | Purged walk-forward OOS | t >= 2 且保留 >= 50% IS |
| G11 | Block bootstrap 95% CI | 下界 > 0 |
| G12 | Registry（runs/registry.jsonl） | 必記錄，不論過或不過 |

## 誠實規則（比程式碼重要）

1. 門檻跑之前鎖死；事後改門檻 = 整條流水線作廢。
2. 失敗後「微調式子再測」= 新候選，M+1，不是同一個因子的修正。
3. registry 記錄所有嘗試（含失敗）；只記成功 = 刪掉多重檢驗校正。
4. 全過只代表單因子合格；進模型前還要過與現有因子池的相關性關卡。

## 真實資料冒煙測試（本機執行，需網路）

```bash
# 1) 編輯板塊清單（sectors/ 下已附 l1 / defi / meme / ai 四份草稿，自行增刪）
# 2) 驗證並產生可交易清單（自動匹配 1000PEPE / SHIB1000 命名、剔除下架與低流動性）
python3 make_universe.py --sector sectors/l1.txt --min-turnover 1e6

# 3) 冒煙：經典因子各跑一次，確認抓料/快取/G1–G12 在真實資料上正常
python3 run_pipeline.py --expr "reverse(rank(ts_sum(returns, 5)))" --symbols-file symbols_l1.txt
python3 run_pipeline.py --expr "rank(ts_std_dev(returns, 20))" --symbols-file symbols_l1.txt

# 4) 正式挖礦：只用 IS 段挖掘，top-20 逐支跑完整 G1–G12
#    驗證階段使用 --no-refresh 固定同一份資料快取
python3 miner.py --symbols-file symbols_l1.txt --no-refresh

# 只產出 top-20、不逐支驗證
python3 miner.py --symbols-file symbols_l1.txt --no-refresh --no-verify
```

真實資料挖掘會將 top 候選保存到 `runs/miner_*.json`，並把每支候選的完整驗證報告與 registry 記錄到 `runs/`。

判讀原則：冒煙測試的重點是流程跑通，不是找 Alpha；經典因子大多 REJECT 是正常現象。
若 G1 失敗，先查幣種上市時間是否太短（歷史不足 750 bar 會被警告）。
