# IHCA RF / LSTM Hybrid

以心臟科病人的生命徵象與檢驗值預測院內心跳停止（IHCA），嘗試結合 Random Forest 與 LSTM。

本 repo 從 [CardiacPatientData_LSTM](https://github.com/oldpincon/CardiacPatientData_LSTM)（`experiment` 分支 commit `97601ba`）分出，只帶必要的程式與參考結果；舊版程式、不可報告的結果與完整實驗歷程留在原 repo。

## 資料
`CardiacPatientData.csv`：公開資料集，112 位病人、19 位 IHCA，每列為一次紀錄，沒有時間戳記。`Outcome == 0` 代表 IHCA。

## 原 repo 的結論（起點）
3 組外層切分的平均 pooled AUC：

| 分析 | LSTM v4G | RF | LR | NEWS2 |
|---|---|---|---|---|
| primary（112 人） | 0.826 | 0.881 | 0.866 | 0.855 |
| no_labs（不用檢驗值） | 0.858 | 0.866 | 0.819 | 0.855 |
| min10（紀錄 ≥ 10 筆） | 0.812 | 0.875 | 0.905 | 0.848 |

- LSTM 數值上低於 RF，所有比較在 Holm 校正後都不顯著。
- LSTM 需要手動給「與上一筆的差」才明顯進步；同樣資訊對 RF 沒有幫助（摘要特徵已含趨勢）。
- 本 repo 要回答：RF 與 LSTM 結合能不能比單獨使用更好。

## 檔案
| 檔案 | 用途 |
|---|---|
| `LSTM_nested_cv_v2.py` | 資料載入與前處理（不合理值 → NaN → 病人內 LOCF → 訓練折中位數）、外層切分、共用函式 |
| `baselines_nested_v2.py` | RF / LR（摘要特徵，`--delta` 加變化量特徵）、EWS 分數 |
| `LSTM_v3.py` | LSTM 各版本，`--variant G` 為目前最佳 LSTM |
| `early_warning_scores.py`、`evaluate_ews_baseline.py`、`test_ews.py` | MEWS / NEWS / NEWS2 計分與單元測試 |
| `delong_v2.py`、`delong_v4.py` | 配對 DeLong 檢定、Holm 校正 |
| `results_v2_*`、`results_baselines_v2_*`、`results_v4G_*` | 原 repo 的參考結果（OOF 預測），用來和 hybrid 逐病人比較 |

## 環境
Windows，執行 `setup_env.bat` 建立 `.venv`（tensorflow 2.21、numpy、pandas、scikit-learn、optuna）。中文輸出需要 `set PYTHONUTF8=1`。

```
python test_ews.py
python LSTM_v3.py --variant G --analysis primary --trials 2 --quick   # 確認環境能跑
```
