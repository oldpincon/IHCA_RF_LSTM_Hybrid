# CLAUDE.md

本檔提供 Claude Code 在此 repo 工作時的背景資訊。

## 專案目的
結合 RF 與 LSTM 預測 IHCA。從 `CardiacPatientData_LSTM` 分出（commit `97601ba`），起點與結論見 `README.md`。資料只有 112 位病人、19 位 IHCA，**評估方法的正確性比模型分數重要**。

## 資料
- `CardiacPatientData.csv`：公開資料集，可以公開在 GitHub。同一病人內列順序視為時間順序，沒有時間戳記，所以預警時效只能用截斷測試（視窗往前移 k 筆）間接檢查。
- 標籤：`Outcome == 0` 代表 IHCA，程式內一律反轉成 `IHCA = 1 - Outcome`。
- 欄位 `Ceratinine` 是原始 CSV 的拼字，不要改。`BT` 單位為 °F。
- 每位病人取最後 `MAX_LEN = 10` 筆，padding 用 `'pre'`。

## 沿用的程式（不要改動行為）
- `LSTM_nested_cv_v2.py`：`load_patients()`、`FoldPreprocessor`、`PLAUSIBLE`、`TRUNC_KS`、`shifted_window`、`truncation_summary`、`youden_threshold`、`bootstrap_ci`。
- `baselines_nested_v2.py`：`summarize()`（RF / LR 摘要特徵）、`MODELS`、`EWS`、`ews_scores()`。
- `LSTM_v3.py`：`VARIANTS["G"]`、`Encoder`、`build_small`、`inner_search`。
- `results_*` 參考結果是原 repo 產生的，不要覆蓋；新結果用新的檔名前綴。

## 評估原則
- Nested CV：外層 `StratifiedKFold(5, shuffle=True, random_state=42 + repeat)`，所有模型用相同的外層折才能逐病人比較。
- 任何需要 fit 的步驟（中位數填補、標準化、調參、門檻、hybrid 的權重或 stacking 模型）只能用該折的訓練病人；stacking 的第二層必須用內層 OOF 預測訓練。
- 三個預先指定的分析都要跑、都要報告：`primary`、`no_labs`、`min10`。
- 新版本先寫預先指定文件並 commit，再跑；依事先規則挑選後，要用新的外層切分（repeat 2、3）確認。
- 比較用配對 DeLong，每個分析內 Holm 校正。
- LSTM 在 Windows 上要用 `--one-fold`（每折一個新程序），否則 TF 2.21 記憶體會一直累積。

## 程式風格
程式註解與輸出訊息使用繁體中文。
