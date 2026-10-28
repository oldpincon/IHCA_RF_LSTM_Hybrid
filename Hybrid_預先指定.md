# RF / LSTM Hybrid：預先指定（2026-10-29，跑之前寫定）

## 要回答的問題
RF 與 LSTM v4G 結合後，AUC 能不能高於單獨的 RF、LR 或 LSTM？

## 事先要說明的事
- RF、LR、v4G 各自的 OOF 預測已經看過（見 `README.md`），沒看過的只有「結合之後」的結果。
  所以本文件能保證的是：結合的方法、比較組與解讀方式在看結合結果之前就寫定，事後不改。
- 112 人、19 位 IHCA，兩個 AUC 0.85 左右的模型要差 0.05 以上才可能顯著，預期大多不顯著。
  因此另外寫定一個「數值上一致」的描述性標準（見最後一節）。

## 基礎模型（全部沿用原 repo 的結果，不重新調參）
| 名稱 | 來源 |
|---|---|
| RF | `results_baselines_v2_<analysis>_oof.csv`（model == RF） |
| LR | 同上（model == LR） |
| v4G | `results_v4G_<analysis>_oof.csv`（小 LSTM + 5 種子平均 + 變化量） |

三者的外層切分都是 `StratifiedKFold(5, shuffle=True, random_state=42 + repeat)`，repeat 1–3，
每位病人在每個 repeat 都有三個模型的外層預測。程式會先檢查三者每位病人的 fold 編號相同。

## 為什麼不能直接把機率平均
三個模型的機率範圍差很多（未看標籤，只看分佈的 10% / 50% / 90% 分位數，primary）：
v4G 0.004 / 0.041 / 0.782，RF 0.055 / 0.19 / 0.614，LR 0.082 / 0.279 / 0.679。
直接平均時，範圍拉得比較開的 v4G 會不成比例地主導結果。所以主分析先把每個模型的分數轉成排名再平均。

## H1（主分析）：排名平均，不需要訓練
每個外層測試折內，把每個模型的預測轉成百分位排名（`rank / n_test`），再取兩個模型的平均。
- 不用標籤、不 fit 任何東西，不會洩漏。限制：排名要用到同一折其他測試病人的預測（transductive），
  實際上線時不能這樣做，所以 H1 是「結合有沒有潛力」的檢查，H2 才是可以部署的版本。
- `Hybrid_RF_v4G`：RF 與 v4G 的排名平均（主要的 hybrid）
- `Hybrid_RF_LR`：RF 與 LR 的排名平均（對照組：如果 RF 加上 LR 也一樣進步，就代表進步來自「平均兩個模型」本身，不是 LSTM 特別有幫助）
- 只報告 AUC。H1 沒有內層預測可以定門檻，所以不報告 recall / precision。
- 敏感度分析（只描述，不檢定）：直接平均機率的 `Hybrid_RF_v4G_prob`。

### H1 比較（`delong_hybrid.py` → `results_delong_hybrid.csv`）
每個分析、每個 repeat（1–3）比 4 組，共 12 組，每個分析內 Holm 校正：
1. Hybrid_RF_v4G vs RF
2. Hybrid_RF_v4G vs v4G
3. Hybrid_RF_v4G vs LR
4. Hybrid_RF_v4G vs Hybrid_RF_LR

## H2（次分析）：用內層 OOF 決定權重，可以部署
H2 要重跑 v4G 才拿得到內層預測，需要數小時以上，H1 跑完後再決定要不要跑。不論跑或不跑，都會在結果中寫明。
- 需要的程式修改：`LSTM_v3.py` 加一個選項，另外輸出最佳 trial 的內層 OOF 預測；預設行為不變。
  RF 的內層 OOF 用相同的內層折（`StratifiedKFold(3, shuffle=True, random_state=seed*10+f)`，
  RF 和 LSTM 的內層折只由標籤決定，所以兩者相同）重新計算。
- 刻度：每個模型的預測轉成「在該模型內層 OOF 預測中的百分位」，只用訓練病人，部署時也能算。
- 權重：`w ∈ {0, 0.1, …, 1.0}`，混合分數 = `w × LSTM + (1 − w) × RF`，選內層 OOF AUC 最高的 w；
  同分（差 < 0.001）時選最接近 0.5 的。門檻 = 混合後內層 OOF 的 Youden 指數。
- 不做 stacking：只有兩個輸入時，第二層 LR 和加權平均幾乎一樣，只是多一個參數。
- 已知的偏誤：LSTM 的內層 OOF 來自 40 個 trial 中挑出的最好的一個，RF 只有 9 組 grid，
  所以 LSTM 的內層 OOF 比較樂觀，選出的 w 可能偏向 LSTM。
- H2 比較：H2 vs RF、H2 vs v4G、H2 vs LR，每個 repeat 3 組，每個分析 9 組，Holm 校正。
  輸出 `results_hybrid_h2_<analysis>_*`。

## 不另外做的
- 「LSTM 輸出 + RF 摘要特徵」的單一模型：原 repo 的 v3C 做過（primary 0.789、no_labs 0.853、
  min10 0.862），沒有贏過 RF，不重做。

## 事先說好的解讀
- 三個分析（primary、no_labs、min10）都要報告，以 3 個 repeat 的 pooled AUC 平均為主要數字。
- **顯著進步**：Hybrid_RF_v4G vs RF 在某個分析的 3 個 repeat 中，Holm 校正後都顯著。
- **數值上一致較高**：在三個分析中，Hybrid_RF_v4G 的平均 AUC 都高於 RF 且都高於 Hybrid_RF_LR。
  只符合這個標準時，只能說「有一致的趨勢，但這個樣本量無法確認」。
- Hybrid_RF_v4G 不比 Hybrid_RF_LR 好 → 進步主要來自平均兩個模型，LSTM 沒有提供 RF/LR 沒有的資訊。
- 這個比較不涉及挑選版本，所以不需要另外確認：3 個 repeat 本身就是完整結果，不挑 repeat。

## 輸出
- `hybrid_avg.py` → `results_hybrid_<analysis>_oof.csv`、`results_hybrid_summary.csv`
- `delong_hybrid.py` → `results_delong_hybrid.csv`

## H1 結果（2026-10-29，3 個 repeat 的 pooled AUC 平均）
| 分析 | RF | LR | v4G | Hybrid_RF_v4G | Hybrid_RF_LR | Hybrid_RF_v4G_prob |
|---|---|---|---|---|---|---|
| primary | 0.881 | 0.866 | 0.826 | 0.884 | 0.897 | 0.867 |
| no_labs | 0.866 | 0.819 | 0.858 | 0.888 | 0.878 | 0.863 |
| min10 | 0.875 | 0.905 | 0.812 | 0.887 | 0.903 | 0.860 |

- 顯著進步：沒有。Hybrid_RF_v4G vs RF 的 9 組比較在 Holm 校正後都不顯著（差 −0.028 到 +0.038）。
- 數值上一致較高：**不符合**。Hybrid_RF_v4G 在三個分析都略高於 RF，但只有 no_labs 高於 Hybrid_RF_LR；
  primary 與 min10 都是 Hybrid_RF_LR 比較高。
- 36 組中唯一 Holm 校正後顯著的：min10 repeat 1，Hybrid_RF_v4G 高於 v4G（+0.060，p_holm = 0.022）。
- 解讀：hybrid 比 RF 略高，主要來自「平均兩個模型」本身，換成 LR 也一樣甚至更好；
  只有在不用檢驗值時（v4G 本身最接近 RF），LSTM 才提供了 LR 沒有的資訊。
- 敏感度分析：直接平均機率（Hybrid_RF_v4G_prob）在三個分析都比排名平均低，也比 RF 低，
  與事先的判斷一致（v4G 的機率範圍較大，會主導平均）。
- H2：H1 結果顯示結合的進步主要來自平均兩個模型，決定不執行（2026-10-29）。
