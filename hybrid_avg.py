"""
hybrid_avg.py — H1：RF / v4G 的排名平均（見 Hybrid_預先指定.md）
====================================================================
每個外層測試折內，把每個模型的預測轉成百分位排名（rank / n_test），再取兩個模型的平均。
不用標籤、不 fit 任何東西；只用原 repo 已有的外層 OOF 預測。

  Hybrid_RF_v4G       RF 與 v4G 的排名平均（主要）
  Hybrid_RF_LR        RF 與 LR 的排名平均（對照組）
  Hybrid_RF_v4G_prob  RF 與 v4G 直接平均機率（敏感度分析，只描述）

執行：python hybrid_avg.py
輸出：results_hybrid_<analysis>_oof.csv、results_hybrid_summary.csv
"""

import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ANALYSES = ["primary", "no_labs", "min10"]
REPEATS = [1, 2, 3]
BASE = ["RF", "LR", "v4G"]
HYBRIDS = {"Hybrid_RF_v4G": ("RF", "v4G"), "Hybrid_RF_LR": ("RF", "LR")}


def load_wide(analysis):
    """回傳長表：repeat、fold、ID、y 加上 RF / LR / v4G 的機率；並檢查三者的折相同。"""
    base = pd.read_csv(f"results_baselines_v2_{analysis}_oof.csv")
    g = pd.read_csv(f"results_v4G_{analysis}_oof.csv").assign(model="v4G")
    oof = pd.concat([base[base.model.isin(["RF", "LR"])], g[base.columns]])
    oof = oof[oof.repeat.isin(REPEATS)]
    key = ["repeat", "ID"]
    wide = oof.pivot_table(index=key, columns="model", values="proba")
    folds = oof.pivot_table(index=key, columns="model", values="fold")
    ys = oof.pivot_table(index=key, columns="model", values="y")
    assert wide.notna().all().all(), "有病人缺少某模型的預測"
    assert (folds.nunique(axis=1) == 1).all(), "RF / LR / v4G 的外層折不一致"
    assert (ys.nunique(axis=1) == 1).all(), "標籤不一致"
    wide = wide[BASE]
    wide["fold"] = folds.iloc[:, 0].astype(int)
    wide["y"] = ys.iloc[:, 0].astype(int)
    return wide.reset_index()


def add_hybrids(wide):
    pct = wide.groupby(["repeat", "fold"])[BASE].transform(
        lambda s: rankdata(s, method="average") / len(s))
    for name, (a, b) in HYBRIDS.items():
        wide[name] = (pct[a] + pct[b]) / 2
    wide["Hybrid_RF_v4G_prob"] = (wide.RF + wide.v4G) / 2
    return wide


def main():
    models = BASE + list(HYBRIDS) + ["Hybrid_RF_v4G_prob"]
    rows = []
    for a in ANALYSES:
        wide = add_hybrids(load_wide(a))
        wide.to_csv(f"results_hybrid_{a}_oof.csv", index=False)
        for m in models:
            aucs = [roc_auc_score(g.y, g[m]) for _, g in wide.groupby("repeat")]
            rows.append({"analysis": a, "model": m, **{f"auc_r{r}": v for r, v in zip(REPEATS, aucs)},
                         "auc_mean": sum(aucs) / len(aucs)})
        print(f"[{a}] 病人 {wide.ID.nunique()}，IHCA {wide[wide.repeat == 1].y.sum()}")
    s = pd.DataFrame(rows)
    s.to_csv("results_hybrid_summary.csv", index=False)
    print("\n各 repeat 的 pooled AUC：")
    print(s.round(3).to_string(index=False))
    print("\n輸出：results_hybrid_<analysis>_oof.csv、results_hybrid_summary.csv")


if __name__ == "__main__":
    main()
