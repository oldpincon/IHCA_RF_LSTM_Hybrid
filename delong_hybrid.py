"""
delong_hybrid.py — H1 的配對 DeLong 檢定（見 Hybrid_預先指定.md）
每個分析、每個 repeat（1–3）：Hybrid_RF_v4G vs RF、v4G、LR、Hybrid_RF_LR，
每個分析 12 組 Holm 校正。

需要先跑 hybrid_avg.py。
執行：python delong_hybrid.py
輸出：results_delong_hybrid.csv
"""

import pandas as pd

from delong_v2 import holm
from delong_v4 import paired

ANALYSES = ["primary", "no_labs", "min10"]
H = "Hybrid_RF_v4G"
PAIRS = [(H, "RF"), (H, "v4G"), (H, "LR"), (H, "Hybrid_RF_LR")]


def compare(analysis):
    oof = pd.read_csv(f"results_hybrid_{analysis}_oof.csv")
    rows = []
    for r, g in oof.groupby("repeat"):
        g = g.set_index("ID")
        wide = g[[H, "RF", "v4G", "LR", "Hybrid_RF_LR"]]
        rows += paired(g.y.values, wide, PAIRS, {"analysis": analysis, "repeat": r})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def main():
    df = pd.concat([compare(a) for a in ANALYSES], ignore_index=True)
    df.to_csv("results_delong_hybrid.csv", index=False)
    cols = ["repeat", "reference", "auc_variant", "auc_reference", "diff",
            "diff_ci_low", "diff_ci_high", "p", "p_holm"]
    for a, g in df.groupby("analysis", sort=False):
        print(f"\n===== {a}（n={g.n_patients.iloc[0]}，IHCA {g.n_ihca.iloc[0]}）=====")
        print(g[cols].round(3).to_string(index=False))
    print("\n輸出：results_delong_hybrid.csv")


if __name__ == "__main__":
    main()
