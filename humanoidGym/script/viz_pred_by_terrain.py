import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infer_terrain_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:-len("_terrain")] if stem.endswith("_terrain") else stem

def load_and_tag(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert "pred_class" in df.columns, f"{csv_path} 缺少列 pred_class"
    df["terrain"] = infer_terrain_name(csv_path)
    return df[["pred_class", "terrain"]]

def summarize(df_all: pd.DataFrame, top_k: int = 20):
    # 原始频次表（行=terrain, 列=pred_class）
    freq = (df_all.groupby(["terrain", "pred_class"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index(axis=0))
    # 占比（行归一化）
    row_sum = freq.sum(axis=1).replace(0, 1)
    pct = freq.div(row_sum, axis=0)

    # 选取出现最多的 Top-K 类别（总体）
    total_counts = freq.sum(axis=0).sort_values(ascending=False)
    keep_classes = list(total_counts.head(top_k).index)

    # 只保留 Top-K 列，其他聚合为 “others”
    freq_top = freq[keep_classes].copy()
    pct_top = pct[keep_classes].copy()
    if len(total_counts) > top_k:
        freq_top["others"] = (freq.drop(columns=keep_classes)
                                   .sum(axis=1))
        pct_top["others"] = (pct.drop(columns=keep_classes)
                                 .sum(axis=1))

    return freq, pct, freq_top, pct_top, keep_classes

def plot_topk_bars_per_terrain(pct_top: pd.DataFrame, save_prefix: str):
    # 为每个 terrain 单独画一张 Top-K 概率柱状图
    os.makedirs("figs", exist_ok=True)
    for terrain, row in pct_top.iterrows():
        vals = row.sort_values(ascending=False)
        plt.figure(figsize=(8,4))
        plt.bar([str(c) for c in vals.index], vals.values)
        plt.title(f"Predicted class distribution — {terrain}")
        plt.xlabel("pred_class"); plt.ylabel("percentage")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = f"figs/{save_prefix}_{terrain}_topk_bar.png"
        plt.savefig(out, dpi=220); plt.close()
        print(f"💾 saved {out}")

def plot_stacked_bars(pct_top: pd.DataFrame, save_prefix: str):
    # 各地形一张图的堆叠条形（行是地形，列是类）
    os.makedirs("figs", exist_ok=True)
    order = pct_top.index.tolist()
    cols = pct_top.columns.tolist()
    bottom = np.zeros(len(order))
    plt.figure(figsize=(max(8, len(order)*1.2), 5))
    for c in cols:
        plt.bar(order, pct_top[c].values, bottom=bottom, label=str(c))
        bottom += pct_top[c].values
    plt.title("Stacked distribution by terrain (Top-K + others)")
    plt.ylabel("percentage"); plt.xticks(rotation=30, ha="right")
    if len(cols) <= 20:
        plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    out = f"figs/{save_prefix}_stacked.png"
    plt.savefig(out, dpi=220); plt.close()
    print(f"💾 saved {out}")

def plot_heatmap(pct_top: pd.DataFrame, save_prefix: str):
    # 热力图：行=terrain，列=pred_class（Top-K+others），值=占比
    os.makedirs("figs", exist_ok=True)
    plt.figure(figsize=(max(8, 0.6*len(pct_top.columns)+4), max(4, 0.35*len(pct_top.index)+2)))
    im = plt.imshow(pct_top.values, aspect='auto', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04, label="percentage")
    plt.yticks(range(len(pct_top.index)), pct_top.index)
    plt.xticks(range(len(pct_top.columns)), [str(c) for c in pct_top.columns], rotation=45, ha="right")
    plt.title("Heatmap — predicted class percentage by terrain")
    plt.tight_layout()
    out = f"figs/{save_prefix}_heatmap.png"
    plt.savefig(out, dpi=220); plt.close()
    print(f"💾 saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Visualize predicted terrain classes across files")
    ap.add_argument("--csv", type=str, nargs="+", required=True,
                    help="CSV 文件列表（含 pred_class 列），例如：discrete_obstacles_terrain.csv pyramid_sloped_terrain.csv ...")
    ap.add_argument("--top_k", type=int, default=20, help="展示最频繁的 Top-K 类别，其余合并为 others")
    ap.add_argument("--out_prefix", type=str, default="pred_by_terrain", help="输出前缀（图像放 figs/ 下）")
    ap.add_argument("--save_summary", action="store_true", help="导出 summary_pred_class.csv")
    args = ap.parse_args()

    # 读取与合并
    dfs = [load_and_tag(p) for p in args.csv]
    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # ====== 新增：过滤掉不想要的类别 ======
    drop_classes = [84, 81]   # 想要去掉的 pred_class 列表
    before = len(df_all)
    df_all = df_all[~df_all["pred_class"].isin(drop_classes)]
    after = len(df_all)
    print(f"Filtered pred_class {drop_classes}: removed {before - after} rows")

    print(f"Loaded rows: {len(df_all)}, terrains: {df_all['terrain'].nunique()}")

    # 统计
    freq, pct, freq_top, pct_top, keep_classes = summarize(df_all, top_k=args.top_k)
    print(f"Top-{args.top_k} classes: {keep_classes}")

    # 导出统计（可选）
    if args.save_summary:
        out_csv = f"{args.out_prefix}_summary_pred_class.csv"
        merged = freq_top.copy()
        merged.columns = [f"count_{c}" for c in merged.columns]
        pct_named = pct_top.copy()
        pct_named.columns = [f"pct_{c}" for c in pct_named.columns]
        summary = pd.concat([merged, pct_named], axis=1)
        summary.to_csv(out_csv)
        print(f"💾 saved {out_csv}")

    # 作图
    plot_topk_bars_per_terrain(pct_top, args.out_prefix)
    plot_stacked_bars(pct_top, args.out_prefix)
    plot_heatmap(pct_top, args.out_prefix)


if __name__ == "__main__":
    main()
