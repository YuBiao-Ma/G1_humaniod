# visualize_multi_latents.py
# 读取多个 *_terrain.csv → 合并 → 可视化 PCA / t-SNE
# 支持按 terrain / pred_class / terrain+pred 着色，并输出可分性指标
# 新增：t-SNE 预处理（Standard/Robust + PCA/whiten）与网格搜索（perplexity/EE/LR）

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visualize multiple terrain latents")
    p.add_argument("--csv", type=str, nargs="+", required=True,
                   help="List of CSV files. e.g. discrete_obstacles_terrain.csv pyramid_sloped_terrain.csv ...")
    p.add_argument("--out_prefix", type=str, default=None, help="Prefix for output images/files")
    p.add_argument("--sample", type=int, default=30000, help="Global subsample size (-1 for all)")
    p.add_argument("--balance_per_terrain", type=int, default=None,
                   help="If set, per-terrain max samples for balanced viz (e.g., 10000)")
    p.add_argument("--color_by", type=str, default="terrain",
                   choices=["terrain", "pred", "terrain+pred"],
                   help="Color scheme for scatter plots")
    p.add_argument("--pca", action="store_true", help="Do PCA 2D scatter")
    p.add_argument("--pca3d", action="store_true", help="Do PCA 3D scatter")
    p.add_argument("--tsne", action="store_true", help="Do t-SNE 2D scatter")
    p.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity (single run)")
    p.add_argument("--tsne_iter", type=int, default=1000, help="t-SNE iterations")
    p.add_argument("--no_show", action="store_true", help="Only save figures, do not show")

    # 预处理 / t-SNE 网格搜索（新增）
    p.add_argument("--preprocess", type=str, default="standard",
                   choices=["none", "standard", "robust"], help="Scaling before PCA/t-SNE")
    p.add_argument("--whiten", action="store_true", help="Use PCA whitening before t-SNE")
    p.add_argument("--pca_var_keep", type=float, default=0.95,
                   help="Keep variance ratio for PCA before t-SNE (0~1)")
    p.add_argument("--pca_max_dim", type=int, default=50, help="Max PCA dims before t-SNE")

    p.add_argument("--tsne_grid", action="store_true",
                   help="Grid-search t-SNE params to maximize 2D silhouette")
    p.add_argument("--perp_grid", type=str, default="20,30,40,60,80",
                   help="Perplexity grid (comma-separated)")
    p.add_argument("--ee_grid", type=str, default="12,24,36",
                   help="Early exaggeration grid (comma-separated)")
    p.add_argument("--lr_grid", type=str, default="200,600,1000",
                   help="Learning rate grid (comma-separated)")
    p.add_argument("--tsne_angle", type=float, default=0.3,
                   help="Barnes-Hut angle; lower=more accurate, slower")

    return p.parse_args()


# -------------------- Utils --------------------
def _infer_terrain_name(path):
    """从文件名推断 terrain 名称：xxx_terrain.csv → xxx"""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("_terrain"):
        return stem[:-len("_terrain")]
    return stem


def _load_and_tag(csv_path):
    df = pd.read_csv(csv_path)
    assert "pred_class" in df.columns, f"{csv_path} 缺少列 pred_class"
    df["terrain"] = _infer_terrain_name(csv_path)
    return df


def _subsample_indices(n, k, seed=0):
    if k is None or k < 0 or k >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=k, replace=False)


def _balanced_by_terrain(df, per_terrain, seed=0):
    if per_terrain is None:
        return df
    chunks = []
    for _, g in df.groupby("terrain"):
        idx = _subsample_indices(len(g), per_terrain, seed=seed)
        chunks.append(g.iloc[idx])
    return pd.concat(chunks, axis=0, ignore_index=True)


def _global_subsample(df, k, seed=0):
    if k is None or k < 0 or k >= len(df):
        return df
    idx = _subsample_indices(len(df), k, seed=seed)
    return df.iloc[idx]


def _encode_labels(series):
    """把任意离散标签编码为 0..K-1，同时返回映射（用于着色/图例）"""
    values = series.values
    uniq = pd.unique(values)  # 按首次出现的顺序
    mapping = {u: i for i, u in enumerate(uniq)}
    encoded = np.array([mapping[v] for v in values], dtype=int)
    return encoded, mapping


def _label_names_from_map(mapping):
    """根据 mapping（原值→index）重建按 index 排列的名字列表"""
    k = len(mapping)
    names = [None] * k
    for lab, idx in mapping.items():
        names[idx] = str(lab)
    return names


def _scatter_2d(
    Z, labels, label_names, title="",
    save_path=None,
    figsize=(7.2, 5.0),
    label_fontsize=12,      # 轴标题/图标题字号
    tick_fontsize=13,       # 坐标轴刻度字号
    legend_fontsize=10,     # 图例字号
    markerscale=2.2,        # 图例中点的放大倍数
):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    cmap = plt.get_cmap("tab20")

    # 多类别上色
    for lab_idx, lab_name in enumerate(label_names):
        mask = (labels == lab_idx)
        if not np.any(mask):
            continue
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=10, alpha=0.9,
            color=cmap(lab_idx % 20),
            label=str(lab_name),
            linewidths=0,
        )

    # 标题与坐标轴字体
    if title:
        ax.set_title(title, fontsize=label_fontsize)
    ax.set_xlabel("Dimension 1", fontsize=label_fontsize)
    ax.set_ylabel("Dimension 2", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # 图例字体
    if len(label_names) <= 30:
        leg = ax.legend(
            loc="best", frameon=True,
            fontsize=legend_fontsize, markerscale=markerscale,
            handletextpad=0.4, borderpad=0.4
        )
        leg.get_frame().set_alpha(0.9)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=240)
        print(f"💾 saved: {save_path}")

    return fig, ax


def _scatter_3d(Z, labels, label_names, title, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    for lab_idx, lab_name in enumerate(label_names):
        mask = (labels == lab_idx)
        if not np.any(mask):
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=6, alpha=0.85,
                   label=str(lab_name), color=cmap(lab_idx % 20))
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    if len(label_names) <= 20:
        ax.legend(markerscale=3, fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
        print(f"💾 saved: {save_path}")
    return plt


def _parse_grid(s):
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def preprocess_for_tsne(X, method="standard", whiten=False, var_keep=0.95, max_dim=50, seed=0):
    """
    1) 标准化/稳健缩放 -> 2) PCA(可选 whiten) -> 3) 截到指定能量或维度
    返回降维后的 Xp 供 t-SNE 使用
    """
    if method == "standard":
        Xs = StandardScaler().fit_transform(X)
    elif method == "robust":
        Xs = RobustScaler().fit_transform(X)
    else:
        Xs = X

    # PCA: 降冗余 + 去相关（可选 whiten）
    pca = PCA(whiten=whiten, svd_solver="full", random_state=seed)
    Xp_full = pca.fit_transform(Xs)

    if 0 < var_keep <= 1:
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, var_keep) + 1)
    else:
        k = Xp_full.shape[1]

    k = max(2, min(k, max_dim))
    return Xp_full[:, :k]


def run_tsne_with_grid(Xp, labels_for_sil, perp_list, ee_list, lr_list, iters=1000, angle=0.3, seed=0, verbose=True):
    """
    在给定网格上跑 t-SNE，按 2D 嵌入的 silhouette 选择最佳参数。
    返回 (Z_best, best_params_dict, best_sil)
    """
    n = len(Xp)
    perp_list = [p for p in perp_list if p < n]  # perplexity < n
    if not perp_list:
        raise ValueError(f"No valid perplexity (< n={n}) in grid.")

    best = None
    best_tuple = None
    for perp in perp_list:
        for ee in ee_list:
            for lr in lr_list:
                try:
                    Z = TSNE(
                        n_components=2,
                        perplexity=perp,
                        learning_rate=lr,
                        early_exaggeration=ee,
                        init="pca",
                        n_iter=iters,
                        angle=angle,
                        random_state=seed,
                        verbose=1 if verbose else 0
                    ).fit_transform(Xp)
                    sil2d = silhouette_score(Z, labels_for_sil, metric="euclidean")
                    if verbose:
                        print(f"[t-SNE] perp={perp}, ee={ee}, lr={lr} -> silhouette(2D)={sil2d:.3f}")
                    if best is None or sil2d > best:
                        best = sil2d
                        best_tuple = (Z, {"perplexity": perp, "early_exaggeration": ee, "learning_rate": lr}, sil2d)
                except Exception as e:
                    if verbose:
                        print(f"[warn] t-SNE failed @ perp={perp}, ee={ee}, lr={lr}: {e}")
                    continue
    if best_tuple is None:
        raise RuntimeError("t-SNE grid search produced no result.")
    return best_tuple


# -------------------- Main --------------------
def main():
    args = parse_args()
    out_prefix = args.out_prefix or "latents_multi"

    # 1) 读入并合并
    dfs = [_load_and_tag(p) for p in args.csv]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # 2) X/y/terrain 提取
    X = df.drop(columns=["pred_class", "terrain"]).values.astype(np.float32)
    y_pred = df["pred_class"].astype(int).values
    terrain = df["terrain"].astype(str)

    print(f"Loaded {len(df)} rows from {len(args.csv)} files.")
    print("Terrains:", dict(df["terrain"].value_counts()))

    # 3) 采样：先按地形平衡（若指定），再全局采样（若指定）
    df_bal = _balanced_by_terrain(df, args.balance_per_terrain, seed=0)
    df_use = _global_subsample(df_bal, args.sample, seed=0)

    Xs = df_use.drop(columns=["pred_class", "terrain"]).values.astype(np.float32)
    ys_pred = df_use["pred_class"].astype(int).values
    terrains_s = df_use["terrain"].astype(str).values

    # 4) 颜色标签
    if args.color_by == "terrain":
        color_series = df_use["terrain"]
    elif args.color_by == "pred":
        color_series = df_use["pred_class"].astype(str)  # 用字符串便于图例
    else:  # "terrain+pred"
        color_series = df_use["terrain"].astype(str) + "_c" + df_use["pred_class"].astype(str)

    color_ids, color_map = _encode_labels(color_series)
    label_names = _label_names_from_map(color_map)

    print(f"After sampling: X={Xs.shape}, #color_labels={len(label_names)} (by {args.color_by})")

    # 5) 指标（整体）
    try:
        # 注意：silhouette 复杂度较高，样本过大可能耗时
        sil = silhouette_score(Xs, color_ids, metric="euclidean")
        print(f"Silhouette (labels={args.color_by}): {sil:.3f}")
    except Exception as e:
        print(f"[warn] silhouette failed: {e}")

    try:
        k = len(label_names)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        ari = adjusted_rand_score(color_ids, km.labels_)
        print(f"KMeans vs labels({args.color_by}) ARI: {ari:.3f}")
    except Exception as e:
        print(f"[warn] kmeans/ARI failed: {e}")

    # 6) 线性探针：latents → terrain（检查 latent 是否线性可分地形）
    try:
        terrain_ids, terrain_map = _encode_labels(terrains_s)
        clf = LogisticRegression(max_iter=2000)  # 避免 n_jobs 兼容性问题
        # 8:2 划分
        n = len(Xs)
        idx = np.random.RandomState(0).permutation(n)
        tr = idx[: int(0.8 * n)]
        te = idx[int(0.8 * n):]
        clf.fit(Xs[tr], terrain_ids[tr])
        acc = accuracy_score(terrain_ids[te], clf.predict(Xs[te]))
        print(f"Linear probe accuracy (latents → terrain): {acc:.3f}")
    except Exception as e:
        print(f"[warn] linear probe failed: {e}")

    # 7) 可视化
    if args.pca:
        pca2 = PCA(n_components=2)
        Zp2 = pca2.fit_transform(Xs)
        var = pca2.explained_variance_ratio_.sum()
        title = f"PCA-2D (var={var:.2%}) — colored by {args.color_by}"
        _scatter_2d(Zp2, color_ids, label_names, title, save_path=f"{out_prefix}_pca2d_{args.color_by}.png")
        if not args.no_show: plt.show()

    if args.pca3d:
        pca3 = PCA(n_components=3)
        Zp3 = pca3.fit_transform(Xs)
        var3 = pca3.explained_variance_ratio_.sum()
        title3 = f"PCA-3D (var={var3:.2%}) — colored by {args.color_by}"
        _scatter_3d(Zp3, color_ids, label_names, title3, save_path=f"{out_prefix}_pca3d_{args.color_by}.png")
        if not args.no_show: plt.show()

    if args.tsne:
        # ---- 先做统一的预处理（更利于分开） ----
        Xp_tsne = preprocess_for_tsne(
            Xs,
            method=args.preprocess,
            whiten=args.whiten,
            var_keep=args.pca_var_keep,
            max_dim=args.pca_max_dim,
            seed=0
        )
        print(f"[preprocess] For t-SNE: X -> {Xp_tsne.shape}  (method={args.preprocess}, whiten={args.whiten}, var_keep={args.pca_var_keep})")

        if args.tsne_grid:
            perp_list = _parse_grid(args.perp_grid)
            ee_list = _parse_grid(args.ee_grid)
            lr_list = _parse_grid(args.lr_grid)
            print(f"[grid] perp={perp_list}, ee={ee_list}, lr={lr_list}, angle={args.tsne_angle}, iters={args.tsne_iter}")

            Zt, best_params, best_sil = run_tsne_with_grid(
                Xp_tsne, color_ids, perp_list, ee_list, lr_list,
                iters=args.tsne_iter, angle=args.tsne_angle, seed=0, verbose=True
            )
            print(f"[best] t-SNE: {best_params}  silhouette(2D)={best_sil:.3f}")
            out_path = f"{out_prefix}_tsne2d_{args.color_by}_best.png"
            _scatter_2d(
                Zt, color_ids, label_names,
                title=f"",
                save_path=out_path
            )
            if not args.no_show:
                plt.show()
        else:
            # 单次 t-SNE（保留你的超参亦可）
            print(f"Running t-SNE (single): perplexity={args.tsne_perplexity}, iters={args.tsne_iter}, angle={args.tsne_angle}")
            Zt = TSNE(
                n_components=2,
                perplexity=args.tsne_perplexity,
                learning_rate="auto",
                init="pca",
                n_iter=args.tsne_iter,
                angle=args.tsne_angle,
                verbose=1
            ).fit_transform(Xp_tsne)
            _scatter_2d(
                Zt, color_ids, label_names,
                title=f"t-SNE — colored by {args.color_by}",
                save_path=f"{out_prefix}_tsne2d_{args.color_by}.png"
            )
            if not args.no_show:
                plt.show()

    # 8) 分地形的局部指标（可选，打印一下）
    print("\n=== Per-terrain quick stats ===")
    for t_name, g in df_use.groupby("terrain"):
        Xg = g.drop(columns=["pred_class", "terrain"]).values.astype(np.float32)
        yg = g["pred_class"].astype(int).values
        try:
            sil_g = silhouette_score(Xg, yg, metric="euclidean")
            print(f"  {t_name}: silhouette(by pred_class) = {sil_g:.3f}  (n={len(g)})")
        except Exception:
            print(f"  {t_name}: silhouette failed (n={len(g)})")
    print("Done.")


if __name__ == "__main__":
    main()
