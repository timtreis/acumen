import pandas as pd
import scanpy as sc
import squidpy as sq

adata = sc.read_h5ad("/tmp/slideseqv2.h5ad")  # sq.datasets.slideseqv2()

res = sq.gr.ligrec(
    adata,
    n_perms=1000,
    cluster_key="cluster",
    clusters=["Polydendrocytes", "Oligodendrocytes"],
    seed=0,
    copy=True,
    show_progress_bar=False,
)

for direction in [
    ("Oligodendrocytes", "Polydendrocytes"),
    ("Polydendrocytes", "Oligodendrocytes"),
]:
    pvals = res["pvalues"][direction]
    means = res["means"][direction]
    df = pd.DataFrame({"pvalue": pvals, "mean": means}).dropna(subset=["mean"])
    sig = df[df["pvalue"] < 0.05].sort_values("mean", ascending=False)
    top = sig.index[0]
    print(f"{direction[0]} -> {direction[1]}: top significant pair = {top[0]}-{top[1]} "
          f"(mean={sig.iloc[0]['mean']:.4f}, p={sig.iloc[0]['pvalue']:.3f})")
