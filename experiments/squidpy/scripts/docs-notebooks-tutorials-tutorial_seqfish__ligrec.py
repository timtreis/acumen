"""Confirms answers for the ligand-receptor interaction task (train + test)."""
import squidpy as sq
import pandas as pd

adata = sq.datasets.seqfish()
res = sq.gr.ligrec(
    adata,
    cluster_key="celltype_mapped_refined",
    n_perms=1000,
    seed=0,
    n_jobs=1,
    copy=True,
    show_progress_bar=False,
)
means = res["means"]
pvalues = res["pvalues"]


def top_pair(source, target):
    col = (source, target)
    m = means[col]
    p = pvalues[col]
    df = pd.DataFrame({"mean": m, "pval": p}).dropna()
    sig = df[(df["mean"] > 0.3) & (df["pval"] < 1e-4)]
    sig = sig.sort_values("mean", ascending=False)
    return sig


for source, target in [
    ("Lateral plate mesoderm", "Intermediate mesoderm"),
    ("Endothelium", "Haematoendothelial progenitors"),
]:
    sig = top_pair(source, target)
    ligand, receptor = sig.index[0]
    print(f"{source} -> {target}: TOP = {ligand}-{receptor} (mean={sig.iloc[0]['mean']:.2f})")
    print(sig.head(5))
