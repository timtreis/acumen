import squidpy as sq
import pandas as pd


def top_pair(pvals, means, src, tgt):
    col = (src, tgt)
    p = pvals[col].dropna()
    m = means[col]
    df = pd.DataFrame({"pval": p, "mean": m.loc[p.index]}).sort_values(
        ["pval", "mean"], ascending=[True, False]
    )
    return df.head(3)


def main():
    adata = sq.datasets.visium_hne_adata()
    res = sq.gr.ligrec(
        adata,
        n_perms=1000,
        cluster_key="cluster",
        seed=0,
        n_jobs=1,
        copy=True,
        threshold=0.0,
    )
    pvals = res["pvalues"]
    means = res["means"]

    print("Striatum -> Lateral_ventricle")
    print(top_pair(pvals, means, "Striatum", "Lateral_ventricle"))
    print()
    print("Hippocampus -> Fiber_tract")
    print(top_pair(pvals, means, "Hippocampus", "Fiber_tract"))

    # train answer: IL1B-PTGDS (Striatum source, Lateral_ventricle target)
    # test answer: CNTN4-APLP1 (Hippocampus source, Fiber_tract target)


if __name__ == "__main__":
    main()
