import squidpy as sq


def top_interaction(adata, cluster_key, source, target):
    res = sq.gr.ligrec(
        adata,
        n_perms=100,
        cluster_key=cluster_key,
        clusters=[source, target],
        seed=0,
        copy=True,
    )
    pvalues = res["pvalues"]
    means = res["means"]
    col = (source, target)
    pcol = pvalues[col].dropna()
    sig = pcol[pcol <= 0.05]
    mcol = means.loc[sig.index, col]
    top_idx = mcol.sort_values(ascending=False).index[0]
    return top_idx, pcol.loc[top_idx], mcol.loc[top_idx]


def main():
    train_adata = sq.datasets.slideseqv2()
    print(
        "TRAIN (slideseqv2, Oligodendrocytes->Polydendrocytes):",
        top_interaction(train_adata, "cluster", "Oligodendrocytes", "Polydendrocytes"),
    )

    test_adata = sq.datasets.seqfish()
    print(
        "TEST (seqfish, Intermediate mesoderm->Lateral plate mesoderm):",
        top_interaction(
            test_adata,
            "celltype_mapped_refined",
            "Intermediate mesoderm",
            "Lateral plate mesoderm",
        ),
    )


if __name__ == "__main__":
    main()
