import squidpy as sq
import pandas as pd


def main():
    adata = sq.datasets.slideseqv2()
    cats = adata.obs["cluster"].cat.categories.tolist()

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", n_jobs=1, seed=0)
    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    df = pd.DataFrame(z, index=cats, columns=cats)

    for target in ["Oligodendrocytes", "Endothelial_Stalk"]:
        row = df.loc[target].drop(target).sort_values(ascending=False)
        print(target, "-> top spatial-enrichment partner:", row.index[0], round(row.iloc[0], 2))


if __name__ == "__main__":
    main()

# train answer: target "Oligodendrocytes" -> "Polydendrocytes"
# test answer:  target "Endothelial_Stalk" -> "Endothelial_Tip"
