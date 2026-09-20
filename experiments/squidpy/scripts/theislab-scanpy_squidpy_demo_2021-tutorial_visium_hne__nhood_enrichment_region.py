import squidpy as sq
import pandas as pd


def main():
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)

    cats = adata.obs["cluster"].cat.categories.tolist()
    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    df = pd.DataFrame(z, index=cats, columns=cats)

    for target in ["Hippocampus", "Cortex_4"]:
        row = df.loc[target].drop(target).sort_values(ascending=False)
        print(target, "-> top enriched neighbor region:", row.index[0], round(row.iloc[0], 2))
        print("   second:", row.index[1], round(row.iloc[1], 2))

    # train answer: Hippocampus -> Pyramidal_layer
    # test answer: Cortex_4 -> Cortex_5


if __name__ == "__main__":
    main()
