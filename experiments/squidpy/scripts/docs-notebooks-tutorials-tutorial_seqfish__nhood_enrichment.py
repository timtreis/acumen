"""Confirms answers for the neighborhood-enrichment task (train + test)."""
import squidpy as sq
import pandas as pd


def main():
    adata = sq.datasets.seqfish()
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="celltype_mapped_refined", seed=0, n_jobs=1)

    zscore = adata.uns["celltype_mapped_refined_nhood_enrichment"]["zscore"]
    cats = adata.obs["celltype_mapped_refined"].cat.categories.tolist()
    df = pd.DataFrame(zscore, index=cats, columns=cats)

    for target in ["Lateral plate mesoderm", "Endothelium"]:
        row = df.loc[target].drop(index=target)
        top = row.sort_values(ascending=False)
        print(target, "-> TRAIN/TEST answer:", top.index[0], round(top.iloc[0], 2))
        print(top.head(3))


if __name__ == "__main__":
    main()
