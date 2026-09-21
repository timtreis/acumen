import squidpy as sq
import pandas as pd


def main():
    adata = sq.datasets.four_i()
    adata.var_names_make_unique()

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)

    cats = adata.obs["cluster"].cat.categories.tolist()
    zscore = adata.uns["cluster_nhood_enrichment"]["zscore"]
    df = pd.DataFrame(zscore, index=cats, columns=cats)

    # train target
    row = df.loc["Cell_periphery_1"].drop("Cell_periphery_1")
    print("train (Cell_periphery_1) top enriched partner:", row.idxmax(), row.max())

    # test target
    row2 = df.loc["Nucleolus"].drop("Nucleolus")
    print("test (Nucleolus) top enriched partner:", row2.idxmax(), row2.max())


if __name__ == "__main__":
    main()
