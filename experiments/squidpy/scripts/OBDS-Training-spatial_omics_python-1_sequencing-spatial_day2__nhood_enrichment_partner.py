import squidpy as sq
import pandas as pd

if __name__ == "__main__":
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", n_jobs=1, seed=0)

    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    cats = adata.obs["cluster"].cat.categories.tolist()
    df = pd.DataFrame(z, index=cats, columns=cats)

    for query in ["Hippocampus", "Cortex_4"]:
        s = df.loc[query].drop(query).sort_values(ascending=False)
        print(query, "->", s.index[0], s.iloc[0], "| runner-up:", s.index[1], s.iloc[1])
