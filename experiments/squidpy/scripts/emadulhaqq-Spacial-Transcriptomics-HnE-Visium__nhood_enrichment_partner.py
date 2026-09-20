import pandas as pd
import squidpy as sq


def top_partner(adata, target):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)
    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    cats = adata.obs["cluster"].cat.categories.tolist()
    df = pd.DataFrame(z, index=cats, columns=cats)
    row = df.loc[target].drop(index=target)
    return row.idxmax(), round(float(row.max()), 2)


def train():
    adata = sq.datasets.visium_hne_adata()
    partner, score = top_partner(adata, "Hippocampus")
    print("TRAIN (visium_hne, Hippocampus) top enriched neighbor:", partner, score)


def test():
    adata = sq.datasets.visium_fluo_adata()
    partner, score = top_partner(adata, "Dentate_gyrus")
    print("TEST (visium_fluo, Dentate_gyrus) top enriched neighbor:", partner, score)


if __name__ == "__main__":
    train()
    test()
