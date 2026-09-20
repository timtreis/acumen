import squidpy as sq
import pandas as pd


def top_enriched_partner(adata, cluster_key, target):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, n_jobs=1)
    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    df = pd.DataFrame(zscore, index=cats, columns=cats)
    s = df[target].drop(target).sort_values(ascending=False)
    return s.index[0], s.iloc[0]


if __name__ == "__main__":
    # --- train: seqfish dataset ---
    adata_train = sq.datasets.seqfish()
    partner, score = top_enriched_partner(adata_train, "celltype_mapped_refined", "Cardiomyocytes")
    print("TRAIN answer:", partner, score)

    # --- test: imc dataset ---
    adata_test = sq.datasets.imc()
    partner, score = top_enriched_partner(adata_test, "cell type", "macrophages")
    print("TEST answer:", partner, score)
