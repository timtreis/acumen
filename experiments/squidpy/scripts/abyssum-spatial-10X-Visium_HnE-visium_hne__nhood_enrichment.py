import pandas as pd
import squidpy as sq


def top_enriched_partner(adata, cluster_key, target_cluster):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, show_progress_bar=False)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    df = pd.DataFrame(z, index=cats, columns=cats)
    row = df.loc[target_cluster].drop(target_cluster)
    return row.idxmax(), row.max()


# train: visium_hne data, target region = Hippocampus
adata_hne = sq.datasets.visium_hne_adata()
train_answer = top_enriched_partner(adata_hne, "cluster", "Hippocampus")
print("train answer:", train_answer)

# test: visium_fluo data, target region = Dentate_gyrus
adata_fluo = sq.datasets.visium_fluo_adata()
test_answer = top_enriched_partner(adata_fluo, "cluster", "Dentate_gyrus")
print("test answer:", test_answer)
