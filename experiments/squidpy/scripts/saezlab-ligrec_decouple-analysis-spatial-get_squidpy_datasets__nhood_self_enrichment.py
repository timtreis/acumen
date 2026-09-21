import numpy as np
import pandas as pd
import squidpy as sq


def self_enrichment_top(adata, cluster_key, spatial_key="spatial"):
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key=spatial_key)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1, show_progress_bar=False)

    cats = adata.obs[cluster_key].cat.categories
    nes = pd.DataFrame(
        np.array(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]),
        index=cats,
        columns=cats,
    )
    diag = pd.Series(np.diag(nes.to_numpy()), index=cats)
    return diag.idxmax()


# train: seqfish
adata_seqfish = sq.datasets.seqfish()
train_answer = self_enrichment_top(adata_seqfish, cluster_key="celltype_mapped_refined")
print("TRAIN ANSWER (seqfish):", train_answer)

# test: merfish
adata_merfish = sq.datasets.merfish()
test_answer = self_enrichment_top(adata_merfish, cluster_key="Cell_class", spatial_key="spatial3d")
print("TEST ANSWER (merfish):", test_answer)
