import os

os.environ["TMPDIR"] = "/tmp"

import numpy as np
import squidpy as sq

adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cluster", n_jobs=1, show_progress_bar=False)

cats = adata.obs["cluster"].cat.categories.tolist()
z = adata.uns["cluster_nhood_enrichment"]["zscore"]


def most_enriched_neighbor(target):
    i = cats.index(target)
    row = z[i].copy()
    row[i] = -np.inf
    j = int(np.argmax(row))
    return cats[j], row[j]


train_answer, train_z = most_enriched_neighbor("Hippocampus")
test_answer, test_z = most_enriched_neighbor("Cortex_1")

print("train (Hippocampus) ->", train_answer, train_z)
print("test (Cortex_1) ->", test_answer, test_z)
