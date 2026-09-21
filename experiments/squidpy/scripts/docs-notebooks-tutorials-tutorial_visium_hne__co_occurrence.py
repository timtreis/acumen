import os

os.environ["TMPDIR"] = "/tmp"

import numpy as np
import squidpy as sq

adata = sq.datasets.visium_hne_adata()
sq.gr.co_occurrence(adata, cluster_key="cluster")

occ = adata.uns["cluster_co_occurrence"]["occ"]
cats = adata.obs["cluster"].cat.categories.tolist()


def most_co_occurring_at_shortest_distance(target):
    i = cats.index(target)
    row = occ[i, :, 0].copy()
    row[i] = -np.inf
    j = int(np.argmax(row))
    return cats[j], row[j]


train_answer, train_val = most_co_occurring_at_shortest_distance("Cortex_4")
test_answer, test_val = most_co_occurring_at_shortest_distance("Fiber_tract")

print("train (Cortex_4) ->", train_answer, train_val)
print("test (Fiber_tract) ->", test_answer, test_val)
