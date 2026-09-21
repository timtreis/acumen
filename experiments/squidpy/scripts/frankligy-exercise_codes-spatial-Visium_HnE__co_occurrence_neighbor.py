import numpy as np
import squidpy as sq


def top_partner(adata, anchor, cluster_key="cluster"):
    sq.gr.co_occurrence(adata, cluster_key=cluster_key)
    res = adata.uns[f"{cluster_key}_co_occurrence"]
    occ = res["occ"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    i = cats.index(anchor)
    vals = occ[i, :, 0].copy()
    vals[i] = -1
    order = np.argsort(vals)[::-1]
    return cats[order[0]], vals[order[0]], cats[order[1]], vals[order[1]]


if __name__ == "__main__":
    # TRAIN: visium_hne_adata, anchor cluster "Cortex_4"
    adata = sq.datasets.visium_hne_adata()
    top, val, runner_up, runner_val = top_partner(adata, "Cortex_4")
    print("TRAIN anchor=Cortex_4 top partner:", top, val, "| runner-up:", runner_up, runner_val)

    # TEST: visium_hne_adata, anchor cluster "Striatum"
    top2, val2, runner_up2, runner_val2 = top_partner(adata, "Striatum")
    print("TEST anchor=Striatum top partner:", top2, val2, "| runner-up:", runner_up2, runner_val2)
