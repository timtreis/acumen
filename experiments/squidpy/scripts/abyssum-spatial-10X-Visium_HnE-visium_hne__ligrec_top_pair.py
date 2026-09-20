import squidpy as sq


def top_pair(adata, cluster_key, source, target):
    res = sq.gr.ligrec(
        adata,
        n_perms=100,
        cluster_key=cluster_key,
        seed=0,
        show_progress_bar=False,
        copy=True,
    )
    means = res["means"]
    col = means[(source, target)]
    top = col.sort_values(ascending=False).head(1)
    (lig, rec), value = top.index[0], top.iloc[0]
    return lig, rec, value


# train: visium_hne data, Hippocampus (source) -> Pyramidal_layer (target)
adata_hne = sq.datasets.visium_hne_adata()
print("train answer:", top_pair(adata_hne, "cluster", "Hippocampus", "Pyramidal_layer"))

# test: visium_fluo data, Hippocampus (source) -> Dentate_gyrus (target)
adata_fluo = sq.datasets.visium_fluo_adata()
print("test answer:", top_pair(adata_fluo, "cluster", "Hippocampus", "Dentate_gyrus"))
