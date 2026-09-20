import squidpy as sq


def top_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata)
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
    mat = adata.uns[f"{cluster_key}_interactions"].astype(float)
    cats = list(adata.obs[cluster_key].cat.categories)
    n = len(cats)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            total = mat[i, j] + mat[j, i]
            if best is None or total > best[2]:
                best = (cats[i], cats[j], total)
    return best


# train: imc data
adata_imc = sq.datasets.imc()
print("train (imc):", top_pair(adata_imc, "cell type"))

# test: slideseqv2 data
adata_ssv2 = sq.datasets.slideseqv2()
print("test (slideseqv2):", top_pair(adata_ssv2, "cluster"))
