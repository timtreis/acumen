"""Task: closest_neighbor
For a given reference cell type, find which *other* cell type is found in its
immediate spatial vicinity most often, using distance-binned co-occurrence
probability ratios (short-range bin), in a bundled dataset. Checked for
robustness across several interval bin counts.
"""
import squidpy as sq


def top_close_neighbor(adata, key, ref, intervals=(10, 20, 50, 100)):
    cats = adata.obs[key].cat.categories.tolist()
    target = cats.index(ref)
    tops = []
    for interval in intervals:
        a = adata.copy()
        sq.gr.co_occurrence(a, cluster_key=key, interval=interval)
        occ = a.uns[f"{key}_co_occurrence"]["occ"]
        first = occ[:, target, 0]
        order = sorted(zip(cats, first), key=lambda x: -x[1])
        top_nonself = [c for c, v in order if c != ref][0]
        tops.append(top_nonself)
    return tops


# --- train: imc dataset, reference = endothelial ---
adata_imc = sq.datasets.imc()
tops_imc = top_close_neighbor(adata_imc, "cell type", "endothelial")
print("TRAIN (imc, ref=endothelial):", tops_imc)
# -> all 'vimentin hi stromal cell'

# --- test: mibitof dataset, reference = Myeloid_CD11c ---
adata_mibi = sq.datasets.mibitof()
tops_mibi = top_close_neighbor(adata_mibi, "Cluster", "Myeloid_CD11c")
print("TEST (mibitof, ref=Myeloid_CD11c):", tops_mibi)
# -> all 'Myeloid_CD68'
