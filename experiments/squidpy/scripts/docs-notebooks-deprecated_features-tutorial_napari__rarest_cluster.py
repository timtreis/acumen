"""
Ground truth for task 'rarest_cluster'.

The napari tutorial loads a Visium AnnData object together with its image
and launches the interactive viewer so the user can browse the precomputed
tissue-region annotation stored in adata.obs['cluster'] as a layer on top of
the tissue image. Without launching the (removed) napari GUI, the underlying
fact being displayed is reproducible directly from the annotation column:
which region label has the fewest spots assigned to it.
"""
import squidpy as sq

# train variant
adata_hne = sq.datasets.visium_hne_adata()
vc_hne = adata_hne.obs["cluster"].value_counts()
print("hne (train) counts:\n", vc_hne.sort_values())
print("hne (train) rarest region:", vc_hne.idxmin(), vc_hne.min())

# test variant
adata_fluo = sq.datasets.visium_fluo_adata()
vc_fluo = adata_fluo.obs["cluster"].value_counts()
print("fluo (test) counts:\n", vc_fluo.sort_values())
print("fluo (test) rarest region:", vc_fluo.idxmin(), vc_fluo.min())
