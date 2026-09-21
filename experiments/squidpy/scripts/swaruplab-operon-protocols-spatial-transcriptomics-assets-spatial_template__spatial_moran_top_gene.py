import squidpy as sq
import warnings
warnings.filterwarnings("ignore")

visium = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(visium)
sq.gr.spatial_autocorr(visium, mode="moran", n_perms=None, n_jobs=1)
print("TRAIN (visium_hne) top gene:", visium.uns["moranI"].index[0])

seqfish = sq.datasets.seqfish()
sq.gr.spatial_neighbors(seqfish)
sq.gr.spatial_autocorr(seqfish, mode="moran", n_perms=None, n_jobs=1)
print("TEST (seqfish) top gene:", seqfish.uns["moranI"].index[0])
