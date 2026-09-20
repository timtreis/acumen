# mined from: https://github.com/theislab/spatial_scog_workshop_2022/blob/45eca39d4db7a86a48c9e25b5f5ce0eba75f823d/squidpy/1_tutorial_spatial_adata.ipynb
# symbols: squidpy.gr.spatial_neighbors, squidpy.im.ImageContainer, squidpy.pl.spatial_scatter

# %%
from anndata import AnnData
import scanpy as sc
import squidpy as sq
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# %%
rng = default_rng(42)
counts = rng.integers(0, 15, size=(10, 100))  # feature matrix
coordinates = rng.uniform(0, 10, size=(10, 2))  # spatial coordinates
image = rng.uniform(0, 1, size=(10, 10, 3))  # image

# %%
adata = AnnData(counts, obsm={"spatial": coordinates}, dtype=np.int64)

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
adata

# %%
sc.pl.spatial(adata, color="leiden", spot_size=1)

# %%
sq.gr.spatial_neighbors(adata, coord_type = "generic", radius=3.0)
sc.pl.spatial(adata, color="leiden", neighbors_key="spatial_neighbors", spot_size=1, edges=True, edges_width=2)

# %%
adata.obsp

# %%
import squidpy as sq
sq.pl.spatial_scatter(adata, shape=None, color="leiden",size=50,connectivity_key="spatial_connectivities",edges_width=2)

# %%
plt.imshow(image)

# %%
spatial_key = "spatial"
library_id = "tissue42"
adata.uns[spatial_key] = {library_id: {}}
adata.uns[spatial_key][library_id]["images"] = {}
adata.uns[spatial_key][library_id]["images"] = {"hires": image}
adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 0.5}

# %%
sq.pl.spatial_scatter(adata, color="leiden")

# %%
adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 0.5, "spot_diameter_fullres": 0.5}
sq.pl.spatial_scatter(adata, color="leiden")

# %%
sq.pl.spatial_scatter(adata, color="leiden", img_cmap="gray")

# %%
img = sq.im.ImageContainer(image)
img.show()
