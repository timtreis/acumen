# mined from: https://github.com/NBISweden/workshop-spatial/blob/0cb2bfbda0b88cd042173dc007135ed5b8a95fc4/labs/07a_spatial_adata.ipynb
# symbols: squidpy.gr.spatial_neighbors, squidpy.im.ImageContainer, squidpy.pl.spatial_scatter

# %%
#%pip install scikit-image>0.19

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

# %%
img

# %%
arr_seg = np.zeros((10, 10,3))
arr_seg[4:6, 4:6] = 1

img.add_img(arr_seg, layer="seg1", dims=('x','y','channels1'))
img

# %%
img["seg2"] = arr_seg
img

# %%
print(list(img))
img["image"]

# %%
img.rename("seg2", "new-name")
img

# %%
img.show('seg1')
