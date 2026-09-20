# mined from: https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/9b94a3cde1d03e30639d2469d3d8672749898a8a/Skills/Genomics/Single_Cell/scverse_Ecosystem/squidpy/tutorials/repo/tutorials/tutorial_read_spatial.ipynb
# symbols: squidpy.gr.spatial_neighbors, squidpy.im.ImageContainer, squidpy.pl.spatial_scatter

# %%
# %matplotlib inline

# %%
from numpy.random import default_rng

import matplotlib.pyplot as plt

import scanpy as sc
import squidpy as sq
from anndata import AnnData

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# %%
rng = default_rng(42)
counts = rng.integers(0, 15, size=(10, 100))  # feature matrix
coordinates = rng.uniform(0, 10, size=(10, 2))  # spatial coordinates
image = rng.uniform(0, 1, size=(10, 10, 3))  # image

# %%
adata = AnnData(counts, obsm={"spatial": coordinates})

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
adata

# %%
sq.pl.spatial_scatter(adata, shape=None, color="leiden", size=50)

# %%
sq.gr.spatial_neighbors(adata, radius=3.0)
sq.pl.spatial_scatter(
    adata,
    color="leiden",
    connectivity_key="spatial_connectivities",
    edges_color="black",
    shape=None,
    edges_width=1,
    size=3000,
)

# %%
plt.imshow(image)

# %%
spatial_key = "spatial"
library_id = "tissue42"
adata.uns[spatial_key] = {library_id: {}}
adata.uns[spatial_key][library_id]["images"] = {}
adata.uns[spatial_key][library_id]["images"] = {"hires": image}
adata.uns[spatial_key][library_id]["scalefactors"] = {
    "tissue_hires_scalef": 1,
    "spot_diameter_fullres": 0.5,
}

# %%
sq.pl.spatial_scatter(adata, color="leiden")

# %%
adata.uns[spatial_key][library_id]["scalefactors"] = {
    "tissue_hires_scalef": 0.5,
    "spot_diameter_fullres": 0.5,
}
sq.pl.spatial_scatter(adata, color="leiden", size=2)

# %%
img = sq.im.ImageContainer(image)
img.show()
