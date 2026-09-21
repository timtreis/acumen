# mined from: https://github.com/HiDiHlabs/SpatialLeiden-Study/blob/65c182765d2c414d9dee1f9279aa3f93e800625f/3_Fig_schematic.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc
import squidpy as sq

# %%
results_dir = Path("./results")
fig_dir = results_dir / "figures"

data_dir = Path("./data")
sdm_dir = data_dir / "sdmbench"
dlpfc_dir = data_dir / "LIBD_DLPFC"

# %%
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
style_kwargs = {"node_size": 30, "width": 1.5, "edge_color": (0.3, 0.3, 0.3, 0.5)}

# %%
adata = ad.read_h5ad(sdm_dir / "E10.5_E2S1.MOSTA.h5ad")

sc.pp.subsample(adata, fraction=0.025)
sc.pp.neighbors(adata, n_neighbors=5)

# %%
graph = nx.from_scipy_sparse_array(adata.obsp["connectivities"])

# %%
fig_umap, ax = plt.subplots(figsize=(6, 6))

nx.draw(graph, pos=adata.obsm["X_umap"][:, :2], **style_kwargs)

# %%
fig_umap.savefig(fig_dir / "Fig_umap.pdf", dpi=600)

# %%
adata = ad.read_h5ad(sdm_dir / "E10.5_E2S1.MOSTA.h5ad")

x, y = adata.obsm["spatial"].T

adata = adata[(x > -130) & (x < -120) & (y > -170) & (y < -160)].copy()

sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=4)

# %%
graph = nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])

# %%
fig_umap, ax = plt.subplots(figsize=(3, 3))

nx.draw(graph, pos=adata.obsm["spatial"], **style_kwargs)

# %%
fig_umap.savefig(fig_dir / "Fig_square_grid.pdf", dpi=600)

# %%
from leiden_utils import get_anndata

# %%
adata = get_anndata(dlpfc_dir / "Br8100_151673")

x, y = adata.obsm["spatial"].T

adata = adata[(x > 5000) & (x < 6000) & (y > 5000) & (y < 6000)].copy()

sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

# %%
graph = nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])

# %%
fig_umap, ax = plt.subplots(figsize=(3, 3))

nx.draw(graph, pos=adata.obsm["spatial"], **style_kwargs)
_ = ax.set(aspect=1)

# %%
fig_umap.savefig(fig_dir / "Fig_isometric_grid.pdf", dpi=600)

# %%
adata = ad.read_h5ad(sdm_dir / "MERFISH_0.24.h5ad")

x, y = adata.obsm["spatial"].T

adata = adata[(x > -3500) & (x < -3250) & (y > -3500) & (y < -3250)].copy()

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=10)

# %%
graph = nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])

# %%
fig_umap, ax = plt.subplots(figsize=(3, 3))

nx.draw(graph, pos=adata.obsm["spatial"], **style_kwargs)

# %%
fig_umap.savefig(fig_dir / "Fig_kNN10.pdf", dpi=600)

# %%
sq.gr.spatial_neighbors(adata, delaunay=True)

# %%
graph = nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])

# %%
fig_umap, ax = plt.subplots(figsize=(3, 3))

nx.draw(graph, pos=adata.obsm["spatial"], **style_kwargs)

# %%
fig_umap.savefig(fig_dir / "Fig_delaunay.pdf", dpi=600)
