# mined from: https://github.com/cliffzhou92/STT/blob/1a1e61e4848002bf127670a198844cb11fd067a5/data/drosophila_processed.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import squidpy as sq

# %%
import scanpy as sc
data_dir = '../data/'
adata = sc.read_h5ad(data_dir+'E7-9h_cellbin_tdr_v2.h5ad')
adata1 = adata[adata.obs['slices']==adata.obs['slices'].unique()[15]].copy()
adata2 = adata[adata.obs['slices']==adata.obs['slices'].unique()[16]].copy()

# %%
adata.obs['slices'].unique()[15]

# %%
adata.obs['slices'].unique()[16]

# %%
sq.gr.spatial_neighbors(adata1, coord_type='grid', n_neighs=8)
sq.gr.spatial_neighbors(adata2, coord_type='grid', n_neighs=8)

# %%
adata1.write_h5ad(data_dir+adata.obs['slices'].unique()[15]+'.h5ad')
adata2.write_h5ad(data_dir+adata.obs['slices'].unique()[16]+'.h5ad')

# %%
adata = sc.read_h5ad(data_dir+'SeqScope.h5ad')

# %%
adata.layers

# %%

