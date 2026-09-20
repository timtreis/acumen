# mined from: https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills/blob/b1f9b6e3306f8d870e23b2023057bf730090036f/skills/bio-spatial-transcriptomics-spatial-neighbors/examples/build_spatial_graph.py
# symbols: squidpy.gr.spatial_neighbors

'''Build spatial neighbor graph'''
# Reference: matplotlib 3.8+, numpy 1.26+, scanpy 1.10+, scikit-learn 1.4+, scipy 1.12+, squidpy 1.3+ | Verify API if version differs

import squidpy as sq
import scanpy as sc

adata = sc.read_h5ad('preprocessed.h5ad')
print(f'Loaded: {adata.n_obs} spots')

# n_neighs=6: Default for hexagonal Visium spots; use 4 for square grids, adjust for spot density
sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type='generic')

conn = adata.obsp['spatial_connectivities']
dist = adata.obsp['spatial_distances']

print(f'\nSpatial neighbor graph:')
print(f'  Edges: {conn.nnz}')
print(f'  Mean neighbors/spot: {conn.nnz / adata.n_obs:.1f}')
print(f'  Mean distance: {dist.data[dist.data > 0].mean():.1f}')

adata.write_h5ad('with_spatial_graph.h5ad')
print('\nSaved to with_spatial_graph.h5ad')
