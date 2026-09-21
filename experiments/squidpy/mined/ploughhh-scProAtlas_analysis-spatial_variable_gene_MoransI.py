# mined from: https://github.com/ploughhh/scProAtlas_analysis/blob/9f7bfaa8d9725678333663aed26ece9b2ab30b67/spatial_variable_gene_MoransI.py
# symbols: squidpy.gr.nhood_enrichment, squidpy.pl.nhood_enrichment

# import sys
# sys.path.append('/data/twang15/spatial_protein/code/spatial_basic_func/svg.py')
# from svg import *
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import squidpy as sq
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm


def find_files(directory, substring, extension):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file and file.endswith(extension):
                matching_files.append(os.path.join(root, file))
    return matching_files

def calculate_neighbor(adata, n_neighbors=1000, x='x', y='y', n_jobs=248):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=n_jobs)
    coords = adata.obs[[x, y]]
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    num_points = coords.shape[0]

    rows = np.repeat(np.arange(num_points), n_neighbors)
    cols = indices.ravel()
    data = distances.ravel()

    full_distance_matrix = csr_matrix((data, (rows, cols)), shape=(num_points, num_points))
    connectivities = nn.kneighbors_graph(coords, mode='connectivity')


    adata.obsp['spatial_connectivities'] = connectivities
    adata.obsp['spatial_distances'] = full_distance_matrix
    adata.uns['spatial_neighbors'] = {
        'connectivities_key': 'spatial_connectivities',
        'distances_key': 'distances',
        'params': {'n_neighbors': n_neighbors,
        'method': 'gaussian',
        'random_state': 0,
        'metric': 'euclidean'}
    }
    # results = sc.metrics.morans_i(adata)
    # adata.var['morans'] = results
    # gene_ranks = adata.var['morans'].rank(ascending=False).sort_index() 
    # rnk_threshold = np.percentile(gene_ranks, 10.0)
    # gene_svgs = gene_ranks[gene_ranks<rnk_threshold].sort_values().index
    return adata


save_path = '/data/twang15/spatial_protein/CODEX_data/hubmap/proximity/lymph_node/'
files = find_files('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/lymph_node', 'integrated', '.h5ad')


for file in tqdm(files):
    tmp = sc.read_h5ad(file)
    tissue = file.split('/')[8].replace('_protein_integrated.h5ad', '')
    print(tissue)
    calculate_neighbor(tmp, n_neighbors=1000, x='x', y='y', n_jobs=248)
    sq.gr.nhood_enrichment(tmp, cluster_key="cell_type")
    sq.pl.nhood_enrichment(tmp, cluster_key="cell_type", method=None, figsize=(5, 5), title='Cell Type spatial community')
    os.makedirs(f'{save_path}/{tissue}/', exist_ok=True)
    plt.savefig(f'{save_path}/{tissue}/proximity_celltype.png', bbox_inches='tight')
    plt.close()
    # calculate_neighbor(tmp, n_neighbors=1000, x='x', y='y', n_jobs=248)
    sq.gr.nhood_enrichment(tmp, cluster_key="neighborhood10")
    sq.pl.nhood_enrichment(tmp, cluster_key="neighborhood10", method=None, figsize=(5, 5), title='Cell Type spatial community')
    plt.savefig(f'{save_path}/{tissue}/proximity_neighborhood.png', bbox_inches='tight')
    plt.close()
    del tmp



save_path = '/data/twang15/spatial_protein/CODEX_data/hubmap/proximity/large_intestine/'
files = find_files('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/large_intestine', 'integrated', '.h5ad')
files.remove('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/large_intestine/combined_protein_integrated.h5ad')
files.remove('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/large_intestine/HBM245.NHMB.685_reg001/protein_integrated.h5ad')

for file in tqdm(files):
    tmp = sc.read_h5ad(file)
    tissue = file.split('/')[8].replace('_protein_integrated.h5ad', '')
    print(tissue)
    calculate_neighbor(tmp, n_neighbors=1000, x='x', y='y', n_jobs=248)
    sq.gr.nhood_enrichment(tmp, cluster_key="cell_type")
    sq.pl.nhood_enrichment(tmp, cluster_key="cell_type", method=None, figsize=(5, 5), title='Cell Type spatial community')
    os.makedirs(f'{save_path}/{tissue}/', exist_ok=True)
    plt.savefig(f'{save_path}/{tissue}/proximity_celltype.png', bbox_inches='tight')
    plt.close()
    # calculate_neighbor(tmp, n_neighbors=1000, x='x', y='y', n_jobs=248)
    sq.gr.nhood_enrichment(tmp, cluster_key="neighborhood10")
    sq.pl.nhood_enrichment(tmp, cluster_key="neighborhood10", method=None, figsize=(5, 5), title='Cell Type spatial community')
    plt.savefig(f'{save_path}/{tissue}/proximity_neighborhood.png', bbox_inches='tight')
    plt.close()
    del tmp


save_path = '/data/twang15/spatial_protein/CODEX_data/hubmap/proximity/small_intestine/'
files = find_files('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/small_intestine', 'integrated', '.h5ad')
files.remove('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/small_intestine/combined_protein_integrated.h5ad')
# files.remove('/data/twang15/spatial_protein/CODEX_data/hubmap/integrated/large_intestine/HBM245.NHMB.685_reg001/protein_integrated.h5ad')

for file in tqdm(files):
    tmp = sc.read_h5ad(file)
    tissue = file.split('/')[8].replace('_protein_integrated.h5ad', '')
    print(tissue)
    calculate_neighbor(tmp, n_neighbors=1000, x='x', y='y', n_jobs=248)
    sq.gr.nhood_enrichment(tmp, cluster_key="cell_type")
    sq.pl.nhood_enrichment(tmp, cluster_key="cell_type", method=None, figsize=(5, 5), title='Cell Type spatial community')
    os.makedirs(f'{save_path}/{tissue}/', exist_ok=True)
    plt.savefig(f'{save_path}/{tissue}/proximity_celltype.png', bbox_inches='tight')
    plt.close()
    # calculate_neighbor(tmp, n_neighbors=1000, x='x', y='y', n_jobs=248)
    sq.gr.nhood_enrichment(tmp, cluster_key="neighborhood10")
    sq.pl.nhood_enrichment(tmp, cluster_key="neighborhood10", method=None, figsize=(5, 5), title='Cell Type spatial community')
    plt.savefig(f'{save_path}/{tissue}/proximity_neighborhood.png', bbox_inches='tight')
    plt.close()
    del tmp
