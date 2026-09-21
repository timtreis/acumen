# mined from: https://github.com/rockdeme/chrysalis/blob/6c18ff8c3e2b1b64fdf8fe7217a96f98ed6749be/article/A2_human_lymph_node/SVG_detection_methods/1_bsp_spatialde_sepal.ipynb
# symbols: squidpy.gr.sepal, squidpy.gr.spatial_neighbors

# %%
from SpatialDE import test
import pickle
import pandas as pd
import scanpy as sc
import squidpy as sq

data_path = '/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/cell2loc_human_lymph_node/'

# %%
# BSP

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

adata.var_names_make_unique()

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

data = adata.to_df().astype(int)
locs = adata.obsm['spatial']
locs_df = pd.DataFrame(locs, columns=['x', 'y'])

data.to_csv(data_path + 'lymph_node/counts.csv')
locs_df.to_csv(data_path + 'lymph_node/locs.csv', index=False)

# BSP was run via CLI using counts.csv and locs.csv
# python BSP.py --datasetName lymph_node --spaLocFilename locs.csv --expFilename counts.csv

# %%
# spatialDE

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

adata.var_names_make_unique()

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

# sc.pp.normalize_total(adata, inplace=True)
# sc.pp.log1p(adata)

results_t = test(adata)

with open(data_path + 'spatialde.pickle', 'wb') as handle:
    pickle.dump(results_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(data_path + 'spatialde.pickle', 'rb') as f:
    test = pickle.load(f)

test[0].to_csv(data_path + 'spatialde.csv')

# %%
# Sepal

adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')

adata.var_names_make_unique()

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_counts=6000)
sc.pp.filter_genes(adata, min_cells=10)

sq.gr.spatial_neighbors(adata)
genes = list(adata.var_names)
sq.gr.sepal(adata, max_neighs=6, genes=genes, n_jobs=1)
adata.uns["sepal_score"].head(10)
sepal_df = adata.uns["sepal_score"]

sepal_df.to_csv(data_path + 'sepal.csv')
