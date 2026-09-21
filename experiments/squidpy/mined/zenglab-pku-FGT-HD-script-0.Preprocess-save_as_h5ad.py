# mined from: https://github.com/zenglab-pku/FGT-HD/blob/6646b2e38020eeed2ea0650cccd22596baafc4f7/script/0.Preprocess/save_as_h5ad.ipynb
# symbols: squidpy.read.visium

# %%
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %%
sample_list = pd.read_csv("HD-OV 100.csv")

# %%
for index, row in sample_list.iterrows():
    sample = row['sample_id']
    input_path = row['path']
    subtype = row['subtype']
    relapse_status = row['relapse_status']
    treatment_status = row['treatment_status']

    path2um = "/outs/binned_outputs/square_002um"
    path8um = "/outs/binned_outputs/square_008um"
    path16um = "/outs/binned_outputs/square_016um"
    path_list = [path2um,path8um,path16um]

    for bin_path in path_list:
        parquet_file_path = f"{input_path}{bin_path}/spatial/tissue_positions.parquet"
        csv_file_path = f"{input_path}{bin_path}/spatial/tissue_positions.csv"
        df = pd.read_parquet(parquet_file_path)
        df.to_csv(csv_file_path, index=False)

    adata_8um = sq.read.visium(f"{input_path}{path8um}")
    adata_8um.obs['in_tissue'] = adata_8um.obs['in_tissue'].astype(float)
    adata_8um.obs['array_row'] = adata_8um.obs['array_row'].astype(float)
    adata_8um.obs['array_col'] = adata_8um.obs['array_col'].astype(float)
    adata_8um.obsm['spatial'] = adata_8um.obsm['spatial'].astype(float)
    adata_8um = adata_8um[:, ~adata_8um.var_names.duplicated()]
    adata_8um.obs['sample'] = sample
    adata_8um.obs['subtype'] = subtype
    adata_8um.obs['relapse_status'] = relapse_status
    adata_8um.obs['treatment_status'] = treatment_status
    adata_8um.obs_names = adata_8um.obs_names + '_' + adata_8um.obs['sample']

    adata_8um.write_h5ad(f"{input_path}/outs/adata_8um.h5ad")

# %%

