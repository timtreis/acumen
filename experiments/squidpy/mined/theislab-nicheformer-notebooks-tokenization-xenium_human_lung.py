# mined from: https://github.com/theislab/nicheformer/blob/485cadbc5caa15119adfd54228f8a8af835fcabc/notebooks/tokenization/xenium_human_lung.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%


# %%
import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import math
import numba
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

import pyarrow.parquet as pq
import pyarrow
from os.path import join
from tqdm import tqdm

# %%
modality_dict = {
    'dissociated': 3,
    'spatial': 4,}

specie_dict = {
    'human': 5,
    'Homo sapiens': 5,
    'Mus musculus': 6,
    'mouse': 6,}

technology_dict = {
    "merfish": 7,
    "MERFISH": 7,
    "cosmx": 8,
    "NanoString digital spatial profiling": 8,
    "Xenium": 9,
    "10x 5' v2": 10,
    "10x 3' v3": 11,
    "10x 3' v2": 12,
    "10x 5' v1": 13,
    "10x 3' v1": 14,
    "10x 3' transcription profiling": 15, 
    "10x transcription profiling": 15,
    "10x 5' transcription profiling": 16,
    "CITE-seq": 17, 
    "Smart-seq v4": 18,
}

# %%
BASE_PATH = '/lustre/groups/ml01/projects/2023_nicheformer/data/data_to_tokenize'
DATA_PATH = '/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/spatial/preprocessed/human'
OUT_PATH = '/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/tokenized/nicheformer_downstream/xenium_lung'
GENE_MAPPER_PATH = '/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/concat'

# %%
def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X

@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.array, max_seq_len: int = -1, aux_tokens: int = 30):
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]    
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len] 
        sorted_indices = sorted_indices + aux_tokens # we reserve some tokens for padding etc (just in case)
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        
        scores_final[i, :] = scores
        
    return scores_final


def tokenize_data(x: np.array, median_counts_per_gene: np.array, max_seq_len: int = None):
    """Tokenize the input gene vector to a vector of 32-bit integers."""

    x = np.nan_to_num(x) # is NaN values, fill with 0s
    x = sf_normalize(x)
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))

    scores_final = _sub_tokenize_data(out, 4096, 30)

    return scores_final.astype('i4')

# %%
model = sc.read_h5ad(
    f"{BASE_PATH}/model.h5ad"
)

# %%
xenium_mean = np.load(
    f"{BASE_PATH}/xenium_mean_script.npy")

# %%
xenium_mean = np.nan_to_num(xenium_mean)
rounded_values = np.where((xenium_mean % 1) >= 0.5, np.ceil(xenium_mean), np.floor(xenium_mean))
xenium_mean = np.where(xenium_mean == 0, 1, rounded_values)
xenium_mean

# %%
healthy = sc.read_h5ad(f"{DATA_PATH}/10xgenomics_xenium_lung_non_diseased_add_on.h5ad")
healthy

# %%
diseased = sc.read_h5ad(f"{DATA_PATH}/10xgenomics_xenium_lung_cancer_add_on.h5ad")
diseased

# %%
xenium = ad.AnnData.concatenate(healthy, diseased)

# %%
xenium

# %%
xenium.obsm['spatial'] = np.array(xenium.obs[['x', 'y']])
sq.gr.spatial_neighbors(xenium, radius =25, coord_type = 'generic', library_key='condition_id')
xenium

# %%
import seaborn as sns

# %%
# copying the data to ensure original one stays clean
adata_figs = adata.copy()

# %%
adata_figs.layers['counts'] = adata_figs.X

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.settings.set_figure_params(dpi=300, facecolor='white')

# %%
sc.pl.umap(adata, color='condition_id')

# %%
adata = ad.concat([model, xenium], join='inner', axis=0)
# dropping the first observation 
xenium = adata[1:].copy()
# for memory efficiency <
del adata

# %%
xenium

# %%
xenium.obs = xenium.obs[
    ['assay', 'organism', 'nicheformer_split', 'batch']
]
xenium.obs['modality'] = 'spatial'
xenium.obs['specie'] = xenium.obs.organism

# %%
xenium.obs.replace({'specie': specie_dict}, inplace=True)
xenium.obs.replace({'modality': modality_dict}, inplace=True)
xenium.obs.replace({'assay': technology_dict}, inplace=True)

# %%
xenium.obs

# %%
xenium

# %%
# dropping the index as the original index can create issues 
xenium.obs.reset_index(drop=True, inplace=True)
# writing the data
#xenium.write(f"{OUT_PATH}/xenium_human_lung_ready_to_tokenize.h5ad")

# %%
obs_xenium = xenium.obs
print('n_obs: ', obs_xenium.shape[0])
N_BATCHES = math.ceil(obs_xenium.shape[0] / 10_000)
print('N_BATCHES: ', N_BATCHES)
batch_indices = np.array_split(obs_xenium.index, N_BATCHES)
chunk_len = len(batch_indices[0])
print('chunk_len: ', chunk_len)

# %%
xenium_mean.shape

# %%
xenium

# %%
obs_xenium = obs_xenium.reset_index().rename(columns={'index':'idx'})
obs_xenium['idx'] = obs_xenium['idx'].astype('i8')

# %%
for batch in tqdm(range(N_BATCHES)):
    obs_tokens = obs_xenium.iloc[batch*chunk_len:chunk_len*(batch+1)].copy()
    tokenized = tokenize_data(xenium.X[batch*chunk_len:chunk_len*(batch+1)], xenium_mean, 4096)

    obs_tokens = obs_tokens[['assay', 'specie', 'modality', 'idx']]
    # concatenate dataframes
    
    obs_tokens['X'] = [tokenized[i, :] for i in range(tokenized.shape[0])]

    # mix spatial and dissociate data
    obs_tokens = obs_tokens.sample(frac=1)
    
    total_table = pyarrow.Table.from_pandas(obs_tokens)
    
    pq.write_table(total_table, f'{join(OUT_PATH)}/test/tokens-{batch}.parquet',
                    row_group_size=1024,)

# %%
# checking for the last object whether everything looks accurate 
obs_tokens.head(2)

# %%
pd.read_parquet(f'{join(OUT_PATH)}/tokens-{batch}.parquet').head(2)

# %%
OUT_PATH
