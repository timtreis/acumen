# mined from: https://github.com/wangjun-hub/CODEX_SCLC/blob/6d2749bfc1e4c94642a0124a3538633e83cff280/script/cosmx_code.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence

# %%
# import packages
import scanpy as sc
import squidpy as sq

sys.path.append('./Banksy_py')
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition


adata = sc.read('sample.h5ad')

# %%
# preprocess
adata.layers["counts"] = adata.X.copy()
sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

sc.external.pp.scrublet(adata, n_prin_comps=30)
adata = adata[adata.obs['predicted_doublet']==False]
adata = adata[(adata.obs['nFeature_RNA']>=60) & (adata.obs['nCount_RNA']>=100) & (adata.obs['nCount_negprobes']<0.1*adata.obs['nCount_RNA']) & (adata.obs['nCount_RNA']/adata.obs['nFeature_RNA']>1)]
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=1000, batch_key='sample')
hvg_filter = adata.var["highly_variable"]
adata_all = adata.copy()
adata = adata[:, hvg_filter]

# %%
# Scanpy umap&cluster
adata = sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)

sc.tl.umap(adata)
sc.tl.leiden(adata,key_added='scanpy_leiden',resolution=1)

# %%
# Banksy umap&cluster
coord_keys=('x_slide_mm','y_slide_mm','spatial')
k_geom=8
nbr_weight_decay="scaled_gaussian"
max_m=1
lambda_list=[0.1]
resolutions=[1]

banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=False,
    plt_nbr_weights=False,
    plt_agf_angles=False,
    plt_theta=False
)
banksy_dict, banksy_matrix = generate_banksy_matrix(adata, banksy_dict, lambda_list, max_m, variance_balance=True)
pca_umap(
    banksy_dict,
    pca_dims = [20],
    add_umap = True,
    plt_remaining_var = False
)
results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn = 50,
    num_iterations = -1,
    partition_seed = 1234,
    match_labels = True
)

banksy_adata = banksy_dict[nbr_weight_decay][0.1]['adata']
reorder_banksy_adata = banksy_adata[adata.obs_names]
adata.obsm['X_umap'] = reorder_banksy_adata.obsm['reduced_pc_20_umap']
adata.obs['banksy_leiden'] = reorder_banksy_adata.obs['scaled_gaussian_pc20_nc0.1_r1']

# %%
# squidpy co_occurrence
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.co_occurrence(adata,spatial_key='spatial',cluster_key='sclc_celltype')
sq.pl.co_occurrence(
    adata,
    cluster_key='sclc_celltype',
    clusters='Endothelial'
)
