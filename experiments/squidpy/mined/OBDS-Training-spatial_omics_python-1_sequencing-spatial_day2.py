# mined from: https://github.com/OBDS-Training/spatial_omics_python/blob/ee490a38e0ab8b0429d3a25ed72ba88215e9631f/1_sequencing/spatial_day2.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter, squidpy.pl.var_by_distance, squidpy.tl.var_by_distance

# %%
# import os package for working with system path
import os
# import numpy for scientific computing 
import numpy as np
# import pandas for dataframe manipulation
import pandas as pd
# import matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# import Scanpy and AnnData for single-cell RNAseq
import anndata as ad
import scanpy as sc
# import squidpy for spatial transcriptomics
import squidpy as sq
# import cell2location for cell type deconvolution
import cell2location
from cell2location.models import RegressionModel
import scvi
from scipy.stats import zscore
from scipy.sparse import csr_matrix
import bin2cell as b2c
from spatialdata_io import visium_hd
import spatialdata_plot

# %%
import ipy_slurm_exec
# %load_ext ipy_slurm_exec

# %%
DATA_FOLDERNAME = '/nvme/project/shared/python/5_python_spatial_omics/data'
PRECOMPUTED_FOLDERNAME = '/nvme/project/shared/python/5_python_spatial_omics/data/precomputed/day1'
OUTPUT_FOLDERNAME = '/PATH/TO/YOUR/DIRECTORY'

# %%
adata = sc.read_h5ad(
    os.path.join(PRECOMPUTED_FOLDERNAME, 'day1.h5ad')
)
adata

# %%
sq.pl.spatial_scatter(
    adata,
    color="clusters"
)

# %%
ref = sc.read_h5ad(
    os.path.join(PRECOMPUTED_FOLDERNAME,'ref.h5ad')
)
ref

# %%
sc.pl.umap(
    ref,
    color=["CellType"]
)

# %%
selected = cell2location.utils.filtering.filter_genes(
    ref,
    cell_count_cutoff=5,
    cell_percentage_cutoff2=0.03,
    nonz_mean_cutoff=1.12
)

# %%
ref = ref[:, selected].copy()
ref

# %%
mt = ref.var_names.str.startswith("mt-")
rb = ref.var_names.str.startswith("Rp")

ref.obsm["mt"] = ref[:, mt].X.toarray()
ref.obsm["rb"] = ref[:, rb].X.toarray()

ref = ref[:, ~(mt | rb)].copy()
ref

# %%
# %%slurm_exec -i ref, -o mod --time=00:20:00 --partition=gpu --gpus=1 --cpus=2 --mem=20G
cell2location.models.RegressionModel.setup_anndata(
    adata=ref,
    batch_key='Sample',   # 10X reaction / sample / batch
    labels_key='CellType' # cell type, covariate used for constructing signatures
)
mod = cell2location.models.RegressionModel(ref)
mod.train(max_epochs=250)

# %%
mod.plot_history(20)

# %%
# %%slurm_exec -i ref,mod -o ref,mod --time=00:20:00 --partition=gpu --gpus=1 --cpus=2 --mem=20G
ref = mod.export_posterior(
    ref,
    sample_kwargs={
        'num_samples': 1000,
        'batch_size': 2500
    }
)

# %%
# Saving your trained model
mod.save(OUTPUT_FOLDERNAME, overwrite=True)

adata_file = f"{OUTPUT_FOLDERNAME}/ref.h5ad"
ref.write(adata_file)

# %%
# Reloading your trained model
ref = sc.read_h5ad(os.path.join(OUTPUT_FOLDERNAME, 'ref.h5ad'))
mod = RegressionModel.load(
    OUTPUT_FOLDERNAME,
    ref
)

# %%
# Loading our precomputed model
ref = sc.read_h5ad(os.path.join(f"{PRECOMPUTED_FOLDERNAME}/cell2location/", 'ref.h5ad')) # TODO: file not found
mod = RegressionModel.load(
    f"{PRECOMPUTED_FOLDERNAME}/cell2location/regression",
    ref
)

# %%
if 'means_per_cluster_mu_fg' in ref.varm.keys():
    inf_aver = ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                    for i in ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = ref.var[[f'means_per_cluster_mu_fg_{i}'
                        for i in ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = ref.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]

# %%
intersect = np.intersect1d(adata.var_names, inf_aver.index)
adata = adata[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()
inf_aver

# %%
adata.obs['sample'] = 'slide1'
adata.X = adata.layers['counts'].copy()

# %%
# %%slurm_exec -i inf_aver,adata -o mod --time=00:60:00 --partition=gpu --gpus=1 --cpus=2 --mem=20G

cell2location.models.Cell2location.setup_anndata(
    adata=adata,
    batch_key="sample"
)

mod = cell2location.models.Cell2location(
    adata,
    cell_state_df=inf_aver,
    N_cells_per_location=20,
    detection_alpha=20
)

mod.train(
    max_epochs=20000,
    batch_size=None,
    train_size=1
)

# %%
# %%slurm_exec -i adata,mod -o adata,mod --time=00:20:00 --partition=gpu --gpus=1 --cpus=2 --mem=20G

adata = mod.export_posterior(
    adata,
    sample_kwargs={
        'num_samples': 1000,
        'batch_size': mod.adata.n_obs
    }
)

# %%
mod.save(f"{OUTPUT_FOLDERNAME}/deconvolution", overwrite=False)
adata.write(f"{OUTPUT_FOLDERNAME}/day1_with_cell2location.h5ad")

# %%
adata = sc.read_h5ad(os.path.join(f"{OUTPUT_FOLDERNAME}/deconvolution/", 'day1_with_cell2location.h5ad'))
mod = cell2location.models.Cell2location.load(
    f"{OUTPUT_FOLDERNAME}/cell2location/deconvolution",
    adata
)

# %%
adata = sc.read_h5ad(os.path.join(f"{PRECOMPUTED_FOLDERNAME}/cell2location/", 'adata_with_cell2location.h5ad'))
mod = cell2location.models.Cell2location.load(
    f"{PRECOMPUTED_FOLDERNAME}/cell2location/deconvolution",
    adata
)

# %%
adata

# %%
adata.obsm['q05_cell_abundance_w_sf']

# %%
adata.obs[adata.uns['mod']['factor_names']] = adata.obsm['q05_cell_abundance_w_sf']
adata.obs

# %%
sq.pl.spatial_scatter(
    adata,
    cmap='viridis',
    color=['T-Cells'],
    vmax=3
)
sq.pl.spatial_scatter(
    adata,
    cmap='viridis',
    color=['Fibroblasts'],
    vmax=3
)
sq.pl.spatial_scatter(
    adata,
    cmap='viridis',
    color=['Enterocytes'],
    vmax=3
)

# %%
clust_labels = ['T-Cells', 'Fibroblasts', 'Enterocytes']

clust_col = ['' + str(i) for i in clust_labels] 

with mpl.rc_context({'figure.figsize': (15, 15)}):
    fig = cell2location.plt.plot_spatial(
        adata=adata,
        color=clust_col, 
        labels=clust_labels,
        show_img=True,
        style='fast',
        max_color_quantile=0.9,
        circle_diameter=6,
        colorbar_position='right'
    )

# %%
sc.pl.violin(
    adata,
    keys="T-Cells",
    groupby="clusters",
    stripplot=False,
    jitter=False
)

# %%
cell_types = adata.uns["mod"]["factor_names"]

ab = adata.obs[cell_types]

ab_cluster = ab.groupby(adata.obs["clusters"]).mean()
ab_cluster_z = ab_cluster.apply(
    zscore,
    axis=0
)
cmap = sns.diverging_palette(
    h_neg=220,
    h_pos=10,
    as_cmap=True
)

sns.clustermap(
    ab_cluster_z,
    cmap=cmap,
    center=0,
    linewidths=0,
    rasterized=True
)

# %%
corr = adata.obs[cell_types].corr()
cmap = sns.diverging_palette(
    h_neg=220,
    h_pos=10,
    as_cmap=True
)
sns.clustermap(
    corr,
    cmap=cmap,
    center=0,
    square=True
)

# %%
adata_file = f"{OUTPUT_FOLDERNAME}/adata_with_cell2location.h5ad"
adata.write(adata_file)

# %%
sq.gr.spatial_neighbors(adata)
adata

# %%
sq.gr.nhood_enrichment(
    adata,
    cluster_key="clusters"
)
sq.pl.nhood_enrichment(
    adata,
    cluster_key="clusters",
    method="average",
    figsize=(5, 5)
)
sq.pl.spatial_scatter(
    adata,
    color="clusters"
)

# %%
sq.gr.co_occurrence(
    adata,
    cluster_key="clusters"
)
sq.pl.co_occurrence(
    adata,
    cluster_key="clusters",
    clusters="0",
    figsize=(8, 5)
)

# %%
sq.tl.var_by_distance(
    adata=adata,
    groups="4",
    cluster_key="clusters",
    design_matrix_key="distance_to_cluster_4"
)

# %%
adata.obsm["distance_to_cluster_4"]

# %%
adata.obs["dist_cl4"] = adata.obsm["distance_to_cluster_4"]["4_raw"]
sq.pl.spatial_scatter(adata, color="dist_cl4")

# %%
sq.pl.var_by_distance(
    adata=adata,
    design_matrix_key="distance_to_cluster_4",
    var=["Cd74"],
    anchor_key="4",
    show_scatter=False
)

# %%
# Initialise the new column
adata.obs["spot_group"] = "Other Spots"
# Identify the spots of interest
adata.obs.loc[adata.obs["dist_cl4"] == 0, "spot_group"] = "Spots of Interest"
# Identify spots adjacents to the spots of interest
adata.obs.loc[(adata.obs["dist_cl4"] > 0) & (adata.obs["dist_cl4"] <= 30), "spot_group"] = "Adjacent Spots"
# Convert to categorical data type
adata.obs["spot_group"] = adata.obs["spot_group"].astype("category")
# Plot
sq.pl.spatial_scatter(adata, color="spot_group")

# %%
sc.pp.normalize_total(
    adata,
    inplace=True
)
sc.pp.log1p(adata)

# %%
adata.obs["spot_group"]

# %%
sc.tl.rank_genes_groups(
    adata,
    groupby="spot_group",
    method="wilcoxon",
    use_raw=False,
    groups=["Adjacent Spots"],
    reference="Spots of Interest"
)
sc.get.rank_genes_groups_df(
    adata,
    group="Adjacent Spots"
)

# %%
sc.pl.violin(
    adata,
    keys=["Cd74", "Atp1a1"],
    groupby="spot_group",
    stripplot=False,
    jitter=False
)

# %%
sc.pl.violin(
    adata,
    keys=["T-Cells", "Macrophages"],
    groupby="spot_group",
    stripplot=False,
    jitter=False
)

# %%
section1 = sc.read_h5ad(os.path.join(PRECOMPUTED_FOLDERNAME, 'day1.h5ad'))
section2 = sc.read_h5ad(os.path.join(PRECOMPUTED_FOLDERNAME, 'day14_precomputed.h5ad'))

# %%
sq.pl.spatial_scatter(
    section1,
    color="clusters"
)
sq.pl.spatial_scatter(
    section2,
    color="clusters"
)

# %%
merged = sc.concat(
    [
        section1,
        section2
    ],
    label="dataset",
    uns_merge="unique",
    keys=[
        'Day0',
        'Day14'
    ],
    index_unique="-",
)

# %%
merged

# %%
merged.obs

# %%
merged.obs['dataset'].value_counts()

# %%
sc.pp.highly_variable_genes(
    merged,
    n_top_genes=2000,
    flavor="seurat",
    batch_key='dataset'
)
merged.var

# %%
sc.pp.pca(merged)
sc.pp.neighbors(
    merged,
    n_pcs=10
)
sc.tl.umap(merged)
sc.tl.leiden(
    merged,
    key_added="cluster",
    flavor="igraph",
    n_iterations=2,
    resolution=0.5
)

# %%
sc.pl.umap(
    merged,
    color=['dataset', 'cluster']
)

# %%
sq.pl.spatial_scatter(
    merged,
    color="cluster",
    library_key="dataset",
    library_id=["Day0", "Day14"]
)

# %%
sc.external.pp.harmony_integrate(
    merged,
    key='dataset',
    basis='X_pca',
    adjusted_basis='X_pca_harmony'
)

# %%
sc.pp.neighbors(
    merged,
    n_pcs=10,
    use_rep="X_pca_harmony"
)
sc.tl.umap(merged)
sc.tl.leiden(
    merged,
    resolution=0.5,
    key_added="harmony_cluster"
)

# %%
sc.pl.umap(
    merged,
    color=['dataset','harmony_cluster']
)

# %%
sq.pl.spatial_scatter(
    merged,
    color="harmony_cluster",
    library_key="dataset",
    library_id=["Day0", "Day14"]
)

# %%
sc.tl.rank_genes_groups(
    merged,
    groupby="dataset",
    method="wilcoxon",
    use_raw=False
)

# %%
sc.get.rank_genes_groups_df(
    merged,
    group="Day0"
)

# %%
sq.pl.spatial_scatter(
    merged,
    color="Krt13",
    library_key="dataset",
    library_id=["Day0", "Day14"],
    vmin=0,
    vmax=merged[:, "Krt13"].X.max()
)

# %%
adata.write(os.path.join(OUTPUT_FOLDERNAME, f'day2_merged.h5ad'))

# %%
path = f"{DATA_FOLDERNAME}/visium_hd_mouse_intestine"
hd = visium_hd(path, dataset_id='Visium_HD_Mouse_Small_Intestine')

# %%
hd

# %%
hd['square_016um']

# %%
sc.pp.filter_cells(
    hd['square_016um'],
    min_counts=200,
    inplace=True
)
hd['square_016um']

# %%
sc.pp.normalize_total(
    hd['square_016um'],
    inplace=True
)
sc.pp.log1p(
    hd['square_016um']
)
sc.pp.highly_variable_genes(
    hd['square_016um'],
    flavor="seurat",
    n_top_genes=2000
)
sc.pp.pca(
    hd['square_016um']
)
sc.pp.neighbors(
    hd['square_016um'],
    n_pcs=20
)
sc.tl.umap(
    hd['square_016um']
)
sc.tl.leiden(
    hd['square_016um'],
    key_added="clusters",
    flavor="igraph",
    n_iterations=2,
    resolution=0.5
)

# %%
sc.pl.umap(
    hd['square_016um'],
    color=["clusters"],
    wspace=0.4
)

# %%
hd['square_016um']

# %%
sq.pl.spatial_scatter(
    hd['square_016um'],
    shape=None,
    color="clusters"
)

# %%
sc.pp.filter_cells(
    hd['square_008um'],
    min_counts=200,
    inplace=True
)

# %%
sc.pp.normalize_total(
    hd['square_008um'],
    inplace=True
)
sc.pp.log1p(
    hd['square_008um']
)
sc.pp.highly_variable_genes(
    hd['square_008um'],
    flavor="seurat",
    n_top_genes=2000
)
sc.pp.pca(
    hd['square_008um']
)
sc.pp.neighbors(
    hd['square_008um'],
    n_pcs=20
)
sc.tl.umap(
    hd['square_008um']
)
sc.tl.leiden(
    hd['square_008um'],
    key_added="clusters",
    flavor="igraph",
    n_iterations=2,
    resolution=0.5
)

# %%
sc.pl.umap(
    hd['square_008um'],
    color=["clusters"],
    wspace=0.4
)

# %%
sq.pl.spatial_scatter(
    hd['square_008um'],
    shape=None,
    color="clusters"
)

# %%
crop = (12000, 12000, 16000, 16000)
sq.pl.spatial_scatter(
    hd['square_016um'],
    shape=None,
    color="clusters",
    crop_coord=crop,
    size=30
)
sq.pl.spatial_scatter(
    hd['square_008um'],
    shape=None,
    color="clusters",
    crop_coord=crop,
    size=3
)

# %%
hd.write(os.path.join(OUTPUT_FOLDERNAME, "visium_hd.zarr"))

# %%
path = f"{DATA_FOLDERNAME}/visium_hd_mouse_intestine/binned_outputs/square_002um/"
source_image_path = f"{DATA_FOLDERNAME}/visium_hd_mouse_intestine/image.tiff"
spaceranger_image_path = f"{DATA_FOLDERNAME}/visium_hd_mouse_intestine/spatial/"

# %%
adata = b2c.read_visium(
    path,
    source_image_path = source_image_path,
    spaceranger_image_path = spaceranger_image_path
)
adata.var_names_make_unique()
adata

# %%
sc.pp.calculate_qc_metrics(
    adata,
    inplace=True
)

# %%
sc.pp.filter_cells(
    adata,
    min_counts=1,
    inplace=True
)

# %%
os.makedirs(os.path.join(OUTPUT_FOLDERNAME, "stardist"), exist_ok=True)

# %%
mpp = 0.3
b2c.scaled_he_image(
    adata,
    mpp=mpp,
    save_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "he.tiff")
)

# %%
b2c.destripe(
    adata,
    counts_key="total_counts"
)

# %%
adata

# %%
# %%slurm_exec --time=00:20:00 --partition=gpu --gpus=1 --cpus=2 --mem=30G

b2c.stardist(
    image_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "he.tiff"),
    labels_npz_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "he.npz"),
    stardist_model="2D_versatile_he",
    prob_thresh=0.01,
    nms_thresh=0.5
)

# %%
b2c.insert_labels(
    adata,
    labels_npz_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "he.npz"),
    basis="spatial",
    spatial_key="spatial_cropped_150_buffer",
    mpp=mpp,
    labels_key="labels_he"
)

# %%
b2c.expand_labels(
    adata,
    labels_key='labels_he',
    expanded_labels_key="labels_he_expanded",
    algorithm="volume_ratio"
)

# %%
mask = (
    (adata.obs['array_row'] >= 1530) &
    (adata.obs['array_row'] <= 1550) &
    (adata.obs['array_col'] >= 390) &
    (adata.obs['array_col'] <= 410)
)

adata_subset = adata[mask].copy()

adata_subset = adata_subset[adata_subset.obs['labels_he']>0].copy()

adata_subset.obs['labels_he'] = adata_subset.obs['labels_he'].astype('category')

sc.pl.spatial(
    adata_subset,
    color="labels_he",
    img_key="0.3_mpp_150_buffer",
    basis="spatial_cropped_150_buffer",
    palette="tab20"
)

# %%
b2c.grid_image(
    adata,
    "n_counts_adjusted",
    mpp=mpp,
    sigma=5,
    log1p=True,
    save_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "gex.tiff")
)

# %%
# %%slurm_exec --time=00:20:00 --partition=gpu --gpus=1 --cpus=2 --mem=20G

b2c.stardist(
    image_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "gex.tiff"),
    labels_npz_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "gex.npz"),
    stardist_model="2D_versatile_fluo",
    prob_thresh=0.05,
    nms_thresh=0.5
)

# %%
b2c.insert_labels(
    adata,
    labels_npz_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "gex.npz"),
    basis="array",
    mpp=mpp,
    labels_key="labels_gex"
)

# %%
adata_subset = adata[mask].copy()

adata_subset = adata_subset[adata_subset.obs['labels_gex']>0].copy()

adata_subset.obs['labels_gex'] = adata_subset.obs['labels_gex'].astype('category')

sc.pl.spatial(
    adata_subset,
    color="labels_gex",
    img_key="0.3_mpp_150_buffer",
    basis="spatial_cropped_150_buffer",
    palette="tab20"
)

# %%
crop = b2c.get_crop(
    adata_subset,
    basis="array",
    mpp=mpp
)

rendered = b2c.view_labels(
    image_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "gex.tiff"),
    labels_npz_path=os.path.join(OUTPUT_FOLDERNAME, "stardist", "gex.npz"),
    crop=crop,
    stardist_normalize=True
)
plt.imshow(rendered)

# %%
b2c.salvage_secondary_labels(
    adata,
    primary_label="labels_he_expanded",
    secondary_label="labels_gex",
    labels_key="labels_joint"
)

# %%
adata_cell = b2c.bin_to_cell(
    adata,
    labels_key="labels_joint",
    spatial_keys=[
        "spatial",
        "spatial_cropped_150_buffer"
    ]
)
adata_cell

# %%
adata_cell.var["mt"] = adata_cell.var_names.str.startswith("mt-")
adata_cell.var["rb"] = adata_cell.var_names.str.startswith("Rp")
sc.pp.calculate_qc_metrics(
    adata_cell,
    qc_vars=[
        "mt",
        "rb"
    ],
    inplace=True
)
adata_cell

# %%
sns.histplot(
    adata_cell.obs,
    x="bin_count",
    kde=True,
    bins=60
)

# %%
mask = (adata_cell.obs["bin_count"] > 3) & (adata_cell.obs["total_counts"] > 200) & (adata_cell.obs["bin_count"] < 25)

print(f"Barcodes before filtering: {adata_cell.n_obs}")

adata_cell = adata_cell[mask].copy()

print(f"Barcodes after cell count filter: {adata_cell.n_obs}")

# %%
sc.pp.normalize_total(
    adata_cell,
    inplace=True
)
sc.pp.log1p(
    adata_cell
)
sc.pp.highly_variable_genes(
    adata_cell,
    flavor="seurat",
    n_top_genes=5000
)
sc.pp.pca(
    adata_cell
)
sc.pp.neighbors(
    adata_cell,
    n_pcs=20
)
sc.tl.umap(
    adata_cell
)
sc.tl.leiden(
    adata_cell,
    key_added="clusters",
    flavor="igraph",
    n_iterations=2,
    resolution=0.5
)

# %%
sc.pl.umap(
    adata_cell,
    color=["clusters"],
    wspace=0.4
)

# %%
sq.pl.spatial_scatter(adata_cell, shape=None, color="clusters")

# %%
sc.pl.umap(adata_cell, color=["Myh11", "Cd74", "Epcam"], vmax="p99")

# %%
adata.write(os.path.join(OUTPUT_FOLDERNAME, f'day2_hd_cell_segmented.h5ad'))
