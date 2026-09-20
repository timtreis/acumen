# mined from: https://github.com/theislab/spatial_pancreas/blob/696f21ee53df2f318044aad15d15f676ac0d5e72/notebooks/spatial/graphcompass/graphcompass_S3_v1.ipynb
# symbols: squidpy.pl.var_by_distance, squidpy.tl.var_by_distance

# %%
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import squidpy as sq
import graphcompass as gc

# %%
folder = '/lustre/groups/ml01/datasets/projects/20230301_Sander_SpatialPancreas_sara.jimenez/spatial/'
file   = 'S1_annotated_l0.h5ad' #'imputed_envi.h5ad'
adata = sc.read(filename = folder + file)
adata

# %%
adata.obs.head()

# %%
# define library_key and cluster_key for computing spatial graphs using `squidpy.gr.spatial_neighbors` and 
# `squidpy.gr.nhood_enrichment` 
# spatial graphs are used by the different methods in GraphCompass

library_key="fov"
cluster_key="cell_type_coarse"

# define condition_key used in comparisons
condition_key="condition" # key in adata.obs where conditions are stored

# %%
import pandas as pd
x = pd.crosstab(adata.obs.fov, adata.obs.condition)
x

# %%
adata.X = adata.X.toarray()

# %%
folder = '/lustre/groups/ml01/workspace/sara.jimenez/spatial_pancreas_data/preprocessed_data/data4downstream/'
file   = 'S3_annotated_l0_wlkernel.h5ad' #'imputed_envi.h5ad'
adata_wlkernel = sc.read(filename = folder + file)
adata_wlkernel

# %%
# define necessary params
control_group="ND" # reference group
metric_key="wasserstein_distance" 
method="wl_kernel"

# %%
# Note: a smaller Wasserstein distance indicates a higher similarity between the two graphs, 
# while a larger distance indicates less similarity.

gc.pl.wlkernel.compare_conditions(
    adata=adata_wlkernel,
    library_key=library_key,
    condition_key=condition_key,
    control_group=control_group,
    metric_key=metric_key,
    method=method,
    figsize=(3,5),
    dpi=100,
    #save="figures/mibitof_wwlkerenl.pdf"
)

# %%
# compute filtration curves
### results are stored in adata.uns["filtration_curves"]

# Note: in our case, the spatial graphs have been computed previously (to obtain the WL kernels), so we can set
## `compute_spatial_graphs` to False to save compute time. Otherwise, it should be set to True and 
## `kwargs_spatial_neighbors` should be set depending on the technology used to obtain the spatial data.

gc.tl.filtration_curves.compare_conditions(
    adata=adata_wlkernel,
    library_key=library_key,
    cluster_key=cluster_key,
    condition_key=condition_key,
    compute_spatial_graphs=False,
#     kwargs_spatial_neighbors={
#         'coord_type': 'generic',
#         'delaunay': True,  
#     }  
    
)

# %%
# define necessary params
node_labels=["Acinar","Alpha", "Beta"] # node labels (e.g. cell types) we are intrested in visualising
metric_key="filtration_curves"

# %%
gc.pl.filtration_curves.compare_conditions(
    adata=adata_wlkernel,
    node_labels=node_labels,
    metric_key=metric_key,
    palette="Set2",
    dpi=100,
    figsize=(20,6),
    #save="figures/mibitof_filtration_curves.pdf"
)

# %%
# compute pairwise similarities between cell-type-specific graphs across samples
### results are stored in adata.uns["pairwise_similarities"]

# Note: in our case, the spatial graphs have been computed previously (to obtain the WL kernels), so we can set
## `compute_spatial_graphs` to False to save compute time. Otherwise, it should be set to True and 
## `kwargs_spatial_neighbors` should be set depending on the technology used to obtain the spatial data.

gc.tl.distance.compare_conditions(
    adata=adata_wlkernel,
    library_key=library_key,
    cluster_key=cluster_key,
    method="portrait",
    compute_spatial_graphs=False,
#     kwargs_spatial_neighbors={
#         'coord_type': 'generic',
#         'delaunay': True,  
#     }  
)

# %%
adata_wlkernel

# %%
# define necessary params
control_group="ND" # reference group

# %%
# Note: The size of the dot is indicative of the similarity score variance over samples. 
# The larger the dot size, the lower the score variance and the higher the score confidence is.

gc.pl.distance.compare_conditions(
    adata=adata_wlkernel,
    library_key=library_key,
    condition_key=condition_key,
    control_group=control_group,
    # add_ncells_and_density_plots=True,
    palette="Greys",
    dpi=100,
    figsize=(8,6),
#     save="figures/mibitof_portrait.pdf"
)

# %%
adata_wlkernel

# %%
sq.tl.var_by_distance(
    adata=adata_wlkernel,
    groups="Beta",
    cluster_key="cell_type_coarse",
    library_key="fov",
    covariates=["condition"],
)

# %%
adata_wlkernel.obsm["design_matrix"]

# %%
adata_wlkernel.X = adata_wlkernel.X.toarray()

# %%
sq.pl.var_by_distance(
    adata=adata_wlkernel,
    design_matrix_key="design_matrix",
    var="CD163",
    anchor_key="Beta",
    covariate="condition",
    line_palette=["blue", "orange"],
    show_scatter=False,
    figsize=(5, 4),
)

# %%
adata_wlkernel

# %%

