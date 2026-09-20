# mined from: https://github.com/broadinstitute/celldega/blob/5cb8402283f5c2def3e511f47a5279a8447c352d/notebooks/Clustergram-Entity.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
# %load_ext autoreload
# %autoreload 2
# %env ANYWIDGET_HMR=1

# %%
import celldega as dega
import scanpy as sc
from spatialdata_io import xenium
import squidpy as sq

# %%
adata_p125 = sc.read_h5ad('data/xenium_data/Xenium_V1_Human_Colon_Cancer_P1-P2-P5/adata_P1-P2-P5.h5ad')
adata_p125

# %%
base_urls = [
    'https://raw.githubusercontent.com/broadinstitute/celldega_Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_outs/main/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_outs',
    'https://raw.githubusercontent.com/broadinstitute/celldega_Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs/main/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs',
    'https://raw.githubusercontent.com/broadinstitute/celldega_Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_outs/main/Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_outs',
]

# %%
adata_p1 = adata_p125[adata_p125.obs["dataset"].isin(['P1'])].copy()
adata_p1.obs.index = [x.split('_')[1] for x in adata_p1.obs.index.tolist()]
adata_p1.obs.head()

# %%
adata = sc.read_h5ad('data/xenium_data/Xenium_V1_Human_Colon_Cancer_P1-P2-P5/P1/adata.h5')
adata.obs.set_index('cell_id', inplace=True)
adata

# %%
common_cells = list(set(adata.obs.index.tolist()).intersection(adata_p1.obs.index.tolist()))
print(len(common_cells))
adata = adata[common_cells]
adata_p1 = adata_p1[common_cells]

# %%
adata.obs['leiden'] = adata_p1.obs['leiden']

# %%
sq.pl.spatial_scatter(
    adata,
    library_id="spatial",
    shape=None,
    color=[
        "leiden",
    ],
    wspace=0.4,
)

# %%
alphas=[50, 70]
gdf_alpha_all = dega.nbhd.alpha_shape_cell_clusters(
    adata, 
    cat='leiden', 
    alphas=alphas
)

gdf_alpha = dega.nbhd.filter_alpha_shapes(
    gdf_alpha_all, 
    alpha=50, 
    min_area=0
)

# %%
gdf_alpha.head()

# %%
gdf_alpha.loc[0, 'geometry']

# %%
adata_nbn = dega.nbhd.calc_nbhd_overlap(
    gdf_alpha, 
    metric='ioa', 
    category='leiden'
)
adata_nbn

# %%
mat_nbn = dega.clust.Matrix(
    adata_nbn,
    row_entity="nbhd",
    col_entity="nbhd",
    row_attr=['leiden'], 
    col_attr=['leiden']
)
mat_nbn.clust()
cgm_nbn = dega.viz.Clustergram(
    matrix=mat_nbn,
)

# %%
landscape = dega.viz.Landscape(
    technology='Xenium',
    base_url = base_urls[0],
    adata=adata,
    cell_name_prefix=True, 
    nbhd=gdf_alpha,
)

# %%
dega.viz.landscape_clustergram(landscape, cgm_nbn)

# %%
gdf_hex = dega.nbhd.generate_hextile(
    adata, 
    diameter=100
)

# %%
gdf_hex

# %%
# gdf_hex['name'] = gdf_hex.index.tolist()

# %%
adata_nbp = dega.nbhd.calc_nbhd_by_pop(
    adata, 
    gdf_hex, 
    category="leiden", 
    min_cells=5
)

# %%
adata_nbp

# %%
# sc.pp.normalize_total(adata_nbp, inplace=True)
# sc.pp.neighbors(adata_nbp)
# sc.tl.umap(adata_nbp)
# sc.tl.leiden(adata_nbp)

adata_nbp = sc.read_h5ad('data/xenium_data/Xenium_V1_Human_Colon_Cancer_P1-P2-P5/P1/niche_clustering.h5')

# %%
# Create niche polygons (dissolved)
gdf_niche = dega.nbhd.hextile_niche(
    gdf_hex, 
    adata_nbp, 
    category="leiden"
)

# %%
# # Or keep individual hexagons with niche assignment
# gdf_hex_niche = dega.nbhd.hextile_niche(
#     gdf_hex, 
#     adata_nbp, 
#     category="leiden", 
#     dissolve=False
# )

# %%
gdf_niche.head()

# %%
gdf_niche.loc[0, 'geometry']

# %%
landscape_niche = dega.viz.Landscape(
    technology='Xenium',
    base_url = base_urls[0],
    adata=adata,
    cell_name_prefix=True, 
    nbhd=gdf_niche,
)

# %%
mat_niche = dega.clust.Matrix(
    adata_nbp, 
    row_entity=("cell", "leiden"),    
    col_entity="nbhd", 

    # col_entity='nbhd', -> nbhd:name (implied attr)
    # col_entity=('nbhd', 'name'),
    # col_entity={'entity': 'nbhd', 'attr':'name'}
    # col_entity={'entity': 'nbhd', 'attr':'leiden'}
    
    row_attr=['leiden'],    
    col_attr=['leiden'],
    name='niche'
)
mat_niche.downsample_to(category='leiden')
mat_niche.clust()
cgm_niche = dega.viz.Clustergram(
    matrix=mat_niche
)

# %%
dega.viz.landscape_clustergram(landscape_niche, cgm_niche)

# %%
mat_gex = dega.clust.Matrix(
    adata,
    col_attr=['leiden']
)
mat_gex.downsample_to(category='leiden')
mat_gex.norm(axis='row', by='zscore')
mat_gex.clust()
cgm_gex = dega.viz.Clustergram(matrix=mat_gex)

# %%
dega.viz.landscape_clustergram(landscape, cgm_gex, enrich=True)

# %%
cgm_gex.row_entity

# %%
cgm_gex.col_entity

# %%
cluster = "14"
adata_14 = adata[adata.obs["leiden"].isin([cluster])].copy()
adata_14

# %%
mat_14 = dega.clust.Matrix(
    adata_14, 
    col_entity={"entity": "cell", "attr": "name"}
)
mat_14.filter(axis='row', by='mean', num=100)
mat_14.norm(axis='row', by='zscore')
mat_14.clust()
cgm_14 = dega.viz.Clustergram(
    matrix=mat_14, 
    # width=500, 
    # height=500,
    manual_col_cat='sub_type', 
)

# %%
landscape_14 = dega.viz.Landscape(
    technology='Xenium',
    base_url = base_urls[0],
    adata=adata_14,
    cell_name_prefix=True, 
    height=600,
)

# %%
dega.viz.landscape_clustergram(landscape_14, cgm_14)

# %%


# %%

