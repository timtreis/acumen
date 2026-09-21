# mined from: https://github.com/hubmapconsortium/user-templates-api/blob/4d3752f16410862630a37a0a0c94d9d179a065fc/src/user_templates_api/templates/jupyter_lab/templates/squidpy/template.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.ripley, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.ripley

# %%
# !pip install --upgrade pip
# !pip install pandas squidpy anndata hubmap_template_helper

# %%
# One of squidpy's functions depends on 'is_categorical_dtype' which generates a FutureWarning, 
# but as it's called many times, the notebook becomes crowded with this same warning. 
# This cell surpresses this warning.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
import requests
import json

import anndata as ad
import squidpy as sq

from hubmap_template_helper import uuids as hth_uuids
from hubmap_template_helper import compatibility as hth_comp

# %%
# linked datasets
uuids = {{ uuids | safe }}

# accepted assay_display_names
accepted_assay_display_names = ['Slide-seq [Salmon]']

# search_api
search_api = 'https://search.api.hubmapconsortium.org/v3/portal/search'

# %%
uuids = hth_comp.check_template_compatibility(uuids, accepted_assay_display_names=accepted_assay_display_names, search_api=search_api)

# %%
# selected dataset
uuid = uuids[0]

# %%
adata = ad.read_h5ad('./datasets/' + uuid + '/secondary_analysis.h5ad')

# %%
adata

# %%
spatial_key = 'X_spatial'
cluster_key = 'predicted_label'

# %%
sq.gr.spatial_neighbors(adata, coord_type='generic', spatial_key=spatial_key)

# %%
adata

# %%
sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
sq.pl.nhood_enrichment(adata, cluster_key=cluster_key)

# %%
# z_scores = sq.gr.nhood_enrichment(adata, cluster_key='predicted_label', copy=True)
# z_scores

# %%
sq.gr.interaction_matrix(adata,  cluster_key=cluster_key)
sq.pl.interaction_matrix(adata,  cluster_key=cluster_key)

# %%
cluster_of_interest = 'neutrophil'

sq.gr.co_occurrence(adata, spatial_key=spatial_key, cluster_key=cluster_key)
sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=cluster_of_interest)

# %%
mode = 'L'
sq.gr.ripley(adata, cluster_key=cluster_key, spatial_key=spatial_key, mode=mode, max_dist=500)
sq.pl.ripley(adata, cluster_key=cluster_key, mode=mode)
