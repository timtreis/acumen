# mined from: https://github.com/simonwm/tacco_examples/blob/ed61ddc584be72217fe83d33b8995589264efd50/workflow/cooccurrence/notebook.ipynb
# symbols: squidpy.datasets.imc, squidpy.datasets.seqfish, squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence

# %%
import os
import sys
import matplotlib
import time

import pandas as pd
import numpy as np
import anndata as ad

import tacco as tc
import squidpy as sq

# The notebook expects to be executed either in the workflow directory or in the repository root folder...
sys.path.insert(1, os.path.abspath('workflow' if os.path.exists('workflow/common_code.py') else '..')) 
import common_code

# %%
data_path = common_code.find_path('results/slideseq_mouse_olfactory_bulb')
plot_path = common_code.find_path('results/cooccurrence',create_if_not_existent=True)

# %%
reference = ad.read(f'{data_path}/reference.h5ad')

# %%
pucks = ad.concat({ key: ad.read(f'{data_path}/{key}.h5ad') for key in ['puck_1_4','puck_1_5','puck_1_6','puck_1_7',] }, label='puck', index_unique='-')

# %%
tc.tl.annotate(pucks,reference,'type',result_key='type',multi_center=10,verbose=0);

# %%
fig = tc.pl.scatter(pucks, 'type', group_key='puck');

# %%
reference.obs[['type','long']].drop_duplicates().reset_index(drop=True)

# %%
single_puck = pucks.query('puck=="puck_1_5"').copy()

# %%
tc.tl.co_occurrence(single_puck, 'type', result_key='type-type', delta_distance=20, max_distance=1000, sparse=False);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='composition', wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='log_composition', log_base=2, wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='occ', wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='log_occ', log_base=2, wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='distance_distribution', wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='log_distance_distribution', log_base=2, wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='relative_distance_distribution', wspace=0.25);

# %%
fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='log_relative_distance_distribution', log_base=2, wspace=0.25);

# %%
pucks.obsm['marker'] = pd.DataFrame(index=pucks.obs.index)
for marker in ['Cd68','Cd4','Cd8a','Epcam','Vwf','Rbfox3']:
    pucks.obsm['marker'][marker] = tc.sum(pucks[:,[marker]].X,axis=1)
pucks.obsm['marker']['other'] = tc.sum(pucks.X,axis=1) - tc.sum(pucks.obsm['marker'],axis=1)
pucks.obsm['marker'] /= tc.sum(pucks.obsm['marker'],axis=1).to_numpy()[:,None]

single_puck = pucks.query('puck=="puck_1_5"').copy()

# %%
# %time tc.tl.distance_matrix(single_puck, max_distance=np.inf, result_key='precomputed');

# %%
tc.tl.co_occurrence(single_puck, 'type', center_key='type', result_key='type-type', delta_distance=20, max_distance=1000, distance_key='precomputed');
tc.tl.co_occurrence(single_puck, 'marker', center_key='type', result_key='marker-type', delta_distance=20, max_distance=1000, distance_key='precomputed');
tc.tl.co_occurrence(single_puck, 'type', center_key='marker', result_key='type-marker', delta_distance=20, max_distance=1000, distance_key='precomputed');
tc.tl.co_occurrence(single_puck, 'marker', center_key='marker', result_key='marker-marker', delta_distance=20, max_distance=1000, distance_key='precomputed');

# %%
marker_colors = tc.pl.get_default_colors(single_puck.obsm['marker'].columns, offset=20)

fig = tc.pl.co_occurrence(single_puck, 'type-type', score_key='log_occ', log_base=2, wspace=0.25);
fig = tc.pl.co_occurrence(single_puck, 'marker-type', score_key='log_occ', log_base=2, wspace=0.25, colors=marker_colors);
fig = tc.pl.co_occurrence(single_puck, 'type-marker', score_key='log_occ', log_base=2, wspace=0.25);
fig = tc.pl.co_occurrence(single_puck, 'marker-marker', score_key='log_occ', log_base=2, wspace=0.25, colors=marker_colors);

# %%
single_puck.obsm['counts'] = single_puck[:,['Cd68','Cd4','Cd8a','Epcam','Vwf','Rbfox3']].to_df()
fig = tc.pl.scatter(single_puck, 'counts');

# %%
pd.DataFrame({'total_counts':single_puck.obsm['counts'].sum(axis=0),'positive_beads':(single_puck.obsm['counts']!=0).sum(axis=0),})

# %%
single_puck.obsm['type'][single_puck.obsm['counts']['Cd8a']!=0]

# %%
individual_pucks = {}
for puck_name,df in pucks.obs.groupby('puck'):
    individual_pucks[puck_name] = pucks[df.index].copy()
    tc.tl.co_occurrence(individual_pucks[puck_name], 'type', result_key='type-type',delta_distance=20,max_distance=1000,sparse=False,);

# %%
fig = tc.pl.co_occurrence(individual_pucks, 'type-type', wspace=0.25);

# %%
fig = tc.pl.co_occurrence(individual_pucks, 'type-type', wspace=0.25, merged=True);

# %%
tc.tl.co_occurrence(pucks, 'type', center_key='type', sample_key='puck', result_key='type-type', delta_distance=20, max_distance=1000, sparse=False);
tc.tl.co_occurrence(pucks, 'marker', center_key='type', sample_key='puck', result_key='marker-type', delta_distance=20, max_distance=1000, sparse=False);
tc.tl.co_occurrence(pucks, 'type', center_key='marker', sample_key='puck', result_key='type-marker', delta_distance=20, max_distance=1000, sparse=False);
tc.tl.co_occurrence(pucks, 'marker', center_key='marker', sample_key='puck', result_key='marker-marker', delta_distance=20, max_distance=1000, sparse=False);

# %%
fig = tc.pl.co_occurrence(pucks, 'type-type', score_key='log_occ', log_base=2, wspace=0.25);
fig = tc.pl.co_occurrence(pucks, 'marker-type', score_key='log_occ', log_base=2, wspace=0.25, colors=marker_colors);
fig = tc.pl.co_occurrence(pucks, 'type-marker', score_key='log_occ', log_base=2, wspace=0.25);
fig = tc.pl.co_occurrence(pucks, 'marker-marker', score_key='log_occ', log_base=2, wspace=0.25, colors=marker_colors);

# %%
pucks.obsm['counts'] = pucks[:,['Cd68','Cd4','Cd8a','Epcam','Vwf','Rbfox3']].to_df()
pd.DataFrame({('single_puck','total_counts'):single_puck.obsm['counts'].sum(axis=0),('single_puck','positive_beads'):(single_puck.obsm['counts']!=0).sum(axis=0),('four_pucks','total_counts'):pucks.obsm['counts'].sum(axis=0),('four_pucks','positive_beads'):(pucks.obsm['counts']!=0).sum(axis=0),})

# %%
adata_squidpy = sq.datasets.imc()
sq.gr.co_occurrence(adata_squidpy, cluster_key="cell type")
sq.pl.co_occurrence(adata_squidpy, cluster_key="cell type", clusters="basal CK tumor cell", figsize=(6,4))

# %%
max_dist = adata_squidpy.uns[f'cell type_co_occurrence']['interval'][-1]
adata_tacco = sq.datasets.imc()
tc.tl.co_occurrence(adata_tacco, annotation_key="cell type", position_key='spatial', result_key=f'cell type_co_occurrence', max_distance=max_dist, sparse=False, delta_distance=max_dist/50)
tc.pl.co_occurrence(adata_tacco, analysis_key=f'cell type_co_occurrence', score_key='occ', show_only_center="basal CK tumor cell", axsize=(3,3));

# %%
sq.pl.co_occurrence(adata_tacco, cluster_key="cell type", clusters="basal CK tumor cell", figsize=(6,4))

# %%
def time_co_occurrence(label, adata, annotation_key):
    # measure always the second execution to avoid overhead from jitting etc.
    timings = []

    sq.gr.co_occurrence(adata, cluster_key=annotation_key, n_jobs=tc.utils.cpu_count())
    start = time.time()
    sq.gr.co_occurrence(adata, cluster_key=annotation_key, n_jobs=tc.utils.cpu_count())
    end = time.time()
    timings.append((label,'Squidpy-parallel',end-start))

    sq.gr.co_occurrence(adata, cluster_key=annotation_key)
    start = time.time()
    sq.gr.co_occurrence(adata, cluster_key=annotation_key)
    end = time.time()
    timings.append((label,'Squidpy-serial',end-start))

    max_dist = adata.uns[f'{annotation_key}_co_occurrence']['interval'][-1]

    tc.tl.co_occurrence(adata, annotation_key=annotation_key, position_key='spatial', result_key=f'{annotation_key}_co_occurrence', max_distance=max_dist, sparse=False, delta_distance=max_dist/50);
    start = time.time()
    tc.tl.co_occurrence(adata, annotation_key=annotation_key, position_key='spatial', result_key=f'{annotation_key}_co_occurrence', max_distance=max_dist, sparse=False, delta_distance=max_dist/50);
    end = time.time()
    timings.append((label,'TACCO-dense',end-start))

    tc.tl.co_occurrence(adata, annotation_key=annotation_key, position_key='spatial', result_key=f'{annotation_key}_co_occurrence', max_distance=max_dist, sparse=True, delta_distance=max_dist/50);
    start = time.time()
    tc.tl.co_occurrence(adata, annotation_key=annotation_key, position_key='spatial', result_key=f'{annotation_key}_co_occurrence', max_distance=max_dist, sparse=True, delta_distance=max_dist/50);
    end = time.time()
    timings.append((label,'TACCO-sparse',end-start))
    
    return timings

timings_cooc = []
timings_cooc.extend(time_co_occurrence('imc', sq.datasets.imc(), "cell type"))
timings_cooc.extend(time_co_occurrence('seqfish', sq.datasets.seqfish(), 'celltype_mapped_refined'))

timings_cooc = pd.DataFrame(timings_cooc,columns=['dataset','method','time (s)'])

timings_cooc['method'] = timings_cooc['method'].astype(pd.CategoricalDtype(['TACCO-sparse','TACCO-dense','Squidpy-serial','Squidpy-parallel'],ordered=True))

# %%
def time_neighbourhood_enrichment(label, adata, annotation_key):
    # measure always the second execution to avoid overhead from jitting etc.
    timings = []

    # let squidpy figure out the distance scale
    sq.gr.spatial_neighbors(adata)
    radius = adata.obsp['spatial_distances'].data.mean()

    sq.gr.spatial_neighbors(adata, coord_type='generic', radius=radius)
    sq.gr.nhood_enrichment(adata, cluster_key=annotation_key,)
    start = time.time()
    sq.gr.spatial_neighbors(adata, coord_type='generic', radius=radius)
    sq.gr.nhood_enrichment(adata, cluster_key=annotation_key,)
    end = time.time()
    timings.append((label,'Squidpy',end-start))

    tc.tl.co_occurrence_matrix(adata, annotation_key=annotation_key, position_key='spatial', result_key=f'{annotation_key}_co_occurrence', max_distance=radius, sparse=True, n_permutation=1000)
    start = time.time()
    tc.tl.co_occurrence_matrix(adata, annotation_key=annotation_key, position_key='spatial', result_key=f'{annotation_key}_co_occurrence', max_distance=radius, sparse=True, n_permutation=1000)
    end = time.time()
    timings.append((label,'TACCO',end-start))

    return timings

timings_neigh = []
timings_neigh.extend(time_neighbourhood_enrichment('imc', sq.datasets.imc(), "cell type"))
timings_neigh.extend(time_neighbourhood_enrichment('seqfish', sq.datasets.seqfish(), 'celltype_mapped_refined'))

timings_neigh = pd.DataFrame(timings_neigh,columns=['dataset','method','time (s)'])

timings_neigh['method'] = timings_neigh['method'].astype(pd.CategoricalDtype(['TACCO','Squidpy'],ordered=True))

# %%
fig,axs = tc.pl.subplots(2, axsize=(2,4), x_padding=2)

for title,timings,ax in zip(['Co-occurrence runtime','Neighbourhood enrichment runtime'],[timings_cooc,timings_neigh],axs.flatten()):
    y = timings['time (s)']

    ynew = np.array([0.1,1,10,60,600,3600,36000])
    ynew_minor = np.concatenate([np.arange(0.1,1,0.1),np.arange(1,10,1),np.arange(10,60,10),np.arange(60,600,60),np.arange(600,3600,600),np.arange(3600,36000,3600)]).flatten()
    ynewlabels = np.array(['0.1s','1s','10s','1min','10min','1h','10h'])
    ymin = y.min() * 0.5
    ymax = y.max() * 2.0
    ynewlabels = ynewlabels[(ynew > ymin) & (ynew < ymax)]
    ynew = ynew[(ynew > ymin) & (ynew < ymax)]
    ynew_minor = ynew_minor[(ynew_minor > ymin) & (ynew_minor < ymax)]
    for yn in ynew:
        ax.axhline(yn, color='gray', linewidth=0.5)

    colors = tc.pl.get_default_colors(2)
    for im,method in enumerate(timings['method'].cat.categories):
        color = colors[method.startswith('Squidpy')]
        marker = ['o','v','^','s',][im]
        sub = timings[timings['method']==method]
        ax.scatter(sub['dataset'],sub['time (s)'], label=method, marker=marker, color=color)
    ax.set_yscale('log')

    ax.set_yticks(ynew_minor,minor=True)
    ax.set_yticks(ynew)
    ax.set_yticklabels(ynewlabels)
    ax.set_yticklabels([],minor=True)

    ax.set_xlim(-0.5,len(timings['dataset'].unique()) - 0.5)

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    
    ax.set_title(title)

# %%

