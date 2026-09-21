# mined from: https://github.com/akiviaho/ST-prostate/blob/df0d2261af3a14a3fa0c65b553a200d3edcbe279/spatial-post-c2l-cell-type-mapping.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors

# %%
# Date: 20.3.2023
# Author: Antti Kiviaho
#
# Notebook for analysing and visualizing visium data after copy number variation, single cell mapping and clustering
# analyses. This is the main results notebook

# %%
import os 
os.chdir('/lustre/scratch/kiviaho/prostate_spatial/')

import scanpy as sc
import numpy as np
import squidpy as sq
import pandas as pd
import anndata as ad

import matplotlib.pyplot as plt
from scripts.utils import load_from_pickle, get_sample_ids_reorder, get_sample_id_mask, get_include_exclude_info
import matplotlib as mpl
from sklearn.decomposition import NMF

import seaborn as sns
sns.set_theme()

sc.set_figure_params(figsize=(6,6))

import warnings
warnings.filterwarnings("ignore")


samples = get_sample_ids_reorder()
sample_id_masks = get_sample_id_mask()

# Change the run_name variable to select the appropriate iteration
run_name = '20240125' 

# %%
# Download data and format cell2location mapping results into obs columns 
adata_vis = sc.read_h5ad('./c2l-results/visium_adata_with_c2l_mapping_'+run_name+'.h5ad')
cell_type_abundances = adata_vis.obsm['q05_cell_abundance_w_sf'].copy()

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

# Remove the prefix
cell_type_abundances.columns = [remove_prefix(col,'q05cell_abundance_w_sf_') for col in cell_type_abundances.columns]

# Save celltype names
cell_types = cell_type_abundances.columns.tolist()

bcode_to_sample_dict = {}

for s in samples:
    sample_map_dict = dict(zip(cell_type_abundances.index[cell_type_abundances.index.str.contains(s)].tolist(),
            np.repeat(s,len(cell_type_abundances.index[cell_type_abundances.index.str.contains(s)]))
    ))
    bcode_to_sample_dict = {**bcode_to_sample_dict,**sample_map_dict}

cell_type_abundances['sample_id'] = cell_type_abundances.index.map(bcode_to_sample_dict)
cell_type_abundances['sample_class'] = [s.split(' ')[0] for s in cell_type_abundances['sample_id'].map(sample_id_masks).tolist()]

# %%
# Save the cell mapping data into anndata format
cell_mapping_dat = ad.AnnData(
    X=np.array(cell_type_abundances[cell_types]),
    obs=cell_type_abundances[['sample_id','sample_class']],
    var=pd.DataFrame(index=cell_types))

cell_mapping_dat.write('c2l_mapping_as_anndata_'+run_name+'.h5ad')

# %%
# Plots and saves top n_types with highest prevelance on visium slides
sns.set_theme(style='white')
n_types = 12

for sample in samples:
    
    slide = sc.read_h5ad('./data/normalized_visium/'+sample+'_normalized.h5ad')
    
    # Collect the cell abundances
    sample_cell_abundances = cell_type_abundances[cell_type_abundances['sample_id']==sample][cell_types]

    # Concat the cell abundances to a slide
    slide.obs = pd.concat([slide.obs,sample_cell_abundances],axis=1)

    # Subset to only plot the cell types with highest mean prevalence
    cell_types_to_plot = slide.obs[cell_types].mean(axis=0).sort_values(ascending=False)[:n_types]

    # plot in spatial coordinates
    with mpl.rc_context({'axes.facecolor':  'black',
                        'figure.figsize': [4.5, 5]}):

        sc.pl.spatial(slide, cmap='magma',
                    # show first 8 cell types
                    color=cell_types_to_plot.index,
                    ncols=4, size=1.3,alpha_img=0.8,
                    # limit color scale at 99.2% quantile of cell abundance
                    vmin=0, vmax='p99.2', show=False
                    )
        plt.savefig('./plots/c2l_mapping_results_'+run_name+'/'+sample+'_c2l_mapping_top12_abundant.png',dpi=200)
        plt.clf()

# %%
# format the observations to show BPH-TRNA-NEADT-CRPC divide

df = pd.DataFrame(cell_mapping_dat.X, columns=cell_mapping_dat.var.index,index=cell_mapping_dat.obs.index)
sample_id_masks = get_sample_id_mask()

sample_indices_from_df = pd.Series([('_').join(s.split('_')[:-1]) for s in df.index])
df.index = pd.Index(list(sample_indices_from_df.map(sample_id_masks)))

# %%
# Save the key cell-type mapping numbers to an excel file (supplementary table)

cell_mapping_characteristics_by_phenotype = {}

phenotypes = ['','BPH','TRNA','NEADT','CRPC']

for phenotype in phenotypes:
    ctype_count_df = df.loc[df.index.str.contains(phenotype)].describe().T

    ctype_count_df['sum'] = df.loc[df.index.str.contains(phenotype)].sum().loc[ctype_count_df.index].astype(float)
    ctype_count_df = ctype_count_df[['sum','mean','min','25%','50%','75%','max','std','count']].round(1)
    ctype_count_df = ctype_count_df.sort_values('sum',ascending=False)
    
    if phenotype == '':
        cell_mapping_characteristics_by_phenotype['all'] = ctype_count_df
    else:
        cell_mapping_characteristics_by_phenotype[phenotype] = ctype_count_df

# %%
# Save the results into an excel-file

# Create a Pandas Excel writer using the file name
writer = pd.ExcelWriter('./supplementary_tables/inferred_cell_type_counts_key_statistics.xlsx', engine='xlsxwriter')

# Iterate through each key-value pair in the dictionary
for key, value in cell_mapping_characteristics_by_phenotype.items():
    # Write each dataframe to a separate sheet in the Excel file
    value.to_excel(writer, sheet_name=key)

# Save and close the Excel writer
writer.save()
    

# %%
# Test whether the inferred number of tumor spots varies across the dataset
from scipy.stats import f_oneway

inferred_tumor_counts_dict = {}
for phenotype in ['BPH','TRNA','NEADT','CRPC']:
    inferred_tumor_counts_dict[phenotype] = df.loc[df.index.str.contains(phenotype)]['tumor'].tolist()

f_stat, pval = f_oneway(*inferred_tumor_counts_dict.values())

pval

# %%
# Run NMF to find 'tissue regions'
nmf_res_dict = {}
nmf_cell_weights_dict = {}
adata = cell_mapping_dat.copy()

# Set the range of components
component_range = range(5, 13)

# Perform NMF
for n_components in component_range:
    nmf = NMF(n_components=n_components,random_state=3456372)
    W = nmf.fit_transform(adata.X)
    H = nmf.components_

    nmf_res = pd.DataFrame(H.T,
                        index=adata.var_names,
                        columns=list(np.arange(0,n_components)+1),)

    nmf_res_dict[n_components] = nmf_res
    nmf_cell_weights_dict[n_components] = W

# %%
# Get inferred cell counts into a dataframe and get the prevalence order for NMF plots
df = pd.DataFrame(cell_mapping_dat.X, columns=cell_mapping_dat.var.index,index=cell_mapping_dat.obs.index)
celltype_order = df.sum().sort_values(ascending=True).index

# Plot the nmf results on two rows
# Create subplots with 2 rows and 4 columns
sns.set_theme(style='white',font_scale=1)
fig, axs = plt.subplots(3, 3, figsize=(20, 25))

for i, n_components in enumerate(component_range):

    nmf_res_for_plotting = nmf_res_dict[n_components].copy()
    nmf_res_for_plotting = nmf_res_for_plotting.loc[celltype_order]

    # Determine the position of the subplot on the grid
    row = i // 3
    col = i % 3
    
    # Plot the heatmap on the corresponding subplot
    sns.heatmap(nmf_res_for_plotting, cmap='Blues', square=True, ax=axs[row, col], vmax=100,cbar=False)


axs[2,2].axis('off')
plt.tight_layout()

#plt.savefig('plots/nmf_components.pdf')
plt.show()

# %%
# Plot the nmf results on two rows
# Create subplots with 2 rows and 4 columns
fig, ax = plt.subplots(figsize=(5, 8))

n_components = 8

# Drop the redundant cell types prior to plotting
nmf_res_for_plotting_subset = nmf_res_dict[n_components].copy()
nmf_res_for_plotting_subset = nmf_res_for_plotting_subset.loc[celltype_order]
# Plot the heatmap on the corresponding subplot
sns.heatmap(nmf_res_for_plotting_subset, cmap='Blues', square=True, ax=ax, vmax=100,cbar=True)

plt.tight_layout()

#plt.savefig('plots/nmf_components_with_colorbar.pdf')
plt.show()

# %%

# Create a Pandas Excel writer using the file name
writer = pd.ExcelWriter('./supplementary_tables/nmf_component_weights_for_celltypes.xlsx', engine='xlsxwriter')

# Iterate through each key-value pair in the dictionary
for key, value in nmf_res_dict.items():
    # Write each dataframe to a separate sheet in the Excel file
    
    value.loc[celltype_order].to_excel(writer, sheet_name=str(key))

# Save and close the Excel writer
writer.save()

# %%
# Select the best number of factors and annotate them
tissue_region_names = {1:'Fibroblast', 2:'Luminal epithelium', 3:'Tumor', 4:'Club epithelium',
                      5:'Muscle', 6:'Basal epithelium', 7:'Immune', 8:'Endothelium'}

# Sorting out the colors for tissue regions (Set2 palette)
sorted_region_names = [tissue_region_names[i] for i in [3,2,6,4,7,8,1,5]]

tissue_region_colors = ['#fc8d62','#8da0cb','#66c2a5','#ffd92f','#a6d854','#e78ac3','#e5c494','#b3b3b3']
region_colors_dict = dict(zip(sorted_region_names,tissue_region_colors))

# Number 8 is the best fit
tissue_regions = nmf_res_dict[n_components].copy().rename(columns=tissue_region_names).loc[celltype_order][sorted_region_names]

# %%
# Read each sample individually – these have been filtered by pathology

cell_type_abundances_pathology_filtered = pd.DataFrame()
for sample in samples:
    # read the extracted cell type counts in here    
    abundance_df = pd.read_csv('./data/inferred_celltype_abundances/'+sample+'_abundances.csv',index_col=0)
    # Save the cell type names into a variable
    cell_types = abundance_df.columns.tolist()
    # Add identifiers
    abundance_df['sample_id'] = sample
    abundance_df['sample_class'] = sample_id_masks[sample].split(' ')[0]
    cell_type_abundances_pathology_filtered = pd.concat([cell_type_abundances_pathology_filtered,abundance_df],axis=0)

# %%
# Spot identity is determined by simply choosing the factor with the highest nmf weight
obs_data = cell_mapping_dat.obs.copy().reset_index(drop=True)
nmf_obs_weights = pd.DataFrame(nmf_cell_weights_dict[n_components],columns=list(tissue_region_names.values()),index=cell_mapping_dat.obs_names)
nmf_obs_annot = nmf_obs_weights.idxmax(axis=1)

obs_data.index = cell_mapping_dat.obs_names
obs_data['nmf_weight_based_regions'] = nmf_obs_annot

print('The number of spots prior to filtering out spots based on pathology: {:d}\n'.format(len(obs_data)))

# Use the pathology annotation to filter out spots with 
# 1) holes 
# 2) lumen
# 3) Poor quality tissue
valid_spot_ids = pd.read_csv('./data/post_qc_and_pathology_annot_valid_spots.csv',index_col=0).index
obs_data = obs_data.loc[valid_spot_ids]

print('Region percentages out of {:d} total spots'.format(len(obs_data)))
(obs_data['nmf_weight_based_regions'].value_counts() / len(obs_data) * 100).round(1)

# %%
# Copy the region annotation into scanpy slide objects, build graph, save and plot

for sample in samples:

    # Read in the data
    slide = sc.read_h5ad('./data/normalized_visium/'+sample+'_normalized.h5ad')

    slide.obs['predicted_region'] = obs_data.loc[slide.obs_names]['nmf_weight_based_regions'].astype('category')

    # Only keep the categories that are present in the sample
    slide_categories = [s for s in sorted_region_names if s in slide.obs['predicted_region'].cat.categories]

    # Sort the categories
    slide.obs['predicted_region'] = slide.obs['predicted_region'].cat.reorder_categories(slide_categories)

    # Get the colors considering some might be missing
    slide.uns['predicted_region_colors'] = [region_colors_dict[region] for region in slide_categories]
    
    # Build the neighbor graph
    sq.gr.spatial_neighbors(slide,n_neighs=6,n_rings=3)

    # Add the relevant column information
    slide.obs['sample_id'] = sample
    slide.obs['sample_class'] = sample_id_masks[sample].split(' ')[0]
    
    # Format the obs columns so that there's no discreprancies
    slide.obs = slide.obs[['in_tissue','array_row','array_col','n_counts','size_factors','sample_id','sample_class','predicted_region']]

    # Save the slide objects to another directory
    slide.write_h5ad('./data/visium_with_regions/'+sample+'_with_regions.h5ad')

    print(sample + ' done.')
    del slide

# %%
# This is in ALL samples
summary_mat = np.zeros((len(sorted_region_names),len(sorted_region_names)))
for sample in samples:
    
    slide = sc.read_h5ad('./data/visium_with_regions/'+sample+'_with_regions.h5ad')
    sq.gr.nhood_enrichment(slide, cluster_key='predicted_region',show_progress_bar=False)
    
    mat = slide.uns['predicted_region_nhood_enrichment']['zscore'].copy()
    mat = np.nan_to_num(mat)

    missing = [c for c in sorted_region_names if c not in list(slide.obs['predicted_region'].cat.categories)]

    for missing_type in missing:

        insert_idx = list(sorted_region_names).index(missing_type)
        mat = np.insert(mat, insert_idx, np.repeat(0,mat.shape[1]),axis = 0)
        mat = np.insert(mat, insert_idx, np.repeat(0,mat.shape[0]),axis = 1)

    summary_mat += mat
sns.set_theme(style='white')
fig,ax = plt.subplots(figsize=(10,8))
df = pd.DataFrame(summary_mat,index=sorted_region_names,columns=sorted_region_names)/len(samples)
sns.heatmap(df,cmap='bwr',ax=ax,annot=True,fmt='.1f',center=0,vmin=-30,vmax=30)
plt.savefig('./plots/tissue_region_interaction_heatmap_all_samples_tampere_arneo.pdf')
plt.show()

# %%

