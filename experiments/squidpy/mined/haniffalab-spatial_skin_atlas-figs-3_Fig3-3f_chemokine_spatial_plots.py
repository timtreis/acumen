# mined from: https://github.com/haniffalab/spatial_skin_atlas/blob/56fa9dee5276a837df3e6edb94c5d344f9997dac/figs/3_Fig3/3f_chemokine_spatial_plots.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import spatialdata as sd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq
import anndata as ad

import pickle

# %%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%matplotlib inline
sc.settings.figdir = "fig3"
sc.settings.set_figure_params(dpi_save=300, facecolor="white", frameon=False, figsize=(20,20))

# %%
#adata=sc.read_h5ad('/nfs/team298/ls34/adult_skin/final_adatas/adata_combined_new.h5ad.v8_nohealthy')
#PATH = '/nfs/team298/ls34/adult_skin/final_adatas/adata_combined_new.h5ad.final.filtered'
PATH = '/nfs/team298/ls34/adult_skin/final_adatas/adata_combined_inc_relapse_webportal.h5ad'

adata=sc.read_h5ad(PATH)
adata=adata[adata.obs["tech"]=="xenium"].copy()
#adata=sc.read_h5ad('/nfs/team298/ls34/adult_skin/final_adatas/adata_combined_new.h5ad.final.filtered.scrna')


# %%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%
adata_5k_i=adata[adata.obs["Site_status"].str.startswith("L")]

# %%
with open('/nfs/team298/ls34/niche_colors.pkl', 'rb') as f:
    colors_new2 = pickle.load(f)
adata_5k_i.obs['niche19'] =adata_5k_i.obs['niche19'] .astype('category')
adata_5k_i.uns['niche19_colors'] = [colors_new2.get(cat, '#D3D3D3')  # Default to light grey if not found
                                     for cat in adata_5k_i.obs['niche19'].cat.categories]

# %%
adata_5k_i.obs['niche19'] =adata_5k_i.obs['niche19'] .astype('category')
adata_5k_i.uns['niche19_colors'] = [colors_new2.get(cat, '#D3D3D3')  # Default to light grey if not found
                                     for cat in adata_5k_i.obs['niche19'].cat.categories]

# %%
"""
find niche
"""
i=0
# try:i
#     adata_5k.uns.pop('niche_name_colors')
# except:
#     1
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False, figsize=(6,5))

sample_counts = adata_5k_i.obs["Sanger patient ID"].value_counts()
samples_to_keep = sample_counts[sample_counts > 1000].index

ORDER = []
for i,DONOR_ID in enumerate(["CE3", "BK39"]):
   # print(i+1, "/", len(samples_to_keep))
    adata_i =  adata_5k_i[adata_5k_i.obs["Sanger patient ID"]==DONOR_ID]
    STATUS =  adata_i.obs["Site_status"].unique()[0]

    #RUN_ID = list(adata_i.obs.RUN_ID.unique())[0]
    INFO_ID = list(adata_i.obs.info_id6.unique())[0]
    print(INFO_ID)
    #del(adata_i.uns["combined_annotation2_colors"])
    #ORDER.append(tissue_section_id)
 
#             print(tissue_section_id)
   # adata_i=adata_i[adata_i.obs["niche11"].str.startswith("Epi")]
    sq.pl.spatial_scatter(
        adata_i,#[adata_i.obs["Timepoint"].str.startswith("Les")],
        library_id="spatial",
        shape=None,
        color=["CCL17", "CCL19", "CCL22"],
        size=5,
        #title=str(STATUS) + "_" + DONOR_ID + "\n" + str(adata_i.shape[0]),
        #title=INFO_ID, #+ "_nonlesional",
        #legend_loc=None,
        # legend_loc="on data",
                edgecolor="black",
        linewidth=0.03,
                legend_fontsize=12,
        cmap="Reds",
                vmax=2,
                save=f"3f_{DONOR_ID}_chemokines.pdf"


    

        #ax=ax,
        #legend_loc="on data"  # Disable the legend for each subplot
    )
#     if i>1:
#         break
    adata_i.obs["niche19"]=[x if x.startswith("Tz") else "Other" for x in adata_i.obs["niche19"]]
    adata_i.uns.pop('niche19_colors')
    adata_i.obs['niche19'] =adata_i.obs['niche19'] .astype('category')
    adata_i.uns['niche19_colors'] = [colors_new2.get(cat, '#D3D3D3')  # Default to light grey if not found
                                     for cat in adata_i.obs['niche19'].cat.categories]
    sq.pl.spatial_scatter(
        adata_i,#[adata_i.obs["Timepoint"].str.startswith("Les")],
        library_id="spatial",
        shape=None,
        color="niche19",
        size=5,
        title="",
        #title=INFO_ID, #+ "_nonlesional",
        legend_loc=None,
        # legend_loc="on data",
                edgecolor="black",
        linewidth=0.01,
                legend_fontsize=12,
        save=f"3f_{DONOR_ID}.pdf"
        #cmap="Reds",
               # vmax=2,

    

       
    )

# %%

i=0

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False, figsize=(6,5))

sample_counts = adata_5k_i.obs["Sanger patient ID"].value_counts()
samples_to_keep = sample_counts[sample_counts > 1000].index

ORDER = []
for i,DONOR_ID in enumerate(["CE3", "BK39"]):
   # print(i+1, "/", len(samples_to_keep))
    adata_i =  adata_5k_i[adata_5k_i.obs["Sanger patient ID"]==DONOR_ID]
    STATUS =  adata_i.obs["Site_status"].unique()[0]

    #RUN_ID = list(adata_i.obs.RUN_ID.unique())[0]
    INFO_ID = list(adata_i.obs.info_id2.unique())[0]
    print(INFO_ID)
    #del(adata_i.uns["combined_annotation2_colors"])
    #ORDER.append(tissue_section_id)
#     niche_names_found = adata_i.obs["niche_name"].unique()
   
#     if NICHE in niche_names_found:
#         adata_ii =  adata_i[adata_i.obs["niche_name"]==NICHE]
#         if adata_ii.shape[0] > 10:
#     #         if i ==0:

#     #             sq.pl.spatial_scatter(
#     #                 adata_i,
#     #                 library_id="spatial",
#     #                 shape=None,
#     #                 color="niche_name",
#     #                 size=1,
#     #                 vmax=1,
#     #                 title=tissue_section_id,
#     #                 #ax=ax,
#     #                 #legend_loc="on data"  # Disable the legend for each subplot
#     #             )       
#             i=i+1

#             print(tissue_section_id)
   # adata_i=adata_i[adata_i.obs["niche11"].str.startswith("Epi")]
    sq.pl.spatial_scatter(
        adata_i,#[adata_i.obs["Timepoint"].str.startswith("Les")],
        library_id="spatial",
        shape=None,
        color="CCL17",
        size=5,
        title=str(STATUS) + "_" + DONOR_ID + "\n" + str(adata_i.shape[0]),
        #title=INFO_ID, #+ "_nonlesional",
        #legend_loc=None,
        # legend_loc="on data",
                edgecolor="black",
        linewidth=0.03,
                legend_fontsize=12,
        cmap="Reds",
                vmax=2,

    

        #ax=ax,
        #legend_loc="on data"  # Disable the legend for each subplot
    )

# %%
"""
find niche
"""
i=0
# try:i
#     adata_5k.uns.pop('niche_name_colors')
# except:
#     1
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False, figsize=(6,5))

sample_counts = adata_5k_i.obs["Sanger patient ID"].value_counts()
samples_to_keep = sample_counts[sample_counts > 1000].index

ORDER = []
for i,DONOR_ID in enumerate(["CE3", "BK39"]):
   # print(i+1, "/", len(samples_to_keep))
    adata_i =  adata_5k_i[adata_5k_i.obs["Sanger patient ID"]==DONOR_ID]
    STATUS =  adata_i.obs["Site_status"].unique()[0]

    #RUN_ID = list(adata_i.obs.RUN_ID.unique())[0]
    INFO_ID = list(adata_i.obs.info_id2.unique())[0]
    print(INFO_ID)
    #del(adata_i.uns["combined_annotation2_colors"])
    #ORDER.append(tissue_section_id)
#     niche_names_found = adata_i.obs["niche_name"].unique()
   
#     if NICHE in niche_names_found:
#         adata_ii =  adata_i[adata_i.obs["niche_name"]==NICHE]
#         if adata_ii.shape[0] > 10:
#     #         if i ==0:

#     #             sq.pl.spatial_scatter(
#     #                 adata_i,
#     #                 library_id="spatial",
#     #                 shape=None,
#     #                 color="niche_name",
#     #                 size=1,
#     #                 vmax=1,
#     #                 title=tissue_section_id,
#     #                 #ax=ax,
#     #                 #legend_loc="on data"  # Disable the legend for each subplot
#     #             )       
#             i=i+1

#             print(tissue_section_id)
   # adata_i=adata_i[adata_i.obs["niche11"].str.startswith("Epi")]
    sq.pl.spatial_scatter(
        adata_i,#[adata_i.obs["Timepoint"].str.startswith("Les")],
        library_id="spatial",
        shape=None,
        color="CCL22",
        size=5,
        title=str(STATUS) + "_" + DONOR_ID + "\n" + str(adata_i.shape[0]),
        #title=INFO_ID, #+ "_nonlesional",
        #legend_loc=None,
        # legend_loc="on data",
                edgecolor="black",
        linewidth=0.03,
                legend_fontsize=12,
        cmap="Reds",
                vmax=2,

    

        #ax=ax,
        #legend_loc="on data"  # Disable the legend for each subplot
    )

# %%
adata_5k_i.obs["Site_status_simple"].value_counts()

# %%
sc.pl.dotplot(adata_5k_i, 
              ["CCL19", "CCL17", "CCL22"],
              groupby="niche12",
              dendrogram=False, 
                standard_scale="var",
             )

# %%
sc.pl.dotplot(adata_5k_i[adata_5k_i.obs["niche12"]=="Tzone-like"], 
              ["CCL19", "CCL17", "CCL22"],
              groupby="disease_overall",
              dendrogram=False, 
                standard_scale="var",
             )
