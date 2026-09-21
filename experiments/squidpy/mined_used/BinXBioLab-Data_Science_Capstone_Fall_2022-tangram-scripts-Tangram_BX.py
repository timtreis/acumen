# mined from: https://github.com/BinXBioLab/Data_Science_Capstone_Fall_2022/blob/3b225c399bb5e3dcb4499dbcb0027d1ee2be7004/tangram/scripts/Tangram_BX.py
# symbols: squidpy.datasets.visium, squidpy.datasets.visium_fluo_image_crop, squidpy.im.ImageContainer, squidpy.im.calculate_image_features, squidpy.im.process, squidpy.im.segment, squidpy.pl.spatial_scatter

#!/usr/bin/env python
# coding: utf-8

# # Tangram: map scseq to stseq analysis
# Bin Xu
# NOTES:
# 11/03/22: Adapted from Ju
# 
# 

# ### #. Mount my drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('mkdir /content/MyDrive')
get_ipython().system('mount --bind /content/drive/My\\ Drive /content/MyDrive')
# the magic % will help with the cd command :)
get_ipython().run_line_magic('cd', '/content')


# ## #0. Install packages (need restart runtime before the analysis!)

# These 4 packages are the basic packages needed to run this code.

# In[ ]:


get_ipython().system('pip install scanpy')
get_ipython().system('pip install squidpy')
get_ipython().system('pip install scikit-image')
get_ipython().system('pip install tangram-sc')
get_ipython().system('pip install gdown')
# to upgrade
get_ipython().system('pip install --upgrade gdown')


# ## #1. Load datasets

# In[2]:


import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
from anndata import AnnData
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage
import seaborn as sns
import tangram as tg

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# scanpy==1.9.1 anndata==0.8.0 umap==0.5.3 numpy==1.20.3 scipy==1.7.1 pandas==1.3.4 scikit-learn==0.24.2 statsmodels==0.12.2 python-igraph==0.10.1 pynndescent==0.5.7
# squidpy==1.2.2
# 
# is the version I personally worked on. You can use tangram.yml to set up the environment.

# This is loading the dataset for this project.
# 
# `adata_sc` stores single cell data.
# 
# `adata_st` stores spatial data.

# ### 1.1 Loading collected single cell data to adata_sc

# In[ ]:


get_ipython().run_line_magic('cd', "'/content/MyDrive/spatial_scRNAseq_test'")


# In[ ]:


#!gdown --fuzzy 'https://drive.google.com/file/d/19O5HyRPUcaeYZut7m_wC7ND6Tb_vNVle/view?usp=share_link'


# In[3]:


adata_sc = sc.read_h5ad("pasca.log1p_liger_med_singleR_noglyc.h5ad")


# In[ ]:


adata_sc


# ### 1.2 Loading spatial data and image data

# In[4]:


adata_st = sq.datasets.visium('V1_Human_Brain_Section_1')   
#adata_st = sq.datasets.visium('V1_Human_Brain_Section_2') 


# In[ ]:


# adata_st = sq.datasets.visium_hne_adata() 
# highest score but uses mouse gene_ids


# In[5]:


img = sq.datasets.visium_fluo_image_crop()
#img = sq.datasets.visium_hne_image_crop()


# ### #1.3 preprocess the adata_st data using scanpy
# 
# https://scanpy-tutorials.readthedocs.io/en/latest/spatial/integration-scanorama.html#Reading-the-data

# In[6]:


sc.pp.calculate_qc_metrics(adata_st, inplace=True)


# In[7]:


adata_st.obs_keys


# In[8]:


sc.pp.normalize_total(adata_st, inplace=True)
sc.pp.log1p(adata_st)
sc.pp.highly_variable_genes(adata_st, flavor="seurat", n_top_genes=2000, inplace=True)


# In[9]:


sc.pp.neighbors(adata_st)
sc.tl.umap(adata_st)
sc.tl.leiden(adata_st, key_added="clusters")


# In[10]:


sc.pl.umap(
    adata_st, color=["clusters"], palette=sc.pl.palettes.default_20
)


# In[11]:


adata_st


# In[12]:


clusters_colors = dict(
    zip([str(i) for i in range(18)], adata_st.uns["clusters_colors"])
)


# In[13]:


plt.rcParams["figure.figsize"] = (8, 8)
ad = adata_st.copy()
sc.pl.spatial(
  ad,
  img_key="hires",
  color=["clusters","n_genes_by_counts"],
  size=1.5,
  palette=[
    v
    for k, v in clusters_colors.items()
    if k in ad.obs.clusters.unique().tolist()
    ],
  legend_loc='right margin',
  show=False,
    )

plt.tight_layout()


# In[14]:


adata_st.write_h5ad('adata_st_processed.h5ad')
adata_st = sc.read_h5ad('adata_st_processed.h5ad')


# ## #2. Run Tangram

# ### 2.1. Data structure

# `{obs_key}` called to see the general structure.

# In[ ]:


adata_sc.obs_keys


# In[ ]:


adata_st.obs_keys


# In[15]:


fig, axs = plt.subplots(1, 2, figsize=(20, 5))
sc.pl.spatial(
    adata_st, color="clusters", alpha=0.7, frameon=False, show=False, ax=axs[0]
)
sc.pl.umap(
    adata_sc, color="nowakowski_med", size=10, frameon=False, show=False, ax=axs[1]
)
plt.tight_layout()


# In[ ]:


adata_sc.obs_keys


# ### #2.2 Select training genes

# This can be used as a solution to some version controls.

# In[16]:


adata_sc.uns['log1p']["base"] = None


# In[33]:


sc.pp.highly_variable_genes(adata_sc)
highvar_df = pd.DataFrame(adata_sc.var["highly_variable"])
highvar_df[highvar_df["highly_variable"] == True]
highvar_markers = list(highvar_df[highvar_df["highly_variable"] == True].index.values)
print(highvar_markers[:20])


# In[17]:


sc.tl.rank_genes_groups(adata_sc, groupby="nowakowski_med", use_raw=False)
markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
markers = list(np.unique(markers_df.melt().value.values))
len(markers)


# In[18]:


markers[1:20]


# In[ ]:


# We can try another option for training gene selection
#sc.tl.rank_genes_groups(adata_sc, groupby="nowakowski_med", use_raw=False)
# sc.pp.highly_variable_genes(adata_sc)


# ### #2.3 Find alignment

# We prepares the data using `pp_adatas`, which does the following:
# - Takes a list of genes from user via the `genes` argument. These genes are used as training genes.
# - Annotates training genes under the `training_genes` field, in `uns` dictionary, of each AnnData. 
# - Ensure consistent gene order in the datasets (_Tangram_ requires that the the $j$-th column in each matrix correspond to the same gene).
# - If the counts for a gene are all zeros in one of the datasets, the gene is removed from the training genes.
# - If a gene is not present in both datasets, the gene is removed from the training genes.

# In[34]:


# This is an essential step for alignment
tg.pp_adatas(adata_sc, adata_st, genes=highvar_markers)


# In[37]:


# This step will take ~25min per 100 eporchs with cpu, it will take ~1min per 100 eporchs with cuda. This step has to be run with GPU option separately.
import torch
torch.cuda.empty_cache()
ad_map = tg.map_cells_to_space(adata_sc, adata_st,
    mode="cells",
#     mode="clusters",
#     cluster_label='cell_subclass',  # .obs field w cell types
    density_prior='rna_count_based',
    num_epochs=50,
#     device="cuda:0",
    device='cpu',
)


# In[47]:


targets = ['PRODH', 'SLC25A1', 'MRPL40', 'TXNRD2', 'TANGO2']
print([i for i in targets if i in highvar_markers])
print([i for i in targets if i in adata_st.uns['training_genes']])


# ### #2.4 Visualize cell type maps

# In[38]:


ad_map.write("ad_map_50.h5ad")
ad_map = sc.read_h5ad("ad_map_50.h5ad")


# In[39]:


ad_map


# To visualize cell types in space, we invoke `project_cell_annotation` to transfer the `annotation` from the mapping to space. We can then call `plot_cell_annotation` to visualize it. You can set the `perc` argument to set the range to the colormap, which would help remove outliers.

# In[40]:


adata_st.obs_keys


# In[41]:


tg.project_cell_annotations(ad_map, adata_st, annotation="nowakowski_med")
annotation_list = list(pd.unique(adata_sc.obs['nowakowski_med']))
tg.plot_cell_annotation_sc(adata_st, annotation_list)


# In[ ]:


ad_map
adata_st.obs_keys


# The first way to get a sense if mapping was successful is to look for known cell type patterns. To get a deeper sense, we can use the helper `plot_training_scores` which gives us four panels:

# In[42]:


tg.plot_training_scores(ad_map, bins=20, alpha=.5)


# - The first panel is a histogram of the simlarity scores for each training gene.
# - In the second panel, each dot is a training gene and we can observe the training score (y-axis) and the sparsity in the scRNA-seq data (x-axis) of each gene. 
# - The third panel is similar to the second one, but contains the gene sparsity of the spatial data. Spatial data are usually more sparse than single cell data, a discrepancy which is often responsible for low quality mapping.
# - In the last panel, we show the training scores as a function of the difference in sparsity between the dataset. For genes with comparable sparsity, the mapped gene expression is very similar to that in the spatial data. However, if a gene is quite sparse in one dataset (typically, the spatial data) but not in other, the mapping score is lower. This occurs as Tangram cannot properly matched the gene pattern because of inconsistent amount of dropouts between the datasets.

# Although the above plots give us a summary of scores at single-gene level, we would need to know _which_ are the genes are mapped with low scores. These information are stored in the dataframe `.uns['train_genes_df']`; this is the dataframe used to build the four plots above.

# In[43]:


ad_map.uns['train_genes_df']


# ### #2.5 Generate new spatial data via aligned single cells

# If the mapping mode is `'cells'`, we can now generate the "new spatial data" using the mapped single cell: this is done via `project_genes`. The function accepts as input a mapping (`adata_map`) and corresponding single cell data (`adata_sc`). The result is a voxel-by-gene `AnnData`, formally similar to `adata_st`, but containing gene expression from the mapped single cell data rather than Visium. For downstream analysis, we always replace `adata_st` with the corresponding `ad_ge`.

# In[48]:


# it takes a few minutes
ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)
ad_ge


# - So far, we only inspected genes used to align the data (training genes), but the mapped single cell data, `ad_ge` contains the whole transcriptome. That includes more than 35k test genes.

# In[49]:


(ad_ge.var.is_training == False).sum()


# In[50]:


#ad_ge.write("ad_ge.h5ad")
# re-imported ad_ge seems not recognized by tangram. might need rerun ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)
ad_ge = sc.read_h5ad('ad_ge.h5ad')


# In[51]:


ad_ge.var


# In[52]:


import gc
gc.collect()


# We can use `plot_genes` to inspect gene expression of test genes as well. Inspecting the test transcriptome is an essential to validate mapping. At the same time, we need to be careful that some prediction might disagree with spatial data because of the technical droputs.
# 
# It is convenient to compute the similarity scores of all genes, which can be done by `compare_spatial_geneexp`. This function accepts two spatial AnnDatas (ie voxel-by-gene), and returns a dataframe with simlarity scores for all genes. Training genes are flagged by the boolean field `is_training`. If we also pass single cell AnnData to `compare_spatial_geneexp` function like below, a dataframe with additional sparsity columns - sparsity_sc (single cell data sparsity) and sparsity_diff (spatial data sparsity - single cell data sparsity) will return. This is required if we want to call `plot_test_scores` function later with the returned datafrme from `compare_spatial_geneexp` function.

# In[53]:


df_all_genes = tg.compare_spatial_geneexp(ad_ge, adata_st, adata_sc)
df_all_genes


# The prediction on test genes can be graphically visualized using `plot_auc`:

# In[54]:


# sns.scatterplot(data=df_all_genes, x='score', y='sparsity_sp', hue='is_training', alpha=.5);  # for legacy
tg.plot_auc(df_all_genes);


# **This above figure is the most important validation plot in _Tangram_.** Each dot represents a gene; the x-axis indicates the score, and the y-axis the sparsity of that gene in the spatial data.  Unsurprisingly, the genes predicted with low score represents very sparse genes in the spatial data, suggesting that the _Tangram_ predictions correct expression in those genes. Note that curve observed above is typical of _Tangram_ mappings: the area under that curve is the most reliable metric we use to evaluate mapping.
# 
# Let's inspect a few predictions. Some of these genes are biologically sparse, but well predicted:

# In[55]:


# we have to use low letter
GOI=['pax6','sox2','foxg1','neurod2','prodh','ranbp1','crkl','hira']


# In[56]:


tg.plot_genes_sc(GOI, adata_measured=adata_st, adata_predicted=ad_ge, perc=0.02)


# In[57]:


plt.rcParams["figure.figsize"] = (20, 20)
ad = ad_ge.copy()
sc.pl.spatial(
  ad,
  img_key="hires",
  color=GOI,
  size=1.5,
  palette=[
    v
    for k, v in clusters_colors.items()
    if k in ad.obs.clusters.unique().tolist()
    ],
  legend_loc='right margin',
  show=False,
    )

plt.tight_layout()


# ### #2.6 Deconvolution via alignment

# The rationale for deconvolving with Tangram, is to constrain the number of mapped single cell profiles. This is different that most deconvolution method. Specifically, we set them equal to the number of segmented cells in the histology, in the following way:
# - We pass `mode='constrained'`. This adds a filter term to the loss function, and a boolean regularizer.
# - We set `target_count` equal to the total number of segmented cells. _Tangram_ will look for the best `target_count` cells to align in space.
# - We pass a `density_prior`, containing the fraction of cells per voxel. 

# In[ ]:


ad_map_d = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode="constrained",
    target_count=adata_st.obs.cell_count.sum(),
    density_prior=np.array(adata_st.obs.cell_count) / adata_st.obs.cell_count.sum(),
    num_epochs=1000,
#     device="cuda:0",
    device='cpu',
)


# In the same way as before, we can plot cell type maps:

# In[101]:


ad_map


# In[92]:


tg.project_cell_annotations(ad_map, adata_st, annotation="nowakowski_med")
annotation_list = list(pd.unique(adata_sc.obs['nowakowski_med']))

# This old tg.plot_cell_annotation does not work!!!
#tg.plot_cell_annotation(adata_st, annotation_list, perc=0.02)

tg.plot_cell_annotation_sc(adata_st, annotation_list)


# We validate mapping by inspecting the test transcriptome:

# In[93]:


ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)
df_all_genes = tg.compare_spatial_geneexp(ad_ge, adata_st, adata_sc)
tg.plot_auc(df_all_genes);


# And here comes the key part, where we will use the results of the previous deconvolution steps. Previously, we computed the absolute numbers of unique segmentation objects under each spot, together with their centroids. Let's extract them in the right format useful for _Tangram_. In the resulting dataframe, each row represents a single segmentation object (a cell). We also have the image coordinates as well as the unique centroid ID, which is a string that contains both the spot ID and a numerical index. _Tangram_ provides a convenient function to export the mapping between spot ID and segmentation ID to `adata.uns`.

# In[94]:


tg.create_segment_cell_df(adata_st)
adata_st.uns["tangram_cell_segmentation"].head()


# We can use `tangram.count_cell_annotation()` to map cell types as result of the deconvolution step to putative segmentation ID.

# In[96]:


tg.count_cell_annotations(
    ad_map,
    adata_sc,
    adata_st,
    annotation="nowakowski_med",
)
adata_st.obsm["tangram_ct_count"].head()


# And finally export the results in a new `AnnData` object.

# In[97]:


adata_segment = tg.deconvolve_cell_annotations(adata_st)
adata_segment.obs.head()


# Note that the AnnData object does not contain counts, but only cell type annotations, as results of the Tangram mapping.  Nevertheless, it's convenient to create such AnnData object for visualization purposes. Below you can appreciate how each dot is now not a Visium spot anymore, but a single unique segmentation object, with the mapped cell type.

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sc.pl.spatial(
    adata_segment,
    color="cluster",
    size=0.4,
    show=False,
    frameon=False,
    alpha_img=0.2,
    legend_fontsize=20,
    ax=ax,
)


# ### #2.7 Generate high resolution maps

# In[61]:


sf = adata_st.uns['spatial']['V1_Human_Brain_Section_1']["scalefactors"]["tissue_hires_scalef"]
img = sq.im.ImageContainer(adata_st.uns['spatial']['V1_Human_Brain_Section_1']['images']['hires'],
                           scale = sf, library_id = 'V1_Human_Brain_Section_1')


# In[62]:


sq.im.process(img=img, layer="image", method="smooth")
sq.im.segment(                                                                                
    img=img,
    layer="image_smooth",
    method="watershed",
    channel=0,
)


# In[63]:


adata_st.obs.columns


# In[64]:


inset_y = 1500
inset_x = 1700
inset_sy = 400
inset_sx = 500

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
# spatial how?
sc.pl.spatial(
    adata_st, 
    color="rna_count_based_density",   
    alpha=0.7, frameon=False, show=False, ax=axs[0], title=""
)
axs[0].set_title("Clusters", fontdict={"fontsize": 20})
rect = mpl.patches.Rectangle(
    (inset_y * sf, inset_x * sf),
    width=inset_sx * sf,
    height=inset_sy * sf,
    ec="yellow",
    lw=4,
    fill=False,
)
axs[0].add_patch(rect)

axs[0].axes.xaxis.label.set_visible(False)
axs[0].axes.yaxis.label.set_visible(False)

axs[1].imshow(
    img["image"][inset_y : inset_y + inset_sy, inset_x : inset_x + inset_sx, 0, 0]
    / 65536,
    interpolation="none",
)
axs[1].grid(False)
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_title("DAPI", fontdict={"fontsize": 20})

crop = img["segmented_watershed"][
    inset_y : inset_y + inset_sy, inset_x : inset_x + inset_sx
].values.squeeze(-1)
crop = skimage.segmentation.relabel_sequential(crop)[0]
cmap = plt.cm.plasma
cmap.set_under(color="black")
axs[2].imshow(crop, interpolation="none", cmap=cmap, vmin=0.001)
axs[2].grid(False)
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_title("Nucleous segmentation", fontdict={"fontsize": 20});


# In[65]:


features_kwargs = {
    "segmentation": {
        "label_layer": "segmented_watershed",
        "props": ["label", "centroid"],
        "channels": [1, 2],
    }
}
# calculate segmentation features
sq.im.calculate_image_features(
    adata_st,
    img,
    layer="image",
    key_added="image_features",
    n_jobs=4,
    features_kwargs=features_kwargs,
    features="segmentation",
    mask_circle=True,
)


# In[67]:


adata_st.obs["cell_count"] = adata_st.obsm["image_features"]["segmentation_label"]
sc.pl.spatial(adata_st, color=["cell_count"], frameon=False)


# In[79]:


adata_st.obsm['image_features']


# In[80]:


adata_st.obs["cell_count"] = adata_st.obsm["image_features"]["segmentation_label"]
sc.pl.spatial(adata_st, color=['rna_count_based_density'], frameon=False)


# In[86]:


adata_st.uns["tangram_cell_segmentation"].head()


# In[ ]:


# avoid write into adata_st directly
ad = adata_st.copy()
sq.pl.spatial_scatter(
    ad,
    color=["EN-V1","RG-early"],
)


# In[122]:


ad_map.obs_keys


# In[89]:


ann_vc = ad_map.obs['nowakowski_med'].value_counts()


# In[76]:


ad.obs = pd.concat([adata_st.obs, adata_st.obsm["tangram_ct_pred"]], axis=1)
annotation_list = ad_map.obs['nowakowski_med']
annotation_list


# In[91]:


tg.count_cell_annotations(
    adata_map = ad_map,
    adata_sc = adata_sc,
    adata_sp = ad,
    annotation = 'nowakowski_med'
)
# nowakowski_med
adata_st.obsm["tangram_ct_count"].head()


# In[98]:


ad_map


# In[ ]:


adata_segment = tg.deconvolve_cell_annotations(adata_st)
adata_segment.obs.head()


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sc.pl.spatial(
    adata_segment,
    color="cluster",
    size=0.4,
    show=False,
    frameon=False,
    alpha_img=0.2,
    legend_fontsize=20,
    ax=ax,
)

