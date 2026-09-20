# mined from: https://github.com/alonmillet/apoe-ad-age-atlas/blob/754663ba478a4c9b8a869d2bb90b401bce35a566/Xenium.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq

# %%
from glob import glob
dirs = glob('output*')
samps = ["3","2","1","1","2","3"]
genos = ['E3','E3','E3','E4','E4','E4']

# %%
adata_list = []
for i in list(range(0,6)):
    temp_adata = sc.read_10x_h5(dirs[i]+'/cell_feature_matrix.h5')
    temp_df = pd.read_csv(dirs[i]+'/cells.csv.gz', compression = "gzip")
    temp_df.set_index(temp_adata.obs_names, inplace=True)
    temp_adata.obs = temp_df.copy()
    temp_adata.obsm['spatial'] = temp_adata.obs[['x_centroid','y_centroid']].copy().to_numpy()
    sc.pp.calculate_qc_metrics(temp_adata, percent_top=(10, 20, 50, 150), inplace=True)
    sc.pp.filter_cells(temp_adata, min_counts=5)
    sc.pp.filter_genes(temp_adata, min_cells=5)
    temp_adata.obs['genotype'] = genos[i]
    temp_adata.obs['sample'] = genos[i]+"_"+samps[i]
    adata_list.append(temp_adata)

# %%
import anndata as ad
adata = ad.concat(adata_list)
adata.obs_names_make_unique()

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.tl.leiden(adata, resolution = 0.3)

# %%
sc.pl.umap(adata,color=["total_counts","n_genes_by_counts","leiden","sample","genotype"],wspace=0.4)

# %%
sq.pl.spatial_scatter(adata, library_key = "sample", shape=None, color=["leiden"],wspace=0.4)

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, library_key = "sample")

# %%
adata = sc.read_h5ad("merged_xenium.h5ad")
adata.uns['log1p']["base"] = None # temporary fix for annoying bug

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)

# %%
# define superclusters
superclusters = ["Oligodendrocyte","Neuron","Astrocyte","OPC",
                 "Immune","Neuron","Astrocyte","Immune",
                 "Astrocyte","Astrocyte","Oligodendrocyte","Astrocyte",
                 "Astrocyte","Oligodendrocyte","Oligodendrocyte","Oligodendrocyte",
                 "Astrocyte","Oligodendrocyte","Oligodendrocyte","Oligodendrocyte",
                "Immune","Immune","Astrocyte","OPC",
                "Oligodendrocyte","Astrocyte","Neuron","Neuron",
                "Neuron","Neuron","Oligodendrocyte","Oligodendrocyte",
                "Astrocyte","Astrocyte","Astrocyte","Immune",
                "Neuron","Neuron","Immune","Astrocyte",
                "Astrocyte","Neuron","OPC","Neuron",
                "Astrocyte","Astrocyte","Neuron","Oligodendrocyte",
                "Neuron","Neuron","Immune","Neuron",
                "Oligodendrocyte","Neuron","OPC","Immune",
                "Astrocyte","Oligodendrocyte","Oligodendrocyte","Oligodendrocyte",
                "Neuron"]

# %%
cluster_dict = dict(zip(list(range(0,61)),superclusters))

# %%
adata.obs['supercluster'] = adata.obs['leiden'].astype(int).map(cluster_dict).astype('category')

# %%
from collections import Counter
Counter(adata.obs['supercluster'])

# %%
immune_clusts = [i for i,x in enumerate(superclusters) if x=="Immune"]
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', groups = immune_clusts)
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', groups = [50,55,4,35])
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)

# %%
sc.pl.violin(adata,keys=["APOE"],groupby="leiden",order=map(str,immune_clusts))

# %%
immune_dict = dict(zip(immune_clusts,["FLT1-hi Inflammatory Microglia","TIMs","Homeostatic Microglia","APP-hi Homeostatic Microglia",
                                      "APOE-hi Inflammatory Microglia","TIMs","APOE-hi Inflammatory Microglia","FLT1-hi Inflammatory Microglia"]))

# %%
neuron_clusts = [i for i,x in enumerate(superclusters) if x=="Neuron"]
olig_clusts = [i for i,x in enumerate(superclusters) if x=="Oligodendrocyte"]
astro_clusts = [i for i,x in enumerate(superclusters) if x=="Astrocyte"]
opc_clusts = [i for i,x in enumerate(superclusters) if x=="OPC"]

neuron_dict = dict(zip(neuron_clusts,["Neuron_" + s for s in map(str,range(0,len(neuron_clusts)))]))
olig_dict = dict(zip(olig_clusts,["Oligodendrocyte_" + s for s in map(str,range(0,len(olig_clusts)))]))
astro_dict = dict(zip(astro_clusts,["Astrocyte_" + s for s in map(str,range(0,len(astro_clusts)))]))
opc_dict = dict(zip(opc_clusts,["OPC_" + s for s in map(str,range(0,len(opc_clusts)))]))

# %%
super_dict = immune_dict|neuron_dict|olig_dict|astro_dict|opc_dict

# %%
adata.obs['subcluster'] = adata.obs['leiden'].astype(int).map(super_dict).astype('category')

# %%
# reorder clustering
sub_levels = ['Homeostatic Microglia','APP-hi Homeostatic Microglia',
         "FLT1-hi Inflammatory Microglia","APOE-hi Inflammatory Microglia",
         "TIMs"] + list(neuron_dict.values()) + list(olig_dict.values()) + list(astro_dict.values()) + list(opc_dict.values()) 
super_levels = ['Neuron','Oligodendrocyte','Astrocyte','Immune','OPC']

adata.obs['subcluster'] = adata.obs['subcluster'].cat.reorder_categories(sub_levels)
adata.obs['supercluster'] = adata.obs['supercluster'].cat.reorder_categories(super_levels)

# %%
# filter out very rare clusters for clarity
adata_clustfilt = adata[((adata.obs['supercluster'] == "Immune") | (adata.obs['leiden'].astype(int) < 12))]
sq.pl.spatial_scatter(adata_clustfilt, library_key = "sample", shape=None, color=["subcluster"],wspace=0.8, ncols = 3)

# %%
neu0 = adata[adata.obs['subcluster'] == "Neuron_0"]
sc.pp.pca(neu0)
sc.pp.neighbors(neu0)
sc.tl.umap(neu0)
sc.tl.leiden(neu0, resolution = 0.9)
sc.pl.umap(neu0,color=["leiden"])

# %%
assignments = ["Mixed Border Neurons","L6 Neurons","L3 Neurons","L4 Neurons","L6 Neurons","Homeostatic Microglia","L2 Neurons","L5 Neurons",
               "L3 Neurons","OPC_0","Mixed Neurons","L4 Neurons"]+["Mixed Neurons"]*9 + ["L1 Neurons"] + ["Mixed Neurons"]*15

neu0_dict = dict(zip(neu0.obs['leiden'].cat.categories,assignments))
neu0.obs['subcluster_neu0'] = neu0.obs['leiden'].map(neu0_dict).astype('category')
sub_levels = ["L1 Neurons","L2 Neurons","L3 Neurons","L4 Neurons","L5 Neurons","L6 Neurons",
              "Mixed Border Neurons","Mixed Neurons","Homeostatic Microglia","OPC_0"]
neu0.obs['subcluster_neu0'] = neu0.obs['subcluster_neu0'].cat.reorder_categories(sub_levels)
sq.pl.spatial_scatter(neu0, library_key = "sample", shape=None, color=["subcluster_neu0"],wspace=0.8, ncols = 3)

# %%
neu0_conversion_dict = neu0.obs['subcluster_neu0'].to_dict()
other_neurons_conversion_dict = dict(zip(["Neuron_" + s for s in map(str,range(1,16))],np.repeat("Mixed Neurons",15)))

# %%
adata.obs['subcluster'] = adata.obs.index.to_series().map(neu0_conversion_dict).fillna(adata.obs['subcluster'])
adata.obs['subcluster'] = adata.obs['subcluster'].map(other_neurons_conversion_dict).fillna(adata.obs['subcluster'])
adata.obs['subcluster'].unique()

# %%
adata.obs['subcluster'] = adata.obs['subcluster'].replace("OPC_0","OPCs")
adata.obs['subcluster'] = adata.obs['subcluster'].replace("OPC_1","OPCs")
adata.obs['subcluster'] = adata.obs['subcluster'].replace("OPC_2","OPCs")
adata.obs['subcluster'] = adata.obs['subcluster'].replace("OPC_3","OPCs")

# %%
olg = adata[adata.obs['supercluster'] == "Oligodendrocyte"]
sc.pp.pca(olg)
sc.pp.neighbors(olg)
sc.tl.umap(olg)
sc.tl.leiden(olg, resolution = 0.3)
sc.pl.umap(olg,color=["leiden"])

# %%
olig_dict = {"0":"ERMN-hi Oligodendrocytes","1":"CNTN2-hi Oligodendrocytes","2":"EFHD1-hi Oligodendrocytes",
             "3":"CNTN2-hi Oligodendrocytes","4":"CNTN2-hi Oligodendrocytes","5":"CNTN2-hi Oligodendrocytes",
             "6":"EFHD1-hi Oligodendrocytes","7":"EFHD1-hi Oligodendrocytes","8":"EFHD1-hi Oligodendrocytes",
             "9":"CNTN2-hi Oligodendrocytes","10":"EFHD1-hi Oligodendrocytes","11":"ERMN-hi Oligodendrocytes",
             "12":"ERMN-hi Oligodendrocytes","13":"EFHD1-hi Oligodendrocytes","14":"CNTN2-hi Oligodendrocytes",
             "15":"ERMN-hi Oligodendrocytes","16":"ERMN-hi Oligodendrocytes","17":"EFHD1-hi Oligodendrocytes",
            "18":"CNTN2-hi Oligodendrocytes","19":"ERMN-hi Oligodendrocytes","20":"CNTN2-hi Oligodendrocytes","21":"EFHD1-hi Oligodendrocytes"}
olg.obs['subcluster_olg'] = olg.obs['leiden'].map(olig_dict).astype('category')
sub_levels = ["CNTN2-hi Oligodendrocytes","ERMN-hi Oligodendrocytes","EFHD1-hi Oligodendrocytes"]
olg.obs['subcluster_olg'] = olg.obs['subcluster_olg'].cat.reorder_categories(sub_levels)
sq.pl.spatial_scatter(olg, library_key = "sample", shape=None, color=["subcluster_olg"],wspace=0.8, ncols = 3)

# %%
olg_conversion_dict = olg.obs['subcluster_olg'].to_dict()
adata.obs['subcluster'] = adata.obs.index.to_series().map(olg_conversion_dict).fillna(adata.obs['subcluster'])
adata.obs['subcluster'].unique()

# %%
ast = adata[adata.obs['supercluster'] == "Astrocyte"]
sc.pp.pca(ast)
sc.pp.neighbors(ast)
sc.tl.umap(ast)
sc.tl.leiden(ast, resolution = 0.3)
sc.pl.umap(ast,color=["leiden"])

# %%
ast_dict = dict(zip(map(str,list(range(0,40))),
                    ["GJA1-hi Astrocytes","VLMCs","ERMN-hi Oligodendrocytes","AQP4-hi Astrocytes","APOE-hi Astrocytes","GJA1-hi Astrocytes",
                    "APP-hi Astrocytes","APP-hi Astrocytes","GJA1-hi Astrocytes","APP-hi Astrocytes","AQP4-hi Astrocytes",
                    "GJA1-hi Astrocytes","APP-hi Astrocytes","APP-hi Astrocytes","APP-hi Astrocytes","GJA1-hi Astrocytes",
                    "APP-hi Astrocytes","AQP4-hi Astrocytes","AQP4-hi Astrocytes","APP-hi Astrocytes","EFHD1-hi Oligodendrocytes",
                    "APP-hi Astrocytes","EFHD1-hi Oligodendrocytes","GJA1-hi Astrocytes","GJA1-hi Astrocytes","AQP4-hi Astrocytes",
                    "AQP4-hi Astrocytes","EFHD1-hi Oligodendrocytes","EFHD1-hi Oligodendrocytes","GJA1-hi Astrocytes",
                    "APOE-hi Astrocytes","APOE-hi Astrocytes","APP-hi Astrocytes","OPCs","APOE-hi Astrocytes","GJA1-hi Astrocytes","AQP4-hi Astrocytes",
                     "APOE-hi Astrocytes","GJA1-hi Astrocytes","GJA1-hi Astrocytes"]))
ast.obs['subcluster_ast'] = ast.obs['leiden'].map(ast_dict).astype('category')
ast_conversion_dict = ast.obs['subcluster_ast'].to_dict()
adata.obs['subcluster'] = adata.obs.index.to_series().map(ast_conversion_dict).fillna(adata.obs['subcluster'])
sorted(adata.obs['subcluster'].unique())

# %%
clust_order = ["L1 Neurons","L2 Neurons","L3 Neurons","L4 Neurons","L5 Neurons","L6 Neurons","Mixed Border Neurons","Mixed Neurons",
              "CNTN2-hi Oligodendrocytes","EFHD1-hi Oligodendrocytes","ERMN-hi Oligodendrocytes","AQP4-hi Astrocytes","GJA1-hi Astrocytes",
              "APP-hi Astrocytes","APOE-hi Astrocytes",'Homeostatic Microglia','APP-hi Homeostatic Microglia',
               "FLT1-hi Inflammatory Microglia","APOE-hi Inflammatory Microglia","TIMs","OPCs","VLMCs"]
adata.obs['subclust'] = adata.obs['subcluster'].astype('category').cat.reorder_categories(clust_order)

# %%
superclust_dict = dict(zip(clust_order,
                       np.concatenate([np.repeat("Neurons",8),np.repeat("Oligodendrocytes",3),np.repeat("Astrocytes",4),
                                       np.repeat("Microglia",5),np.repeat("Other",2)])))
superclust_order = ["Neurons","Oligodendrocytes","Astrocytes","Microglia","Other"]
adata.obs['superclust'] = adata.obs['subclust'].map(superclust_dict).astype('category').cat.reorder_categories(superclust_order)

# %%
adata.obs = adata.obs.drop(['supercluster','subcluster'], axis=1)

# %%
sq.pl.spatial_scatter(adata, library_key = "sample", shape=None, color=["subclust"],wspace=0.9,ncols=3,dpi=600,save="all_sections_scatter.png",
                      title = adata.obs['sample'].cat.categories.values)

# %%
sq.pl.spatial_scatter(adata, library_key = "sample", shape=None, color=["subclust"],wspace=0.9,ncols=3)

# %%
df = adata.obs
df.to_csv("xenium_md.csv")

# %%
adata.write("merged_xenium.h5ad", compression = "gzip")

# %%
adata_subsample = sc.pp.subsample(adata, fraction=0.5, copy=True)
sq.gr.co_occurrence(
    adata_subsample,
    cluster_key="subclust",
    n_jobs = 32
)

# %%
sq.pl.co_occurrence(
    adata_subsample,
    cluster_key="subclust",
    clusters="TIMs",
    figsize=(10, 12),
    save = "cooccurrence.png",
    dpi=600
)

# %%
adata_subsample_e3 = adata_subsample[adata_subsample.obs['genotype'] == "E3"]
adata_subsample_e4 = adata_subsample[adata_subsample.obs['genotype'] == "E4"]

sq.gr.co_occurrence(
    adata_subsample_e3,
    cluster_key="subclust",
    n_jobs = 32
)
sq.gr.co_occurrence(
    adata_subsample_e4,
    cluster_key="subclust",
    n_jobs = 32
)

# %%
sq.pl.co_occurrence(
    adata_subsample_e3,
    cluster_key="subclust",
    clusters="TIMs",
    figsize=(10, 8),
    save = "cooccurrence_e3.png",
    dpi=600
)

# %%
sq.pl.co_occurrence(
    adata_subsample_e4,
    cluster_key="subclust",
    clusters="TIMs",
    figsize=(10, 8),
    save = "cooccurrence_e4.png",
    dpi=600
)

# %%
occurrence_data = adata_subsample.uns['subclust_co_occurrence']
out = occurrence_data["occ"]
interval = occurrence_data["interval"][1:]
tims_out = pd.DataFrame(out[19,:,:]).T
tims_out.columns = clust_order
tims_out.insert(0,"interval",interval)
tims_out.to_csv("cooccurrence_raw.csv")

# %%
occurrence_data = adata_subsample_e3.uns['subclust_co_occurrence']
out = occurrence_data["occ"]
interval = occurrence_data["interval"][1:]
tims_out = pd.DataFrame(out[19,:,:]).T
tims_out.columns = clust_order
tims_out.insert(0,"interval",interval)
tims_out.to_csv("cooccurrence_raw_e3.csv")

# %%
occurrence_data = adata_subsample_e4.uns['subclust_co_occurrence']
out = occurrence_data["occ"]
interval = occurrence_data["interval"][1:]
tims_out = pd.DataFrame(out[19,:,:]).T
tims_out.columns = clust_order
tims_out.insert(0,"interval",interval)
tims_out.to_csv("cooccurrence_raw_e4.csv")

# %%
temp_superclust_dict = dict(zip(clust_order,
                       np.concatenate([np.repeat("Neurons",8),np.repeat("Oligodendrocytes",3),np.repeat("Astrocytes",4),
                                       np.repeat("Microglia",4),["TIMs"],np.repeat("Other",2)])))
temp_superclust_order = ["Neurons","Oligodendrocytes","Astrocytes","Microglia","Other","TIMs"]
adata_subsample.obs['temp_superclust'] = adata_subsample.obs['subclust'].map(temp_superclust_dict).astype('category').cat.reorder_categories(temp_superclust_order)
adata_subsample_e3 = adata_subsample[adata_subsample.obs['genotype'] == "E3"]
adata_subsample_e4 = adata_subsample[adata_subsample.obs['genotype'] == "E4"]

sq.gr.co_occurrence(
    adata_subsample_e3,
    cluster_key="temp_superclust",
    n_jobs = 32
)
sq.gr.co_occurrence(
    adata_subsample_e4,
    cluster_key="temp_superclust",
    n_jobs = 32
)

# %%
occurrence_data = adata_subsample_e3.uns['temp_superclust_co_occurrence']
out = occurrence_data["occ"]
interval = occurrence_data["interval"][1:]
tims_out = pd.DataFrame(out[5,:,:]).T
tims_out.columns = temp_superclust_order
tims_out.insert(0,"interval",interval)
tims_out.to_csv("cooccurrence_raw_e3_super.csv")

# %%
occurrence_data = adata_subsample_e4.uns['temp_superclust_co_occurrence']
out = occurrence_data["occ"]
interval = occurrence_data["interval"][1:]
tims_out = pd.DataFrame(out[5,:,:]).T
tims_out.columns = temp_superclust_order
tims_out.insert(0,"interval",interval)
tims_out.to_csv("cooccurrence_raw_e4_super.csv")

# %%
sq.gr.nhood_enrichment(adata, cluster_key="subclust")

# %%
sq.pl.nhood_enrichment(
    adata,
    cluster_key="subclust",
    figsize=(5, 5),
    title="Neighborhood Enrichment",
    dpi=600,
    save = "neighborhood_enrichment_matrix.png",
)

# %%
immune_clusts = ['Homeostatic Microglia', 'APP-hi Homeostatic Microglia','FLT1-hi Inflammatory Microglia', 'APOE-hi Inflammatory Microglia','TIMs']
sc.pl.violin(adata,keys=["RNASET2"],groupby="subclust",order=map(str,immune_clusts),rotation=90, save = "rnaset2.png")

# %%
sc.pl.violin(adata,keys=["PTPRC"],groupby="subclust",order=map(str,immune_clusts),rotation=90, save = "ptprc.png")

# %%
sc.pl.violin(adata,keys=["GPR183"],groupby="subclust",order=map(str,immune_clusts),rotation=90, save = "gpr183.png")

# %%
sc.pl.umap(adata,color=["subclust"], title = "UMAP of All Cells", save = "umap.png")
