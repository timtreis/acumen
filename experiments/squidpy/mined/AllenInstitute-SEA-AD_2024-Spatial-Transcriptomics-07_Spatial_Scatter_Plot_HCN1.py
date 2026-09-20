# mined from: https://github.com/AllenInstitute/SEA-AD_2024/blob/126b3cf77095c4e0ac851ff579895b0e02c2d790/Spatial Transcriptomics/07_Spatial_Scatter_Plot_HCN1.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import transforms as tsf
import seaborn as sns

from scipy.stats import energy_distance, wasserstein_distance, ranksums

# %%
h5ad_file = '/allen/programs/celltypes/workgroups/hct/SEA-AD/MERSCOPE/proportion_analysis/manuscript_with_all_mtg_tsne_selected_with_layers.h5ad'
adata = sc.read_h5ad(h5ad_file)

# %%
#Remove blank genes from dataset
blanks = np.array([i.startswith("Blank") for i in adata.var_names])
adata = adata[:, ~blanks]

# %%
adata.X = adata.layers["raw"].copy()

# %%
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
# log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
vtypes = [
"Lamp5_3",
"Lamp5_5",
"Sncg_2",
"Sncg_1",
"Sncg_8",
"Vip_2",
"Vip_11",
"Vip_13",
"Vip_1",
"Vip_12",
"Sst_3",
"Sst_19",
"Sst_11",
"Sst_20",
"Sst_22",
"Sst_23",
"Sst_25",
"Sst_2",
"Pvalb_6",
"Pvalb_5",
"Pvalb_8",
"Pvalb_3",
"Pvalb_2",
"Pvalb_15",
"Pvalb_14",
"L2/3 IT_1",
"L2/3 IT_6",
"L2/3 IT_7",
"L2/3 IT_5",
"L2/3 IT_13",
"L2/3 IT_10",
"L2/3 IT_8",
"L2/3 IT_12",
"L2/3 IT_3",
"Astro_2",
"OPC_2",
"Oligo_2",
"Micro-PVM_3-SEAAD"
]

# %%
adata.obs.columns

# %%
adata.obs["Affected"] = adata.obs.supertype_scANVI_leiden.isin(vtypes)

# %%
layer_adata = adata[adata.obs["layer_annotation"] != '']

# %%
section_key = "filename"
sections = adata.obs[section_key].unique()

# %%
#Calculate the biggest difference in distributions  across samples
cell_type_key = "subclass"
cell_type = "Sst"
gene = "HCN1"
distance_dict = {}
for section in sections:
    print(section)
    section_adata = adata[(adata.obs[section_key] == section) & (adata.obs[cell_type_key] == cell_type)]
    affected_adata = section_adata[section_adata.obs["Affected"]]
    unaffected_adata = section_adata[~section_adata.obs["Affected"]]

    affected_gene = affected_adata[:, gene].X.squeeze()
    unaffected_gene = unaffected_adata[:, gene].X.squeeze()
    
    distance_dict[section] = ranksums(affected_gene, unaffected_gene)[0]
    

# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

color_map = mpl.colormaps['YlGnBu']
color_map = color_map(np.linspace(0.15, 1, 8))

# %%
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', })

section_value = sorted(((v,k) for k,v in distance_dict.items()))[-1][1]
print(f"Section Barcode: {section_value}")
section_adata = layer_adata[(layer_adata.obs[section_key] == section_value)]
cell_adata = section_adata[(section_adata.obs[cell_type_key] == cell_type)]
fig, ax = plt.subplots(nrows=1, ncols=2, dpi = 600)

max_exp = cell_adata[cell_adata.obs["Affected"], gene].layers["log1p_norm"].max() * 0.75
sq.pl.spatial_scatter(section_adata, shape=None, size=5, title = "Affected", alpha = 0.0025, ax = ax[0])
sq.pl.spatial_scatter(cell_adata[cell_adata.obs["Affected"]],title = f"Affected {cell_type} {gene}", color = "HCN1", shape = None, size = 5, ax = ax[0], vmin = 0, vmax = max_exp, layer = "log1p_norm", cmap = LinearSegmentedColormap.from_list("mycmap", color_map))

sq.pl.spatial_scatter(section_adata, shape=None, size=5, title = "Unaffected", alpha = 0.0025,  ax = ax[1])
sq.pl.spatial_scatter(cell_adata[~cell_adata.obs["Affected"]], color = "HCN1", title = f"Unaffected {cell_type} {gene}", shape = None, size = 5, ax = ax[1], vmin = 0, vmax = max_exp, layer = "log1p_norm",  cmap = LinearSegmentedColormap.from_list("mycmap", color_map))
fig.tight_layout()
#plt.savefig(f"../figures/{section_value}_HCN1_sst_spatial_scatter.png", dpi = 600)

# %%
# section_value = max(distance_dict, key = distance_dict.get)
# section_adata = adata[(adata.obs[section_key] == section_value)]
# coords = section_adata.obsm["spatial"]
# section_adata.obsm["spatial_invert"] = coords[:, [1, 0]]

# cell_adata = section_adata[(section_adata.obs[cell_type_key] == cell_type)]
# fig, ax = plt.subplots(nrows=1, ncols=2)
# sq.pl.spatial_scatter(section_adata, shape=None, size=1, title = "Affected", alpha = 0.05, ax = ax[0], spatial_key="spatial_invert")
# sq.pl.spatial_scatter(cell_adata[cell_adata.obs["Affected"]],title = f"Affected {cell_type} {gene}", color = "HCN1", shape = None, size = 5, ax = ax[0], vmin = 0,  vmax = 2.5, spatial_key="spatial_invert", layer = "log1p_norm")

# sq.pl.spatial_scatter(section_adata, shape=None, size=1, title = "Unaffected", alpha = 0.05, ax = ax[1], spatial_key="spatial_invert")
# sq.pl.spatial_scatter(cell_adata[~cell_adata.obs["Affected"]], color = "HCN1", title = f"Unaffected {cell_type} {gene}", shape = None, size = 25, ax = ax[1], vmin = 0, vmax = 2.5, spatial_key="spatial_invert", layer = "log1p_norm")

# plt.show

# %%
gene = "HCN1"
cell_adata = adata[adata.obs[cell_type_key] == cell_type]
df = cell_adata.obs.copy()
df[gene] = np.array(cell_adata[:, gene].layers["log1p_norm"]).squeeze()

# %%
color = sns.color_palette("tab10")

# %%
#plot histogram
sns.displot(data = df, x = gene, hue = "Affected", stat = "percent", alpha = 0.5, legend = False, hue_order = [True, False])

#Plot early median
early_median = np.median(df[df["Affected"]][gene])
plt.axvline(early_median, color = color[0], label = "Affected Median", linestyle = "--")
#Plot late median
late_median = np.median(df[~df["Affected"]][gene])
plt.axvline(late_median, color = color[1], label = "Unaffected Median", linestyle = "--")

#Add labels
plt.xlabel("Normalized Expression")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f"{gene} expresion in {cell_type}")

# %%
sns.violinplot(x = "Affected", y = gene, data = df, inner = "quartile",)
plt.xticks([0, 1], ["Unaffected", "Affected"])
plt.xlabel(None)

# %%

