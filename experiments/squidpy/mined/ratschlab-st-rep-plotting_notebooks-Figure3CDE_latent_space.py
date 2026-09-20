# mined from: https://github.com/ratschlab/st-rep/blob/3246895f76b5d12c65b6dc3828bde98e73929148/plotting_notebooks/Figure3CDE_latent_space.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# %%
import logging
# Configure the logging module
logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logging.getLogger("pyvips").setLevel(logging.CRITICAL)

# %%
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotnine as p9
import squidpy as sq
import pandas as pd
import scanpy as sc
import numpy as np
import pyvips
import umap
import json
import os

# %%
from aestetik.utils.utils_morphology import extract_morphology_embeddings
from aestetik.utils.utils_transcriptomics import preprocess_adata
from aestetik.utils.utils_vizualization import get_spot
from aestetik import AESTETIK
AESTETIK.version()

# %%
dataset = "10x_TuPro_v2"
sample = "MACEGEJ-2-2"
training_split = "MAJOFIJ-2-1_MAJOFIJ-2-2_test_MANOFYB-1-1_MANOFYB-1-2_MELIPIT-1-1_MELIPIT-1-2_MACEGEJ-1-1_MACEGEJ-1-2_MAKYGIW-1-1_MAKYGIW-1-2_MAHEFOG-1-1_MAHEFOG-1-2_MIDEKOG-1-1_MIDEKOG-1-2_MACEGEJ-2-1_MACEGEJ-2-2_MAJOFIJ-1-1_MAJOFIJ-1-2"
n_components = 15

# %%
out_folder = "out_benchmark"
img_path = f"../{dataset}/{out_folder}/data/image/{sample}.tif"

json_path = f"../{dataset}/{out_folder}/data/meta/{sample}.json"
adata_in = f"../{dataset}/{out_folder}/data/h5ad/{sample}.h5ad"
latent_path = f"../{dataset}/{out_folder}/{training_split}/AESTETIK_evaluate/latent/model-{sample}-best_param.csv"
cluster_path = f"../{dataset}/{out_folder}/{training_split}/AESTETIK_evaluate/clusters/model-{sample}-best_param.csv"
img_features_path = f"../{dataset}/{out_folder}/data/image_features/{sample}_inception.npy"

# %%
spot_diameter_fullres = json.load(open(json_path))["spot_diameter_fullres"]
dot_size = float(json.load(open(json_path))["dot_size"])
spot_diameter_fullres, dot_size

# %%
adata = sc.read(adata_in)
adata = preprocess_adata(adata)

# %%
adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"][:,0:n_components]
img_features = np.load(img_features_path)
adata.obsm["X_pca_morphology"] = img_features[:,0:n_components]

# %%
model = AESTETIK(adata, 
                 nCluster=adata.obs.ground_truth.unique().size,
                 img_path=img_path,
                 morphology_weight=0,
                 spot_diameter_fullres=spot_diameter_fullres)

# %%
latent = pd.read_csv(latent_path, index_col=0)
cluster = pd.read_csv(cluster_path)
model.adata.obsm["AESTETIK"] = latent.values
model.adata.obs["AESTETIK_cluster"] = cluster.best_param.values + 1
model.adata.obs["AESTETIK_cluster"] = model.adata.obs["AESTETIK_cluster"].astype(str)

# %%
grount_truth_ordered = {"Normal lymphoid tissue": "Normal\nlymphoid",
                "Blood and necrosis": "Blood/\nnecrosis",
                "Stroma": "Stroma",
                "Tumor": "Tumor",
                }
adata.obs["ground_truth"] = adata.obs["ground_truth"].apply(lambda x: grount_truth_ordered[x])
grount_truth_ordered = list(grount_truth_ordered.values())
adata.obs["ground_truth"] = pd.Categorical(adata.obs["ground_truth"], grount_truth_ordered)
sq.pl._color_utils._maybe_set_colors(adata, adata, "ground_truth")
model_str = "AESTETIK_cluster"
result_df = adata.obs[["ground_truth",model_str]].groupby("ground_truth")[model_str].apply(lambda x: x.mode().iloc[0]).reset_index()
result_df.set_index("ground_truth", inplace=True)
result_df = result_df.loc[grount_truth_ordered]
result_df = result_df[~result_df[model_str].duplicated()]

model_ordered = list(result_df[model_str].values)

for label in adata.obs[model_str].unique():
    if label not in model_ordered:
        model_ordered.append(label)

model_ordered_dict = dict(zip(model_ordered, range(1, len(model_ordered) + 1)))

adata.obs[model_str] = adata.obs[model_str].apply(lambda x: model_ordered_dict[x])
adata.obs[model_str] = pd.Categorical(adata.obs[model_str], model_ordered_dict.values())
adata.uns[f"{model_str}_colors"] = adata.uns[f"ground_truth_colors"][:len(model_ordered)]

# %%
model.vizualize(plot_clusters=True, plot_centroid=True)

# %%
def get_umap2(emb):
    reducer = umap.UMAP()
    umap_emb = reducer.fit_transform(emb)
    return umap_emb

# %%
transcriptomics_umap = get_umap2(adata.obsm["X_pca_transcriptomics"])
morphology_umap = get_umap2(adata.obsm["X_pca_morphology"])
aestetik_umap = get_umap2(adata.obsm["AESTETIK"])

# %%
df = pd.DataFrame(np.concatenate((transcriptomics_umap, morphology_umap, aestetik_umap), axis=1), 
             columns=["Transcriptomics_1", "Transcriptomics_2",
                      "Morphology_1", "Morphology_2",
                      "AESTETIK_1", "AESTETIK_2",
                     ]
            )
df["ground_truth"] = adata.obs.AESTETIK_cluster.values
df["barcode"] = df.index
df = df.melt(["ground_truth", "barcode"])
df["space"] = df["variable"].apply(lambda x: x.split("_")[0])
df["dim"] = df["variable"].apply(lambda x: f'UMAP_{x.split("_")[1]}')
df.drop({"variable"}, axis=1, inplace=True)
df = df.pivot_table(index=['ground_truth', 'barcode', 'space'], columns='dim', values='value').reset_index()
df

# %%
cols = sq.pl._color_utils._get_palette(adata, "AESTETIK_cluster", adata.obs.AESTETIK_cluster.unique())
df.space = pd.Categorical(df.space, ["Transcriptomics", "Morphology", "AESTETIK"])
p = (p9.ggplot(df, p9.aes("UMAP_1", "UMAP_2", color="ground_truth")) 
    + p9.geom_point()
    + p9.facet_wrap("~space", nrow=3, scales="free")
    + p9.theme_bw()
    + p9.theme(subplots_adjust={'wspace': 0.0}, figure_size=(8, 12),
            axis_text_x = p9.element_blank(),
            axis_text_y = p9.element_blank(),
            text=p9.element_text(size=15),
            strip_text=p9.element_text(size=17),
            legend_title=p9.element_text(size=17),
            legend_text=p9.element_text(size=16))
    + p9.scale_color_manual(values=adata.uns["ground_truth_colors"], guide=False)
)
p.save(f"figures/{sample}_{dataset}_st_m_aestetik_latent.png", dpi=300)
p

# %%
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def get_spot(image, x, y, spot_diameter_fullres):
    x = x - int(spot_diameter_fullres // 2)
    y = y - int(spot_diameter_fullres // 2)
    spot = image.crop(x, y, spot_diameter_fullres // 1.3, spot_diameter_fullres)
    spot_array = np.ndarray(buffer=spot.write_to_memory(),
                            dtype=format_to_dtype[spot.format],
                            shape=[spot.height, spot.width, spot.bands])
    return spot_array

def create_pc_qcut(pc1, labels, n=10):
    pc1 = np.array(pc1)
    labels = np.array(labels)
    
    pc_q = np.empty(len(labels)).astype(str)
    
    labels_unique = np.unique(labels)
    
    for label in labels_unique:
        values = np.array([f"{i}_{label}" for i in pd.qcut(pc1[label == labels], n, labels=False)])
        pc_q[label == labels] = values
    
    return pc_q
    

def select_idx(pc1, labels, n=4):
    import itertools
    
    pc_q = create_pc_qcut(pc1, labels)
    
    labels = np.array(labels)
    
    pc_q_unique = np.unique(pc_q)
    pc_q_unique = [qc for qc in pc_q_unique if int(qc.split("_")[0]) in [1,5,8]]
    print(pc_q_unique)
    labels_unique = np.unique(labels)
    
    indxs = []    
    for pc_q_group, label_group in itertools.product(pc_q_unique, labels_unique):
        indx_pass = np.logical_and(pc_q == pc_q_group, labels == label_group)
        if sum(indx_pass) > 0:
            size_n = min(n, sum(indx_pass) - 1)
            indx = np.random.choice(np.where(indx_pass)[0], size_n, replace=False)
            indxs.extend(indx)
    return np.array(indxs)

def getImage(spot, zoom=1):
    return OffsetImage(spot, zoom=zoom)

# %%
dict(zip(model_ordered_dict.values(), adata.uns["AESTETIK_cluster_colors"]))

# %%
labels = model.adata.obs["AESTETIK_cluster"]
# Randomly select 10 points for each label using NumPy and get the indices
selected_indices = [np.random.choice(np.where(labels == label)[0], 8, replace=False) for label in np.unique(labels)]
# Concatenate the indices from all labels into a single list
selected_indices = np.concatenate(selected_indices)


image = pyvips.Image.new_from_file(img_path)
reducer = umap.UMAP()
emb = reducer.fit_transform(model.adata.obsm["AESTETIK"])
# Randomly select 10 points for each label using NumPy and get the indices
indeces_to_plot = select_idx(emb[:,0], labels)
# Concatenate the indices from all labels into a single list
tab = adata.obs.iloc[indeces_to_plot]
x, y = emb[indeces_to_plot][:,0], emb[indeces_to_plot][:,1]
colormap = dict(zip(model_ordered_dict.values(), adata.uns["AESTETIK_cluster_colors"]))

n_labels = np.unique(labels).size
fig, ax = plt.subplots()
ax.scatter(emb[:,0], emb[:,1], c=[colormap[labels[i]] for i in range(len(labels))], alpha=1, s=5)
labels = tab["AESTETIK_cluster"]
for i in range(len(tab)):
    row = tab.iloc[i]
    img = get_spot(image, row.y_pixel, row.x_pixel, spot_diameter_fullres)
    bbox_props = dict(boxstyle="square,pad=0.02", fc="none", ec=colormap[labels[i]], lw=3)
    ab = AnnotationBbox(getImage(img, zoom=0.2), (x[i], y[i]), frameon=True, bboxprops=bbox_props)
    ax.add_artist(ab)

    legend_patches = []
for l in np.unique(labels):
    legend_patches.append(mpl.patches.Patch(color=colormap[l], label=f"{l}"))
# Display the legend
ax.legend(handles=legend_patches)
ax.set_xlabel(f"UMAP_1")
ax.set_ylabel(f"UMAP_2")
plt.savefig(f"figures/{sample}_{dataset}_spots_in_latent_space.png", dpi=300)
plt.show()

# %%
import plotnine as p9
from sklearn.decomposition import PCA
import pandas as pd
from plotnine_prism import *
import squidpy as sq
cols = sq.pl._color_utils._get_palette(adata, "AESTETIK_cluster", adata.obs.AESTETIK_cluster.unique())
cols

# %%
adata.obs["Tumor_distance"] = pd.qcut(adata.obs["dist_from_4"], 
                                      [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
                                      labels=["[0-0.1]", "(0.1-0.25]", "(0.25-0.5]", 
                                              "(0.5-0.75]", "(0.75-0.9]", "(0.9-1]"])
adata.obs["Tumor_distance"].value_counts()

# %%
genes = ["Tumor_distance"]
plt.rcParams.update({'font.size': 11})
pad=20
bounds = (adata.obsm["spatial"][:,0].min() - pad, 
              adata.obsm["spatial"][:,1].min() - pad,
               adata.obsm["spatial"][:,0].max() + pad,
               adata.obsm["spatial"][:,1].max() + pad)

sq.pl.spatial_scatter(
            adata,
            crop_coord=bounds,
            title="Tumor distance",
            img_alpha=0.8,
            color=genes,
            size=4,
            use_raw=False,
            ncols=3, wspace=0, dpi=300, frameon=False, figsize=(6,5),
            palette = 'YlOrRd',
            legend_fontweight="semibold",
            save=f"{sample}_{'_'.join(genes)}.png"
)

# %%

