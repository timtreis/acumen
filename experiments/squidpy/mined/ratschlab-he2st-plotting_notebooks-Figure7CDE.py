# mined from: https://github.com/ratschlab/he2st/blob/17087753ce410a9fbd928255679c8c42c088bc58/plotting_notebooks/Figure7CDE.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import yaml
import anndata as ad
import matplotlib.pyplot as plt
import squidpy as sq
from sklearn.linear_model import LinearRegression, LogisticRegression

# %%
from src.knn import EnhancedKNNClassifier
from sklearn.decomposition import PCA

# %%
with open("../Lung_Xenium/config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)
samples = config_dataset["SAMPLE_LQ"]
genes_of_interest = config_dataset["genes_of_interest"]
samples

# %%
ref_lung_atlas = sc.read_h5ad("/cluster/customapps/biomed/grlab/users/knonchev/ref_atlas/b351804c-293e-4aeb-9c4c-043db67f4540.h5ad")
ref_lung_atlas.var.index = ref_lung_atlas.var.feature_name.values
ref_lung_atlas

# %%
for i in ref_lung_atlas.obs.ann_level_2.unique():
    print(i)

# %%
adata_genes = pd.read_csv("../Lung_Xenium/out_benchmark/info_highly_variable_genes.csv").query("isPredicted == True").gene_name.values
all_genes = np.array(list((set(genes_of_interest) | set(adata_genes)) & set(ref_lung_atlas.var.feature_name)))
shared_genes = np.array(list(set(ref_lung_atlas.var.feature_name.values) & set(adata_genes)))
len(all_genes), len(shared_genes)

# %%
ref_lung_atlas = ref_lung_atlas[:, all_genes].copy()
sc.pp.filter_cells(ref_lung_atlas, min_genes=10)
ref_lung_atlas.X = ref_lung_atlas.X.toarray()

# %%
sc.pp.pca(ref_lung_atlas, n_comps=15)
sc.pp.neighbors(ref_lung_atlas)
sc.tl.umap(ref_lung_atlas)

# %%
ref_lung_atlas_subset = ref_lung_atlas[:, shared_genes].copy()
ref_lung_atlas_subset

# %%
lr = LinearRegression(n_jobs=-1)
lr.fit(ref_lung_atlas_subset.X, ref_lung_atlas.X)

# %%
model = "DeepCell"
adatas_atlas= []
for sample in tqdm(samples):


    adata_path_pred = f"../Lung_Xenium/out_benchmark/prediction/{model}/data/h5ad/{sample}.h5ad"
    adata_pred = sc.read_h5ad(adata_path_pred)
    adata_pred = adata_pred[:, shared_genes].copy()
    

    counts_atlas = lr.predict(adata_pred.X)
    counts_atlas[counts_atlas < 0] = 0

    adata_atlas = ad.AnnData(counts_atlas, obs=adata_pred.obs, obsm=adata_pred.obsm, uns=adata_pred.uns)
    adata_atlas.var.index = ref_lung_atlas.var.feature_name.values
    
    sc.tl.ingest(adata_atlas, ref_lung_atlas, embedding_method=('umap', 'pca'))
    adatas_atlas.append(adata_atlas)

# %%
adata_concat = ad.concat([ref_lung_atlas, *adatas_atlas], label="Modality", 
                         keys=["Single-cell", *[f"H&E {s}" for s in samples]])
adata_concat

# %%
ref_lung_atlas.obs.ann_level_2 = ref_lung_atlas.obs.ann_level_2.apply(lambda x: x.replace(" cells", "\ncells"))

# %%
clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=15)
clf.fit(ref_lung_atlas.obsm["X_pca"], ref_lung_atlas.obs.ann_level_2)

# %%
adata_concat.obs["label"] = clf.predict(adata_concat.obsm["X_pca"])

# %%
with plt.rc_context({
    "figure.figsize": (10, 8),
    "figure.dpi": 300,
#    "font.size": 17,  # Increase font size
    "axes.titlesize": 17,
    "axes.labelsize": 17,
#    "legend.fontsize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17
}):

    fig = sc.pl.umap(
        adata_concat, 
        color=["Modality", "label"],
        hspace=7,
        size=3,
        show=False,
        frameon=False,
        return_fig=True
    )

    # Custom titles for each subplot
    custom_titles = ["Integrated single-cell atlas and H&E slides", "Single-cell annotations"]
    for ax, title in zip(fig.axes, custom_titles):
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("")  # Hide x-axis label
        ax.set_ylabel("")  # Hide y-axis label

    fig.savefig("figures/Figure7C_Lung_Xenium_ref_atlas_umap.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
lr = LinearRegression(n_jobs=-1)
lr.fit(ref_lung_atlas_subset.X, ref_lung_atlas.X)

# %%
samples

# %%
adata = adatas_atlas[1]
adata.obs["label"] = clf.predict(adata.obsm["X_pca"])

# %%
plt.rcParams.update({'font.size': 14})


bounds = (adata.obsm["spatial"][:, 0].min(),
              adata.obsm["spatial"][:, 1].min()+50,
              adata.obsm["spatial"][:, 0].max()-50,
              adata.obsm["spatial"][:, 1].max()-350)
bounds

# %%
#del adata.uns["label_colors"]

# %%
sq.pl.spatial_scatter(adata, 
                      color="label", 
                      title="Transferred single-cell annotations",
                      #img_alpha=0.5,
                      img=False,
                      crop_coord=bounds, 
                      wspace=0.1, 
                      hspace=0.1,
                      size=3,      
                      ncols=1, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure7E_lung_xenium_NCBI867_labels.png", 
                      dpi=150,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(7, 7))
plt.show()

# %%
sq.pl.spatial_scatter(adata, 
                      color="label", 
                      title="H&E image",
                      #img_alpha=0.5,
                      img=True,
                      crop_coord=bounds, 
                      wspace=0.1, 
                      hspace=0.1,
                      size=0,      
                      ncols=1, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure7D_lung_xenium_NCBI867_image.png", 
                      dpi=150,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(7, 7))
plt.show()

# %%

