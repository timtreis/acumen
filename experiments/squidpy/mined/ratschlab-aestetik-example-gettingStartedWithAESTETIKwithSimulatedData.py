# mined from: https://github.com/ratschlab/aestetik/blob/932bb7ad91f64562b29d54830a95f05a2658bda2/example/gettingStartedWithAESTETIKwithSimulatedData.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "test_data").is_dir() else Path.cwd().parent
TEST_DATA = REPO_ROOT / "test_data"

# %%
from sklearn.cluster import KMeans
from aestetik import AESTETIK
import squidpy as sq
import scanpy as sc
import numpy as np

# %%
adata = sc.read(str(TEST_DATA / "A.h5ad"))
adata

# %%
adata.obsm["X_pca"].shape

# %%
adata.obsm["image"].shape

# %%
adata.obsm["combined"] = np.concatenate((adata.obsm["X_pca"], adata.obsm["image"]), axis=1)
adata.obsm["combined"].shape

# %%
#based only on transcriptomics
sc.pp.neighbors(adata, use_rep="X_pca")
sc.tl.umap(adata)

adata.obs["transcriptomics_kmeans"] = KMeans(5).fit_predict(adata.obsm["X_pca"]).astype(str)
sc.pl.umap(adata, color=["ground_truth", 
                         "transcriptomics_kmeans"])

# %%
#based only on morphology
sc.pp.neighbors(adata, use_rep="image")
sc.tl.umap(adata)

adata.obs["morphology_kmeans"] = KMeans(5).fit_predict(adata.obsm["image"]).astype(str)
sc.pl.umap(adata, color=["ground_truth", 
                         "morphology_kmeans"])

# %%
#based only on combined
sc.pp.neighbors(adata, use_rep="combined")
sc.tl.umap(adata)

adata.obs["combined_kmeans"] = KMeans(5).fit_predict(adata.obsm["combined"]).astype(str)
sc.pl.umap(adata, color=["ground_truth", 
                         "combined_kmeans"])

# %%
sc.pl.umap(adata, color=["ground_truth", 
                         "transcriptomics_kmeans",
                         "morphology_kmeans",
                         "combined_kmeans"])

# %%
sq.pl.spatial_scatter(adata, color=["ground_truth", 
                         "transcriptomics_kmeans",
                         "morphology_kmeans",
                         "combined_kmeans"], 
                      size=0.5)

# %%
# we set the transcriptomics modality
adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"][:,0:15]
# we set the morphology modality
adata.obsm["X_pca_morphology"] = adata.obsm["image"][:,0:15]

# %%
parameters = {
    'morphology_weight': 1.5,
    'refine_cluster': True,
    'window_size': 3,
    'clustering_method': 'kmeans',
}
parameters

# %%
model = AESTETIK(n_cluster=adata.obs.ground_truth.unique().size,
                 **parameters)

# %%
# sklearn-style: fit_predict returns the cluster labels.
adata.obs['AESTETIK_cluster'] = model.fit_predict(adata)
adata.obsm['AESTETIK'] = model.embedding_

# %%
#based on AESTETIK representation
sc.pp.neighbors(adata, use_rep="AESTETIK")
sc.tl.umap(adata)

sc.pl.umap(adata, color=["ground_truth", 
                         "AESTETIK_cluster"])

# %%
sq.pl.spatial_scatter(adata, color=["ground_truth",
                         "combined_kmeans",
                         "AESTETIK_cluster"], 
                      ncols=5,
                      wspace=0,
                      dpi=150,
                      size=0.5)

# %%
sq.pl.spatial_scatter(adata, color=["ground_truth",
                                    "transcriptomics_kmeans",
                                     "morphology_kmeans",
                                     "combined_kmeans",
                                     "AESTETIK_cluster"], 
                                  ncols=5,
                                  wspace=0,
                                  dpi=150,
                                  size=0.5,
                                  save="AESTETIK_clustering.png"
                     )
