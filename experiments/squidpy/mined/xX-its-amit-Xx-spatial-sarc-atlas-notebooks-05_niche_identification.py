# mined from: https://github.com/xX-its-amit-Xx/spatial-sarc-atlas/blob/d115625850006b43525510bd3a4d3a7cd4ea106a/notebooks/05_niche_identification.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment

# %%
sample      = "GSE227469_angiosarcoma_01"
data_dir    = "data"
results_dir = "results"
n_niches    = 6
n_neighs    = 6

# %%
from pathlib import Path
import numpy as np, pandas as pd
import scanpy as sc, squidpy as sq
from sklearn.cluster import KMeans
from scipy.sparse import find
import matplotlib.pyplot as plt

adata = sc.read_h5ad(Path(results_dir) / "visium" / sample / "adata_c2l.h5ad")

# %%
frac_cols = [c for c in adata.obs.columns if c.startswith("fraction_")]
props = adata.obs[frac_cols].copy()
props.columns = [c.replace("fraction_", "") for c in props.columns]
adata.obs["dominant_celltype"] = props.idxmax(axis=1).astype("category")
sc.pl.spatial(adata, color="dominant_celltype", size=1.4)

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs)
sq.gr.nhood_enrichment(adata, cluster_key="dominant_celltype")
fig, ax = plt.subplots(figsize=(6, 6))
sq.pl.nhood_enrichment(adata, cluster_key="dominant_celltype",
                       method="ward", ax=ax)
fig.savefig(Path(results_dir) / "figures" / f"{sample}_nhood_enrichment.pdf",
            bbox_inches="tight")

# %%
sq.gr.co_occurrence(adata, cluster_key="dominant_celltype")
fig, ax = plt.subplots(figsize=(8, 4))
sq.pl.co_occurrence(adata, cluster_key="dominant_celltype",
                    clusters=["malignant"], ax=ax)
fig.savefig(Path(results_dir) / "figures" / f"{sample}_cooccurrence.pdf",
            bbox_inches="tight")

# %%
adj = adata.obsp["spatial_connectivities"]
row_sum = np.asarray(adj.sum(axis=1)).ravel()
row_sum[row_sum == 0] = 1.0
neigh_mean = (adj @ props.values) / row_sum[:, None]
neigh_df = pd.DataFrame(neigh_mean, columns=props.columns, index=adata.obs_names)

# %%
km = KMeans(n_clusters=n_niches, random_state=0, n_init=10)
adata.obs["niche"] = pd.Categorical(
    [f"N{i}" for i in km.fit_predict(neigh_df.values)]
)
sc.pl.spatial(adata, color="niche", size=1.4)

# %%
# Niche composition profile
niche_profile = (neigh_df.assign(niche=adata.obs["niche"].values)
                          .groupby("niche").mean())
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(niche_profile, cmap="viridis", ax=ax, cbar_kws={"label": "mean fraction"})
ax.set_title("Niche composition")
fig.savefig(Path(results_dir) / "figures" / f"{sample}_niche_profile.pdf",
            bbox_inches="tight")

# %%
out_path = Path(results_dir) / "visium" / sample / "niche_metrics.h5ad"
adata.write(out_path)
print("Wrote", out_path)
