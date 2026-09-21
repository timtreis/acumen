# mined from: https://github.com/kaitlinmoore/visium-spatial-transcriptomics/blob/3ef4b7b87741585854ef23a9e677a558db2a5ff2/notebooks/nhood.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.pl.nhood_enrichment

# %%
import warnings; warnings.filterwarnings("ignore")
import pathlib, sys
import numpy as np

# Offline smoke data: the project's committed synthetic fixture (no download).
_root = pathlib.Path.cwd()
_root = _root if (_root / "pyproject.toml").exists() else _root.parent
sys.path.insert(0, str(_root / "tests" / "fixtures"))
from synthetic import make_synthetic_visium

import squidpy as sq
from visium_spatial.build_graph import build_spatial_graph
from visium_spatial.preprocess import normalize
from visium_spatial.cluster import leiden_clusters

adata = make_synthetic_visium(seed=0)
build_spatial_graph(adata)      # the SPATIAL graph — what enrichment is measured on
normalize(adata)
leiden_clusters(adata)          # the EXPRESSION graph + Leiden (a separate object)

# n_jobs=1 avoids a Windows multiprocessing-spawn issue; trivial cost at this size.
sq.gr.nhood_enrichment(adata, cluster_key="leiden", seed=0, n_jobs=1)
sq.pl.nhood_enrichment(adata, cluster_key="leiden")

# %%
import numpy as np, scipy.sparse as sp
import scanpy as sc, squidpy as sq
from pathlib import Path
from scanpy import settings

from visium_spatial.load_visium import load_visium_dataset
from visium_spatial.qc import compute_qc_metrics, filter_spots, filter_genes
from visium_spatial.preprocess import normalize, select_hvg
from visium_spatial.build_graph import build_spatial_graph
from visium_spatial.cluster import leiden_clusters

def gene_vector(ad, g):
    v = ad[:, g].X
    return (v.toarray() if sp.issparse(v) else np.asarray(v)).ravel().astype(float)

if not (Path(settings.datasetdir) / "visium" / "V1_Human_Lymph_Node").exists():
    print("Cache the section first:  load_visium_dataset('V1_Human_Lymph_Node')  (~40 MB).")
else:
    ad = load_visium_dataset("V1_Human_Lymph_Node")
    compute_qc_metrics(ad); ad = filter_spots(ad); filter_genes(ad); normalize(ad)
    select_hvg(ad, n_top_genes=2000)
    sc.pp.pca(ad, n_comps=50)               # expression-space representation for Leiden
    build_spatial_graph(ad)                 # SPATIAL graph — what enrichment is measured on
    leiden_clusters(ad, resolution=1.0)     # EXPRESSION neighbors + Leiden (uses X_pca)
    print("leiden clusters:", ad.obs["leiden"].nunique())

    # Annotate clusters by compartment markers so the matrix is interpretable.
    for g in ["MS4A1", "CD3D"]:
        ad.obs[g] = gene_vector(ad, g)
    ann = ad.obs.groupby("leiden", observed=True)[["MS4A1", "CD3D"]].mean()
    B, T = ann["MS4A1"].idxmax(), ann["CD3D"].idxmax()
    print(f"B-like cluster (max MS4A1) = {B}; T-like cluster (max CD3D) = {T}")

    # n_jobs=1 avoids a Windows multiprocessing-spawn issue.
    sq.gr.nhood_enrichment(ad, cluster_key="leiden", seed=0, n_jobs=1)
    Z = ad.uns["leiden_nhood_enrichment"]["zscore"]
    print(f"self-enrichment z: B-B={Z[int(B),int(B)]:.1f}  T-T={Z[int(T),int(T)]:.1f}"
          f"  |  cross B-T={Z[int(B),int(T)]:.1f}  (negative = spatially segregated)")
    sq.pl.nhood_enrichment(ad, cluster_key="leiden")
