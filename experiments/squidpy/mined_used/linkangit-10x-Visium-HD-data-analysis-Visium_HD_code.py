# mined from: https://github.com/linkangit/10x-Visium-HD-data-analysis/blob/22cc85a0c18a92748e312c42b687d7da708a829e/Visium_HD_code.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
# ============================================
# 10x Visium HD (8 µm) — End-to-End Analysis
# One self-contained block (no custom functions).
# Requires: scanpy squidpy anndata h5py pandas numpy matplotlib pillow pyarrow leidenalg seaborn (optional) scikit-learn (optional)
# First time: 
# !pip install scanpy squidpy anndata h5py pandas numpy matplotlib pillow pyarrow leidenalg seaborn scikit-learn
# ============================================

import os, re, json
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Paths (EDIT THIS) ----------
RUN_DIR = "Visium_HD_data"         # <-- your dataset root
BIN = "008"                        # '008' for 8 µm
BIN_DIR = os.path.join(RUN_DIR, f"binned_outputs/square_{BIN}um")
SPAT_DIR = os.path.join(BIN_DIR, "spatial")
MATRIX_H5 = os.path.join(BIN_DIR, "filtered_feature_bc_matrix.h5")
lib_id = f"square_{BIN}um"

# ---------- Load matrix ----------
adata = sc.read_10x_h5(MATRIX_H5)
adata.var_names_make_unique()
print(adata)

# ---------- Wire spatial coordinates + images ----------
pos = pd.read_parquet(os.path.join(SPAT_DIR, "tissue_positions.parquet"))
if "barcode" not in pos.columns:
    pos = pos.rename(columns={pos.columns[0]: "barcode"})
pos = pos.set_index("barcode")

# Align to adata.obs_names (handle '-1' suffix)
_base = lambda s: s.to_series().str.replace(r"-\d+$", "", regex=True)
if not pos.index.isin(adata.obs_names).all():
    pos.index = _base(pos.index)
    adata.obs["__base"] = _base(adata.obs_names)
    pos = pos.reindex(adata.obs["__base"])
else:
    pos = pos.reindex(adata.obs_names)

# Set spatial coordinates in FULLRES pixel space [x=col, y=row]
pxr, pxc = "pxl_row_in_fullres", "pxl_col_in_fullres"
adata.obsm["spatial"] = np.c_[pos[pxc].to_numpy(), pos[pxr].to_numpy()]

# Load scalefactors & images
with open(os.path.join(SPAT_DIR, "scalefactors_json.json"), "r") as fh:
    scales = json.load(fh)
img_hires = np.array(Image.open(os.path.join(SPAT_DIR, "tissue_hires_image.png")))
img_low   = np.array(Image.open(os.path.join(SPAT_DIR, "tissue_lowres_image.png")))

adata.uns["spatial"] = {
    lib_id: {"images": {"hires": img_hires, "lowres": img_low}, "scalefactors": scales, "metadata": {}},
    "library_id": [lib_id],
}

# Optional: parse grid coords from barcodes (helpful for grid neighbors)
coords = adata.obs_names.to_series().str.extract(r"s_(\d{3})um_(\d+)_(\d+)(?:-\d+)?$")
coords.columns = ["um","array_row","array_col"]
if coords.notna().all().all():
    adata.obs[["array_row","array_col"]] = coords[["array_row","array_col"]].astype(float).values

# ---------- QC tags & metrics ----------
adata.var["mt"]   = adata.var_names.str.startswith(("mt-","mt."))
adata.var["ribo"] = adata.var_names.str.match(r"^(Rps|Rpl)\d", na=False)
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo"], inplace=True)
print("spots:", adata.n_obs, "| genes:", adata.n_vars)
qc_keys = [k for k in ["n_genes_by_counts","total_counts","pct_counts_mt","pct_counts_ribo"] if k in adata.obs.columns]
if qc_keys and adata.n_obs > 0:
    sc.pl.violin(adata, qc_keys, jitter=0.3, multi_panel=True)

# ---------- Adaptive filtering (dataset-driven thresholds) ----------
low, high = adata.obs['total_counts'].quantile([0.02, 0.99])
min_counts = max(100, int(low))
max_counts = max(int(high), 30000)
print(f"Using min_counts={min_counts}, max_counts={max_counts}")
sc.pp.filter_cells(adata, min_counts=min_counts)
sc.pp.filter_cells(adata, max_counts=max_counts)
if 'pct_counts_mt' in adata.obs:
    adata = adata[adata.obs['pct_counts_mt'] < 25].copy()
if 'pct_counts_ribo' in adata.obs:
    adata = adata[adata.obs['pct_counts_ribo'] < 30].copy()
sc.pp.filter_genes(adata, min_counts=10)

# Recompute QC after filtering
adata.var["mt"]   = adata.var_names.str.startswith(("mt-","mt."))
adata.var["ribo"] = adata.var_names.str.match(r"^(Rps|Rpl)\d", na=False)
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo"], inplace=True)
print("Remaining spots:", adata.n_obs, "| genes:", adata.n_vars)
qc_keys = [k for k in ["n_genes_by_counts","total_counts","pct_counts_mt","pct_counts_ribo"] if k in adata.obs.columns]
if qc_keys and adata.n_obs > 0:
    sc.pl.violin(adata, qc_keys, jitter=0.3, multi_panel=True)
else:
    print("⚠️ Skipping violin plot: no spots or missing QC keys.")

# ---------- Normalize, log, HVGs, PCA, neighbors, UMAP, Leiden ----------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)
import leidenalg  # ensure installed
sc.tl.leiden(adata, resolution=0.8, key_added="leiden_bin")
adata.obs["cluster"] = adata.obs["leiden_bin"].astype("category")

sc.pl.umap(adata, color=["cluster","total_counts","pct_counts_mt"], wspace=0.4)
sc.pl.spatial(adata, color="cluster", library_id=lib_id, spot_size=1.2)

# ---------- Spatial neighbors + enrichment + co-occurrence ----------
try:
    if {"array_row","array_col"}.issubset(adata.obs.columns):
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=4, n_rings=1)
    else:
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=8)
except Exception:
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=8)

sq.gr.nhood_enrichment(adata, cluster_key="cluster")
sq.pl.nhood_enrichment(adata, cluster_key="cluster")

sq.gr.co_occurrence(adata, cluster_key="cluster")
seed_cluster = adata.obs["cluster"].cat.categories[0] if "cluster" in adata.obs and adata.n_obs > 0 else None
if seed_cluster is not None:
    sq.pl.co_occurrence(adata, cluster_key="cluster", clusters=seed_cluster, figsize=(8,4))

# ---------- Spatial gene panels (example) ----------
# Guard missing genes:
for g in ["Olfm1","Plp1"]:
    if g not in adata.var_names:
        print(f"⚠️ Gene '{g}' not found in adata.var_names")
sq.pl.spatial_scatter(adata, color=[g for g in ["Olfm1","Plp1","cluster"] if (g=="cluster" or g in adata.var_names)],
                      library_id=lib_id, size=1.2)

# ---------- Marker discovery ----------
sc.tl.rank_genes_groups(adata, "cluster", method="t-test")
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby="cluster", show_gene_labels=True)

rg_df = sc.get.rank_genes_groups_df(adata, group=None).sort_values(
    ["group","pvals_adj","scores"], ascending=[True, True, False]
)
topN = 6
top_df = rg_df.groupby("group", as_index=False, sort=False).head(topN)
ordered_genes = top_df.groupby("group")["names"].apply(list).explode().drop_duplicates().tolist()
sc.pl.dotplot(adata, var_names=ordered_genes, groupby="cluster", standard_scale="var", dendrogram=False)
sc.pl.heatmap(adata, var_names=ordered_genes, groupby="cluster",
              swap_axes=True, vmin=-2, vmax=2, cmap="viridis", show_gene_labels=True)

# ---------- Marker module scoring & annotation (Allen/Linnarson-style panels) ----------
marker_sets = {
    "Excitatory neurons": ["Slc17a7","Slc30a10","Tbr1","Cux1","Cux2","Rorb","Foxp2","Reln"],
    "Inhibitory neurons": ["Gad1","Gad2","Slc6a1","Pvalb","Sst","Vip","Reln"],
    "Oligodendrocytes":  ["Mbp","Plp1","Mog","Cnp","Cldn11"],
    "OPCs":              ["Pdgfra","Cspg4","Tnr"],
    "Astrocytes":        ["Aqp4","Gfap","Aldh1l1","Slc1a3"],
    "Microglia":         ["C1qa","C1qb","Cx3cr1","Tyrobp","P2ry12"],
    "Endothelial":       ["Kdr","Pecam1","Klf2","Cldn5","Rgs5"],
    "Ependymal/Choroid": ["Foxj1","Ttr","Krt18"],
    "Hippocampus-enriched": ["Prox1","Zbtb20","Itpka","Calb2","Pcp4"],
}
for name, genes in marker_sets.items():
    present = [g for g in genes if g in adata.var_names]
    if len(present) == 0:
        adata.obs[f"score_{name}"] = 0.0
    else:
        sc.tl.score_genes(adata, present, score_name=f"score_{name}", use_raw=False)

cluster_key = "cluster"
score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
avg_scores = adata.obs.groupby(cluster_key)[score_cols].mean()
best_class = avg_scores.idxmax(axis=1).str.replace("^score_", "", regex=True)
mapping = {cl: lbl for cl, lbl in zip(best_class.index.astype(str), best_class.values)}
adata.obs["cluster_annot"] = adata.obs[cluster_key].astype(str).map(mapping).astype("category")

print("Cluster → annotation mapping:")
print(mapping)
sc.pl.umap(adata, color=[cluster_key, "cluster_annot"], wspace=0.4)
sc.pl.spatial(adata, color="cluster_annot", library_id=lib_id, spot_size=1.2)

# Side-by-side panels (genes + clusters)
sq.pl.spatial_scatter(
    adata,
    color=[c for c in ["Olfm1","Plp1","cluster"] if (c=="cluster" or c in adata.var_names)],
    library_id=lib_id, size=1.2, ncols=3, title=["Olfm1","Plp1","Leiden clusters"]
)
sq.pl.spatial_scatter(
    adata,
    color=[c for c in ["Olfm1","Plp1","cluster_annot"] if (c=="cluster_annot" or c in adata.var_names)],
    library_id=lib_id, size=1.2, ncols=3, title=["Olfm1","Plp1","Annotated clusters"]
)
sq.pl.spatial_scatter(
    adata,
    color=["cluster","cluster_annot"], library_id=lib_id, size=1.2, ncols=2,
    title=["Leiden clusters","Annotated clusters"]
)

# ---------- Validation: contingency + ARI/NMI + marker score heatmaps ----------
try:
    import seaborn as sns
    # Contingency
    ct = pd.crosstab(adata.obs["cluster"].astype(str), adata.obs["cluster_annot"].astype(str))
    plt.figure(figsize=(max(6, 0.5*ct.shape[1]), max(4, 0.4*ct.shape[0])))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Leiden × Annotated — Contingency (counts)"); plt.tight_layout(); plt.show()

    ct_norm = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    plt.figure(figsize=(max(6, 0.5*ct_norm.shape[1]), max(4, 0.4*ct_norm.shape[0])))
    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="magma")
    plt.title("Leiden × Annotated — Row-normalized fractions"); plt.tight_layout(); plt.show()

    # Agreement metrics
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(adata.obs["cluster"].astype(str), adata.obs["cluster_annot"].astype(str))
        nmi = normalized_mutual_info_score(adata.obs["cluster"].astype(str), adata.obs["cluster_annot"].astype(str))
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
    except Exception:
        print("Install scikit-learn for ARI/NMI: pip install scikit-learn")

    # Marker score heatmaps
    if len(score_cols) > 0:
        avg_by_leiden = adata.obs.groupby("cluster")[score_cols].mean()
        avg_by_annot  = adata.obs.groupby("cluster_annot")[score_cols].mean()
        plt.figure(figsize=(1.2*len(score_cols), 0.6*len(avg_by_leiden)))
        sns.heatmap(avg_by_leiden, cmap="viridis"); plt.title("Marker scores by Leiden"); plt.tight_layout(); plt.show()
        plt.figure(figsize=(1.2*len(score_cols), 0.6*len(avg_by_annot)))
        sns.heatmap(avg_by_annot, cmap="viridis"); plt.title("Marker scores by Annotation"); plt.tight_layout(); plt.show()
except Exception as e:
    print("Validation heatmaps skipped (seaborn not available or plotting error):", e)

# ---------- Methods blurb ----------
METHODS_TEXT = """
Preprocessing followed the Scanpy spatial tutorial workflow:
https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
Briefly, we computed QC metrics (including mitochondrial and ribosomal fractions),
performed adaptive filtering, library-size normalization, log1p transform, HVG selection,
PCA, neighborhood graph construction, UMAP, and Leiden clustering.

Cluster annotation was guided by external resources:
- Allen Brain Atlas: https://mouse.brain-map.org/experiment/thumbnails/100048576?image_type=atlas
- Mouse Brain Gene Expression Atlas (Linnarson lab): http://mousebrain.org/
- Spatial transcriptomics preprint: https://www.biorxiv.org/content/10.1101/2020.07.24.219758v1

We generated module scores for literature-derived marker sets (astrocytes, oligodendrocytes, OPCs,
microglia, endothelial, inhibitory/excitatory neurons, hippocampus-enriched, etc.), averaged per
cluster, and assigned labels by the top-scoring class. Labels can be refined by inspecting
individual marker genes and spatial context.
"""
print(METHODS_TEXT)

# ---------- Save processed object ----------
OUT = os.path.join(RUN_DIR, f"visium_hd_{BIN}um_processed.h5ad")
adata.write(OUT, compression="gzip")
print("Saved:", OUT)
