# mined from: https://github.com/JakobElkholtei/retinoblastoma/blob/9168442feebea053191e6ac88f5bb773233d738a/notebooks/01_exploration.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.spatial_scatter, squidpy.read.visium

# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json, shutil, tempfile
from pathlib import Path

sc.settings.verbosity = 1
DATA_DIR  = Path("../downloaded_data")
FIG_DIR   = Path("../figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
def load_visium_sample(stem: str) -> sc.AnnData:
    """Stage flat files into Visium folder layout and load with squidpy."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spatial_dir = tmp / "spatial"
        spatial_dir.mkdir()

        shutil.copy(DATA_DIR / f"{stem}_filtered_feature_bc_matrix.h5",
                    tmp / "filtered_feature_bc_matrix.h5")
        shutil.copy(DATA_DIR / f"{stem}_tissue_positions_list.csv",
                    spatial_dir / "tissue_positions_list.csv")
        shutil.copy(DATA_DIR / f"{stem}_tissue_hires_image.png",
                    spatial_dir / "tissue_hires_image.png")
        shutil.copy(DATA_DIR / f"{stem}_tissue_hires_image.png",
                    spatial_dir / "tissue_lowres_image.png")
        shutil.copy(DATA_DIR / f"{stem}_scalefactors_json.json",
                    spatial_dir / "scalefactors_json.json")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Variable names are not unique")
            adata = sq.read.visium(tmp, library_id=stem)
        adata.var_names_make_unique()
        return adata


h5_files = sorted(DATA_DIR.glob("*filtered_feature_bc_matrix.h5"))
stems = [f.stem.replace("_filtered_feature_bc_matrix", "") for f in h5_files]

complete = [s for s in stems if all([
    (DATA_DIR / f"{s}_tissue_positions_list.csv").exists(),
    (DATA_DIR / f"{s}_tissue_hires_image.png").exists(),
    (DATA_DIR / f"{s}_scalefactors_json.json").exists(),
])]
print("Samples with full spatial data:", complete)

adatas = {s: load_visium_sample(s) for s in complete}
print(f"\nLoaded {len(adatas)} samples.")

# %%
records = []
for sid, adata in adatas.items():
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    mt = adata.var_names.str.startswith("MT-")
    adata.obs["pct_mt"] = np.array(adata.X[:, mt].sum(1)).flatten() / adata.obs["total_counts"] * 100
    in_tissue = adata.obs["in_tissue"].sum() if "in_tissue" in adata.obs else adata.n_obs
    records.append({
        "sample":         sid,
        "spots_total":    adata.n_obs,
        "spots_tissue":   int(in_tissue),
        "n_genes":        adata.n_vars,
        "median_UMI":     adata.obs["total_counts"].median(),
        "median_genes":   adata.obs["n_genes_by_counts"].median(),
        "median_pct_mt":  adata.obs["pct_mt"].median(),
    })

pd.DataFrame(records).set_index("sample")

# %%
for sid, adata in adatas.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sq.pl.spatial_scatter(adata, color="total_counts", library_id=sid,
                          title=f"{sid} — total UMI", ax=axes[0], show=False)
    sq.pl.spatial_scatter(adata, color="pct_mt", library_id=sid,
                          title=f"{sid} — % MT", ax=axes[1], show=False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"01_spatial_{sid}.png", dpi=150)
    plt.show()

# %%
for sid, adata in adatas.items():
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    print(f"{sid}: {adata.obsp['spatial_connectivities'].nnz} neighbour edges "
          f"across {adata.n_obs} spots")

# %%
sid = complete[0]   # run on first sample as demonstration
adata = adatas[sid]

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, subset=True)

# n_perms=None skips permutation p-values (avoids joblib/PIL worker crash)
# re-enable once `conda install -c conda-forge libtiff` fixes the PIL dylib issue
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None)

morans = adata.uns["moranI"].sort_values("I", ascending=False)
print(f"Top 10 spatially variable genes in {sid}:")
morans.head(10)

# %%
top_svg = morans.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for gene, ax in zip(top_svg, axes.flatten()):
    sq.pl.spatial_scatter(adata, color=gene, library_id=sid,
                          title=gene, ax=ax, show=False)
plt.suptitle(f"Top spatially variable genes — {sid}", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / f"01_svg_{sid}.png", dpi=150)
plt.show()

# %%
import rds2py, warnings
warnings.filterwarnings("ignore")

rds_path = DATA_DIR / "human_RB.rds"
seurat = rds2py.read_rds(str(rds_path))

print("class:", seurat.get("class_name"))
attrs = seurat.get("attributes", {})
print("Top-level slots:", list(attrs.keys()))

# %%
def get_slot(attrs, *keys):
    """Navigate nested rds2py dict by slot names."""
    cur = attrs
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def rds_list_to_values(node):
    """Extract the values list from an rds2py vector node."""
    if isinstance(node, dict):
        return node.get("values", node.get("data", []))
    return []

# Seurat meta.data lives at attributes -> meta.data
meta_node = get_slot(attrs, "meta.data")
if meta_node is None:
    print("meta.data slot not found; top-level keys:", list(attrs.keys()))
else:
    meta_attrs = meta_node.get("attributes", {})
    print("meta.data columns:", list(meta_attrs.keys()))

# %%
# Build a pandas DataFrame from the meta.data columns
def extract_meta(meta_node):
    meta_attrs = meta_node.get("attributes", {})
    cols = {}
    for col_name, col_node in meta_attrs.items():
        if col_name in ("row.names", "names", "class"):
            continue
        vals = rds_list_to_values(col_node)
        if vals:
            cols[col_name] = vals
    # row names = cell barcodes
    row_names = rds_list_to_values(meta_attrs.get("row.names", {}))
    df = pd.DataFrame(cols, index=row_names if row_names else None)
    return df

meta_df = extract_meta(meta_node)
print(f"Cells: {len(meta_df)}  |  Columns: {list(meta_df.columns)}")
meta_df.head()

# %%
# Print value counts for any column that looks like a grouping variable
for col in meta_df.columns:
    n_unique = meta_df[col].nunique()
    if 1 < n_unique <= 50:
        print(f"\n--- {col} ({n_unique} levels) ---")
        print(meta_df[col].value_counts().to_string())

# %%
def safe_keys(node):
    if isinstance(node, dict):
        return list(node.get("attributes", node).keys())
    return []

print("Assays:    ", safe_keys(get_slot(attrs, "assays")))
print("Reductions:", safe_keys(get_slot(attrs, "reductions")))
print("Graphs:    ", safe_keys(get_slot(attrs, "graphs")))
print("Commands:  ", safe_keys(get_slot(attrs, "commands")))

# %%
reductions_node = get_slot(attrs, "reductions")

red_names = rds_list_to_values(reductions_node.get("attributes", {}).get("names", {}))
red_data  = reductions_node.get("data", [])
reductions = dict(zip(red_names, red_data))
print("Reductions found:", red_names)

for red_name, red_node in reductions.items():
    if not isinstance(red_node, dict):
        continue
    emb_node = get_slot(red_node, "attributes", "cell.embeddings")
    if emb_node is None:
        print(f"{red_name}: no cell.embeddings found")
        continue

    vals = rds_list_to_values(emb_node)
    dims = rds_list_to_values((emb_node.get("attributes") or {}).get("dim", {}))

    if vals is None or dims is None or len(dims) < 2:
        continue

    n_cells, n_dims = int(dims[0]), int(dims[1])
    print(f"{red_name}: {n_cells} cells × {n_dims} dims")

    mat = np.array(vals, dtype=float).reshape(n_cells, n_dims, order="F")
    colour_col = next((c for c in ["seurat_clusters", "celltype", "cell_type",
                                    "label", "orig.ident"] if c in meta_df.columns), None)

    fig, ax = plt.subplots(figsize=(7, 6))
    if colour_col:
        groups = meta_df[colour_col].astype(str)
        for grp in sorted(groups.unique()):
            idx = (groups == grp).values
            ax.scatter(mat[idx, 0], mat[idx, 1], s=1, alpha=0.5, label=grp)
        ax.legend(markerscale=6, bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=7, title=colour_col)
    else:
        ax.scatter(mat[:, 0], mat[:, 1], s=1, alpha=0.3, c="steelblue")

    ax.set_xlabel(f"{red_name}_1"); ax.set_ylabel(f"{red_name}_2")
    ax.set_title(f"{red_name} — human_RB.rds")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"01_rds_{red_name}.png", dpi=150)
    plt.show()
