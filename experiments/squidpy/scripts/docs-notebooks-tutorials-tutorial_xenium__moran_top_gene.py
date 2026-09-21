"""
Task: moran_top_gene
Reproduces the spatial-autocorrelation section of tutorial_xenium.ipynb (cells 63-64):
build a Delaunay spatial neighbor graph and compute Moran's I per gene, report
the gene with the highest score (strongest spatially clustered expression pattern).

train -> 10x Xenium human lung 2-FOV example dataset
test  -> 10x Xenium human breast 2-FOV example dataset
"""
import os
import shutil
import subprocess
import tempfile
import zipfile

import pandas as pd
import scanpy as sc
import squidpy as sq

URLS = {
    "lung": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip",
    "breast": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip",
}


def load(sample: str):
    workdir = tempfile.mkdtemp()
    zip_path = os.path.join(workdir, "outs.zip")
    subprocess.run(["curl", "-sL", URLS[sample], "-o", zip_path], check=True)
    extract_dir = os.path.join(workdir, "outs")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    adata = sc.read_10x_h5(os.path.join(extract_dir, "cell_feature_matrix.h5"), gex_only=False)
    adata.var_names_make_unique()
    adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()

    cells = pd.read_csv(os.path.join(extract_dir, "cells.csv.gz"), index_col="cell_id")
    cells = cells.loc[adata.obs_names]
    adata.obs = adata.obs.join(cells)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    shutil.rmtree(workdir, ignore_errors=True)
    return adata


def top_moran_gene(adata) -> str:
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=1, seed=0)
    moran = adata.uns["moranI"]
    return moran.index[0], moran.iloc[0]["I"]


if __name__ == "__main__":
    for sample in ("lung", "breast"):
        adata = load(sample)
        gene, score = top_moran_gene(adata)
        print(f"{sample}: top Moran's I gene = {gene} (I={score:.4f})")
