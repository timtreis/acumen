"""
Ground-truth script for task `liver_moran_autocorr`.

Reproduces the "Autocorrelation: Moran's I Score" section of
docs/notebooks/tutorials/tutorial_vizgen_mouse_liver.ipynb using the real
Vizgen MERFISH Liver1Slice1 dataset (347+ gene panel, ~395k liver cells).

Vizgen gates the raw CSVs behind a registration form, but the same dataset is
mirrored, un-gated, as a Bioconductor ExperimentHub resource (SFEData package,
`VizgenLiverData()`, backed by an R `SpatialFeatureExperiment` object). This
script fetches that RDS via the public ExperimentHub location, exports the
counts matrix / cell metadata with a small R helper (base R + the built-in
Matrix package only -- no extra R packages needed), then reproduces the
squidpy analysis in Python.

Requires: R (Rscript) with the base `Matrix` package (ships with R), and
internet access to download the ~280MB RDS file (one-time).
"""

import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp
import squidpy as sq
import anndata as ad

WORKDIR = "/tmp/liver_moran_repro"
RDS_URL = "https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/SFEData/merfish_liver1.rds"
RDS_PATH = os.path.join(WORKDIR, "merfish_liver1.rds")

R_EXPORT_SCRIPT = """
library(Matrix)
x <- readRDS("{rds}")
a <- attributes(x)

ad_ <- attributes(a$assays)$data
ld <- attributes(ad_)$listData
m <- ld[[1]]  # genes x cells, dgCMatrix
writeMM(m, file = "{workdir}/liver_counts.mtx")
writeLines(rownames(m), "{workdir}/liver_genes.txt")
writeLines(colnames(m), "{workdir}/liver_cells.txt")

cd <- a$colData
df <- as.data.frame(attributes(cd)$listData)
rownames(df) <- attributes(cd)$rownames
sc_ <- attributes(a$int_colData)$listData$spatialCoords
df$center_x <- sc_[, "center_x"]
df$center_y <- sc_[, "center_y"]
df$cell_id <- rownames(df)
write.csv(df, "{workdir}/liver_cell_metadata.csv", row.names = FALSE)
cat("R export done\\n")
"""


def ensure_data():
    os.makedirs(WORKDIR, exist_ok=True)
    if not os.path.exists(RDS_PATH):
        print("Downloading Vizgen liver RDS from public ExperimentHub mirror...")
        urllib.request.urlretrieve(RDS_URL, RDS_PATH)

    mtx_path = os.path.join(WORKDIR, "liver_counts.mtx")
    if not os.path.exists(mtx_path):
        r_script_path = os.path.join(WORKDIR, "export.R")
        with open(r_script_path, "w") as f:
            f.write(R_EXPORT_SCRIPT.format(rds=RDS_PATH, workdir=WORKDIR))
        env = dict(os.environ)
        env["EDITOR"] = "vi"  # avoids a broken options("editor") on some R installs (empty EDITOR env var)
        env["VISUAL"] = "vi"
        subprocess.run(["Rscript", r_script_path], check=True, env=env)


def build_adata():
    m = sio.mmread(os.path.join(WORKDIR, "liver_counts.mtx")).tocsr().T.tocsr()  # cells x genes
    genes = [l.strip() for l in open(os.path.join(WORKDIR, "liver_genes.txt"))]
    cells = [l.strip() for l in open(os.path.join(WORKDIR, "liver_cells.txt"))]
    meta = pd.read_csv(os.path.join(WORKDIR, "liver_cell_metadata.csv"), dtype={"cell_id": str})
    meta = meta.set_index("cell_id").loc[cells]

    adata = ad.AnnData(X=m.astype(np.float32), obs=meta, var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].to_numpy()
    adata.var_names_make_unique()
    return adata


def main():
    ensure_data()
    adata = build_adata()

    # QC + normalization, mirroring the tutorial's preprocessing.
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=(50, 100, 200, 300), inplace=True)
    sc.pp.filter_cells(adata, min_counts=50)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial")
    sq.gr.spatial_autocorr(adata, mode="moran")

    moran = adata.uns["moranI"].sort_values("I", ascending=False)
    print(moran.head(10)[["I"]])

    top_gene = moran.index[0]
    second_gene = moran.index[1]
    print(f"\nTop spatially autocorrelated gene (train answer): {top_gene}")
    print(f"Runner-up spatially autocorrelated gene (test answer): {second_gene}")


if __name__ == "__main__":
    main()
