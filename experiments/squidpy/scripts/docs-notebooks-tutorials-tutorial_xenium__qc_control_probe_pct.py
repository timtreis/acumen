"""
Task: qc_control_probe_pct
Reproduces the QC-metrics computation from tutorial_xenium.ipynb (cells 16-19):
percentage of total transcript counts attributable to negative-control probes.

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

URLS = {
    "lung": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip",
    "breast": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip",
}


def load_and_compute(sample: str) -> float:
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

    cprobes = adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100
    shutil.rmtree(workdir, ignore_errors=True)
    return cprobes


if __name__ == "__main__":
    for sample in ("lung", "breast"):
        pct = load_and_compute(sample)
        print(f"{sample}: negative control probe % = {pct:.6f} -> rounded 4dp = {round(pct, 4)}")
