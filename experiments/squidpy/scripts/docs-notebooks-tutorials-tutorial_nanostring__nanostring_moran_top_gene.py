"""Ground truth for task 'nanostring_moran_top_gene'.

Goal: which gene shows the strongest spatial autocorrelation (Moran's I) in a
given field of view of the Lung5_Rep2 Nanostring CosMx lung sample.

train -> field of view 16
test  -> field of view 20
"""
import subprocess
import tarfile
from pathlib import Path

import scanpy as sc
import squidpy as sq

DATA_DIR = Path("data")
TAR_PATH = DATA_DIR / "Lung5_Rep2.tar.gz"
SAMPLE_ROOT = DATA_DIR / "Lung5_Rep2" / "Lung5_Rep2-Flat_files_and_images"
URL = (
    "https://nanostring-public-share.s3.us-west-2.amazonaws.com/"
    "SMI-Compressed/Lung5_Rep2/Lung5_Rep2+SMI+Flat+data.tar.gz"
)
MEMBERS = [
    "Lung5_Rep2/Lung5_Rep2-Flat_files_and_images/Lung5_Rep2_exprMat_file.csv",
    "Lung5_Rep2/Lung5_Rep2-Flat_files_and_images/Lung5_Rep2_metadata_file.csv",
    "Lung5_Rep2/Lung5_Rep2-Flat_files_and_images/Lung5_Rep2_fov_positions_file.csv",
]

if not SAMPLE_ROOT.exists():
    DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(["curl", "-sL", "-o", str(TAR_PATH), URL], check=True)
    with tarfile.open(TAR_PATH) as tf:
        tf.extractall(DATA_DIR, members=[m for m in tf.getmembers() if m.name in MEMBERS])
    TAR_PATH.unlink()

sample_dir = SAMPLE_ROOT
adata = sq.read.nanostring(
    path=sample_dir,
    counts_file="Lung5_Rep2_exprMat_file.csv",
    meta_file="Lung5_Rep2_metadata_file.csv",
    fov_file="Lung5_Rep2_fov_positions_file.csv",
)

adata.var["NegPrb"] = adata.var_names.str.startswith("NegPrb")
sc.pp.calculate_qc_metrics(adata, qc_vars=["NegPrb"], inplace=True)

sc.pp.filter_cells(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=400)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

for label, fov in [("train", "16"), ("test", "20")]:
    sub = adata[adata.obs.fov == fov].copy()
    sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
    sq.gr.spatial_autocorr(sub, mode="moran", n_perms=None, n_jobs=1)
    print(label, fov, sub.uns["moranI"].head(5))
    # train answer: KRT19
    # test answer: TYK2
