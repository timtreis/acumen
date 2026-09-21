"""Ground truth for task 'nanostring_negprobe_pct'.

Goal: what percentage of total transcript counts come from negative control
probes (background noise), for the Lung5_Rep2 Nanostring CosMx lung sample.

train  -> whole sample
test   -> restricted to field of view 12
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

# train: whole sample
pct_all = adata.obs["total_counts_NegPrb"].sum() / adata.obs["total_counts"].sum() * 100
print("train (whole sample):", round(pct_all, 2))  # 0.37

# test: field of view 12 only
sub = adata[adata.obs["fov"] == "12"]
pct_fov12 = sub.obs["total_counts_NegPrb"].sum() / sub.obs["total_counts"].sum() * 100
print("test (fov 12):", round(pct_fov12, 2))  # 0.33
