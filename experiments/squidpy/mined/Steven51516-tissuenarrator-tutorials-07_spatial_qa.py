# mined from: https://github.com/Steven51516/tissuenarrator/blob/6b8bfb46c91031da4ff32a4a08794cd950e90e53/tutorials/07_spatial_qa.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd()))
import tutorial_utils as tu
from tutorial_utils import data_path
import qa_build as qb
tu.setup_style()
print("ready")

# %%
import anndata as ad
merfish = ad.read_h5ad(data_path("merfish_preprocessed.h5ad"))
hxg = qb.build_hxg_qa(merfish, region_col="parcellation_structure", min_cells=20)
print(f"{len(hxg)} items over {hxg.region.nunique()} regions x {hxg.cell_class.nunique()} classes\n")
print("PROMPT:\n", hxg.prompt.iloc[0])
print("\nANSWER:\n", hxg.answer.iloc[0])

# %%
secs = merfish.obs["section"].astype(str).value_counts().index[:2].tolist()
sub = merfish[merfish.obs["section"].astype(str).isin(secs)].copy()
cell_df = qb.cell_sentences_from_adata(sub)
rvf = qb.build_real_vs_fake_qa(cell_df)
print(f"{len(rvf)} items | labels: {rvf.answer.value_counts().to_dict()}\n")
print("PROMPT (real example, truncated):\n", rvf[rvf.answer=='Yes'].prompt.iloc[0][:520], "...")
print("\nANSWER:", rvf[rvf.answer=='Yes'].answer.iloc[0])

# %%
import squidpy as sq
pert = ad.read_h5ad(data_path("pertfish_tumors_test.h5ad"))
pert.obsm["spatial"] = pert.obsm["spatial"] * 0.108
sq.gr.spatial_neighbors(pert, coord_type="generic", radius=30)
PERTS = qb.single_perturbation_labels(pert)   # all single-KO labels (matches production)
de_df, nonde_df = qb.compute_perturb_de(pert, PERTS)
print(f"{len(PERTS)} single-KO labels; DE computed for {de_df.perturbation.nunique()} "
      f"(others skipped: too few neighboring T cells)")
cancer_pool = qb.load_cancer_pool(pd.read_parquet(data_path("pertfish_neighbor.parquet")))
fwd = qb.build_forward_spatial_qa(de_df, nonde_df, cancer_pool)
print(f"{len(fwd)} items | answers: {fwd.answer.value_counts().to_dict()}\n")
print("PROMPT (truncated):\n", fwd.prompt.iloc[0][:680], "...")
print("\nANSWER:", fwd.answer.iloc[0])

# %%
pw = qb.build_pathway_spatial_qa(de_df, nonde_df, cancer_pool, panel_genes={}, permutation_num=100)
print(f"{len(pw)} items | answers: {pw.answer.value_counts().to_dict()}\n")
if len(pw):
    print("PROMPT (truncated):\n", pw.prompt.iloc[0][:560], "...")
    print("\nANSWER:", pw.answer.iloc[0])

# %%
import matplotlib.pyplot as plt
TABLE = pd.read_csv(data_path("qa_results.csv")).set_index("task")
MODELS = ["v2_base", "v2_tn"]
ABBR = {"v2_base": "Base-sft", "v2_tn": "TN-sft"}
COLORS = {"v2_base": "#c6cae3", "v2_tn": "#99a1c9"}
TASKS = [("highly_expressed_genes", "Overlap", (0, 0.8)),
         ("spatial_real_vs_fake",   "Accuracy", (0, 1.0)),
         ("forward_spatial",        "Accuracy", (0, 0.6)),
         ("pathway_spatial",        "Accuracy", (0, 0.8))]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (task, ylabel, ylim) in zip(axes, TASKS):
    vals = TABLE.loc[task, MODELS].astype(float).values
    ax.bar(np.arange(len(MODELS)), vals, width=0.6, color=[COLORS[m] for m in MODELS])
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([ABBR[m] for m in MODELS], rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=14); ax.set_ylim(*ylim)
    ax.set_title(f"{task}\n({TABLE.loc[task, 'metric']})", fontsize=12)
    for pos in ("top", "right"): ax.spines[pos].set_visible(False)
    ax.spines["left"].set_linewidth(1.5); ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="y", labelsize=12, width=1.5); ax.tick_params(axis="x", labelsize=12)
plt.tight_layout(); plt.show()
