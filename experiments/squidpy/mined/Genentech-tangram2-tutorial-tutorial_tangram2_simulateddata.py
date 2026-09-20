# mined from: https://github.com/Genentech/tangram2/blob/0af3056ff537c4e063b960a5f02a60b1711a89c4/tutorial/tutorial_tangram2_simulateddata.ipynb
# symbols: squidpy.datasets.sc_mouse_cortex

# %%
import tangram2 as tg2
import anndata as ad
import numpy as np
import scanpy as sc
import squidpy as sq

from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots a bit nicer
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (5, 4)

# %%
adata = sq.datasets.sc_mouse_cortex()
label_col = "cell_subclass"

# show counts per subclass
adata.obs[label_col].value_counts().head()

# choose receiver and sender from the two most frequent subclasses
receiver_name, signaler_name = adata.obs[label_col].value_counts().index[0:2]
print("Receiver subclass:", receiver_name)
print("Sender   subclass:", signaler_name)

# %%
def pp_adata(adata_sc: ad.AnnData, n_top_genes: int = 2000) -> ad.AnnData:
    """Basic preprocessing: library-size normalize, log1p, mark HVGs."""
    adata_sc = adata_sc.copy()
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.pp.highly_variable_genes(
        adata_sc, n_top_genes=n_top_genes, subset=False
    )
    return adata_sc

# %%
def run_one_simulation(
    adata_seed: ad.AnnData,
    label_col: str,
    receiver_name: str,
    signaler_name: str,
    effect_direction: str = "up",
    effect_scaling: float = 3.0,
    effect_base: float = 0.95,
    n_spots: int = 250,
    n_effect_genes: int = 25,
    seed: int = 0,
):
    """
    One synthetic CCC experiment:

      1) generate synthetic spatial + scRNA with Tangram2-evalkit cellmix
      2) map cells with Tangram2-mapping (via evalkit.met)
      3) infer interactions with Tangram2-CCC
      4) return:
           - synthetic ad_sp, ad_sc
           - beta_df: interaction coefficients for all receiver/sender/gene combinations
           - params: the synthetic settings used

    NOTE: This function does *not* perform any evaluation (no AUROC).
          Evaluation is done later using 'beta_df' and 'ad_sc'.
    """
    np.random.seed(seed)

    # --- 3.1 Generate synthetic data with cellmix ---------------------------
    ad_sp, ad_sc = tg2.evalkit.datagen.cellmix.cellmix.cellmix(
        adata_seed.copy(),
        n_spots=n_spots,
        n_cells_per_spot=10,
        n_types_per_spot=5,
        label_col=label_col,
        signaler_names=signaler_name,
        receiver_names=receiver_name,
        n_interactions=1,
        effect_size=n_effect_genes,
        effect_direction=effect_direction,
        signal_effect_base=effect_base,
        signal_effect_scaling=effect_scaling,
        p_inter=0.8,
        p_signal_spots=0.9,
    )

    # --- 3.2 Preprocess synthetic scRNA-seq and get HVGs -------------------
    ad_sc = pp_adata(ad_sc)
    hvg_genes = ad_sc.var_names[ad_sc.var.highly_variable.values].tolist()

    # --- 3.3 Prepare evalkit "input_dict" for Tangram2-mapping ------------
    input_dict = tg2.evalkit.met.utils.adatas_to_input(
        {"from": ad_sc.copy(), "to": ad_sp.copy()},
        categorical_labels={"from": [label_col]},  # include subclass in design matrix
    )

    # Standard Tangram2 preprocessing (normalization, gene filtering, etc.)
    tg2.evalkit.met.pp.StandardTangram2.run(input_dict)

    # --- 3.4 Run Tangram2-mapping -----------------------------------------
    map_res = tg2.evalkit.met.map_methods.Tangram2Map.run(
        input_dict,
        num_epochs=1000,
        genes=hvg_genes,
    )

    # Update input_dict with mapping result (so CCC sees the mapping)
    input_dict.update(map_res)

    # --- 3.5 Run Tangram2-CCC on mapped data ------------------------------
    inter_res = tg2.ccc.TangramCCC.run(
        input_dict,
        n_epochs=1000,
        seed=seed,
        verbose=False,
    )

    # --- 3.6 Extract interaction coefficients only -------------------------
    # inter_res["beta"] is a (receiver_type x sender_type x gene) tensor-like object
    beta_df = inter_res["beta"].to_dataframe()["beta"]
    # beta_df is a pandas Series with a MultiIndex (receiver, sender, gene)

    return {
        "ad_sp": ad_sp,
        "ad_sc": ad_sc,
        "beta_df": beta_df,
        "params": dict(
            effect_direction=effect_direction,
            effect_scaling=effect_scaling,
            effect_base=effect_base,
            n_spots=n_spots,
            n_effect_genes=n_effect_genes,
        ),
    }

# %%
demo_up = run_one_simulation(
    adata_seed=adata,
    label_col=label_col,
    receiver_name=receiver_name,
    signaler_name=signaler_name,
    effect_direction="up",
    effect_scaling=3.0,
    effect_base=0.99,
    n_spots=250,
    n_effect_genes=25,
    seed=0,
)

# %%
demo_up['params']

# %%
beta_up = demo_up['beta_df'].reset_index()
beta_up['inter'] = beta_up['labels'].astype(str) + '_vs_' + beta_up['labels_'].astype(str)
beta_up.drop(labels=['labels', 'labels_'], inplace=True, axis=1)

# %%
beta_up

# %%
# 5.1 Extract beta for the chosen receiver–sender pair
beta_df = demo_up["beta_df"]
ad_sc_up = demo_up["ad_sc"]

# scores for this specific receiver–sender pair
scores_up = beta_df.loc[receiver_name, signaler_name].sort_values(ascending=False)
scores_up = pd.DataFrame(scores_up, columns=["beta"])
scores_up["names"] = scores_up.index

# %%
# 5.2 Ground-truth effect genes are named "effect_*" in the synthetic scRNA
effect_genes_up = [g for g in ad_sc_up.var.index if g.startswith("effect_")]

# add a column flagging ground-truth effect genes
scores_up["is_effect"] = scores_up["names"].isin(effect_genes_up)

# %%
# 5.3 Compute AUROC of beta vs. GT effect gene set
dea_score_up = tg2.evalkit.dig.dea.compute_dea_score(
    scores_up,
    effect=effect_genes_up,
    score_by="beta",
    method="auroc",
    reverse=False,  # effect_direction == "up"
)

print("AUROC (up-regulated effects):", dea_score_up["score"])

# %%
# ### 5.2 Simple visualization: beta for effect vs. non-effect genes

# %%
plt.figure(figsize=(4, 4))
sns.boxplot(
    data=scores_up,
    x="is_effect",
    y="beta",
)
plt.xlabel("Ground-truth effect gene?")
plt.ylabel("Interaction coefficient (beta)")
plt.xticks([0, 1], ["No", "Yes"])
plt.title("Up-regulated interaction: effect vs non-effect genes")
plt.tight_layout()
plt.show()

# %%

