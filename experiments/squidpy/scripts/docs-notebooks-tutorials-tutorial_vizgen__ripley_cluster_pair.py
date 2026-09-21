import squidpy as sq

adata = sq.datasets.merfish()

# Restrict to the Bregma -29 tissue section (batch '0'), used for both variants.
sub = adata[adata.obs["batch"] == "0"].copy()
sub.obs["Cell_class"] = sub.obs["Cell_class"].cat.remove_unused_categories()

sq.gr.ripley(sub, cluster_key="Cell_class", mode="L", seed=0)
df = sub.uns["Cell_class_ripley_L"]["L_stat"]
last_bin = df["bins"].max()
obs = df[df["bins"] == last_bin].set_index("Cell_class")["stats"]

for label, a, b in [
    ("train", "Astrocyte", "Endothelial 1"),
    ("test", "OD Immature 1", "Microglia"),
]:
    va, vb = obs[a], obs[b]
    winner = a if va > vb else b
    print(label, a, va, "|", b, vb, "-> more clustered:", winner)
