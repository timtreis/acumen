import pandas as pd
import scanpy as sc
import squidpy as sq

adata = sc.read_h5ad("/tmp/slideseqv2.h5ad")  # sq.datasets.slideseqv2()

sq.gr.ripley(adata, cluster_key="cluster", mode="L", max_dist=500, seed=0)
res = adata.uns["cluster_ripley_L"]
df = res["L_stat"]
maxbin = df["bins"].max()
sub = df[df["bins"] == maxbin].set_index("cluster")["stats"]

pairs = [("Astrocytes", "Mural"), ("CA1_CA2_CA3_Subiculum", "Endothelial_Tip")]
for a, b in pairs:
    winner = a if sub[a] > sub[b] else b
    print(f"{a} ({sub[a]:.2f}) vs {b} ({sub[b]:.2f}) -> more clustered: {winner}")
