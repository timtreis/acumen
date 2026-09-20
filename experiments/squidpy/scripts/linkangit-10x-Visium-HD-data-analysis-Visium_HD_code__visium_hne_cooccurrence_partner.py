import squidpy as sq
import pandas as pd

adata = sq.datasets.visium_hne_adata()
cats = adata.obs["cluster"].cat.categories.tolist()

occ, interval = sq.gr.co_occurrence(adata, cluster_key="cluster", copy=True)

for target in ["Cortex_4", "Pyramidal_layer_dentate_gyrus"]:
    idx = cats.index(target)
    vals = pd.Series(occ[idx, :, 0], index=cats).drop(target).sort_values(ascending=False)
    print(f"TRAIN/TEST target={target} -> strongest co-occurrence partner at shortest distance: "
          f"{vals.index[0]} (ratio={vals.iloc[0]:.3f}); runner-up: {vals.index[1]} (ratio={vals.iloc[1]:.3f})")

# train answer: Cortex_4 -> Cortex_5
# test answer: Pyramidal_layer_dentate_gyrus -> Hippocampus
