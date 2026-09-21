import squidpy as sq
import pandas as pd

adata = sq.datasets.slideseqv2()
cats = adata.obs["cluster"].cat.categories.tolist()

sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.interaction_matrix(adata, cluster_key="cluster")
im = adata.uns["cluster_interactions"]
df = pd.DataFrame(im, index=cats, columns=cats)

for target in ["Microglia", "Neurogenesis"]:
    row = df.loc[target].drop(target).sort_values(ascending=False)
    print(target, "-> most direct spatial-neighbor contacts with:", row.index[0], round(row.iloc[0], 0))

# train answer: target "Microglia" -> "Astrocytes"
# test answer:  target "Neurogenesis" -> "DentatePyramids"
