"""
Ground truth for task `imc_centrality`.

Builds the spatial neighbor graph for the imc dataset and computes
per-cell-type centrality scores (degree centrality) within that graph.
Reports the cell type that is most centrally connected, and the one that
is least centrally connected.
"""

import squidpy as sq

adata = sq.datasets.imc()

sq.gr.spatial_neighbors(adata)
sq.gr.centrality_scores(adata, cluster_key="cell type")

df = adata.uns["cell type_centrality_scores"].sort_values("degree_centrality")
print(df)
print("least connected (lowest degree centrality):", df["degree_centrality"].idxmin())
print("most connected (highest degree centrality):", df["degree_centrality"].idxmax())

# train answer (most connected): apoptotic tumor cell
# test answer (least connected): CK low HR low tumor cell
