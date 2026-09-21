# mined from: https://github.com/frankligy/exercise_codes/blob/034483cea18d1118635a6c4864da3670db165a3e/spatial/seqFISH.py
# symbols: squidpy.datasets.seqfish, squidpy.gr.co_occurrence, squidpy.gr.ligrec, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.ligrec, squidpy.pl.nhood_enrichment

import scanpy as sc
import squidpy as sq
import numpy as np

# load
adata = sq.datasets.seqfish()

# plot
sc.pl.spatial(adata, color="celltype_mapped_refined", spot_size=0.03)

# spatial graph
sq.gr.spatial_neighbors(adata, coord_type="generic")

# neighborhood enrichment
sq.gr.nhood_enrichment(adata, cluster_key="celltype_mapped_refined")
sq.pl.nhood_enrichment(adata, cluster_key="celltype_mapped_refined", method="ward")

# co-occurence
sq.gr.co_occurrence(adata, cluster_key="celltype_mapped_refined")
sq.pl.co_occurrence(adata,cluster_key="celltype_mapped_refined",clusters="Lateral plate mesoderm",figsize=(10, 5))

# ligand-receptor analysis
sq.gr.ligrec(adata,n_perms=100,cluster_key="celltype_mapped_refined")
sq.pl.ligrec(adata,cluster_key="celltype_mapped_refined",source_groups="Lateral plate mesoderm",target_groups=["Intermediate mesoderm", "Allantois"],means_range=(0.3, np.inf),alpha=1e-4,swap_axes=True)