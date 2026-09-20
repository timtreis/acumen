# mined from: https://github.com/shayshay42/spatial_transcriptomics_playground/blob/d34b9ec81ba5d74a76f3d568a5291b57474f5653/ST_tutorial.py
# symbols: squidpy.datasets.seqfish, squidpy.pl.spatial_scatter

import numpy as np

import scanpy as sc
import squidpy as sq

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# load the pre-processed datast
adata = sq.datasets.seqfish()  

sq.pl.spatial_scatter(
    adata, color="celltype_mapped_refined", shape=None, figsize=(10, 10)
)