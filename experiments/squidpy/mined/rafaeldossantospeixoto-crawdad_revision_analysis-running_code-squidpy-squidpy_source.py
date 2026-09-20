# mined from: https://github.com/rafaeldossantospeixoto/crawdad_revision_analysis/blob/40b6c10b5ef21adf8b5d57c87883b0a00e35500f/running_code/squidpy/squidpy_source.py
# symbols: squidpy.pl.spatial_scatter

# Data

import squidpy as sq
import numpy as np
import pandas as pd
import numba.types as nt


import matplotlib.pyplot as plt

# pip install git+https://github.com/h2oai/datatable.git
import datatable

import os
import csv

meta = datatable.fread("exported_data/sim.csv")
meta = meta.to_pandas().set_index('C0')
meta.head()

from anndata import AnnData
from numpy.random import default_rng
rng = default_rng(42)

counts = rng.integers(0, 15, size=(meta.shape[0], 50))
counts.shape

adata = AnnData(counts, obsm={"spatial": np.array(meta[["x", "y"]], dtype=float)})

adata.obs = meta

sq.pl.spatial_scatter(
        adata,
        shape = None,
        color = "celltypes",
        size=3
)

