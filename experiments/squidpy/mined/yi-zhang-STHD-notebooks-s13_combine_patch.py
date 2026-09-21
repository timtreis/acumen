# mined from: https://github.com/yi-zhang/STHD/blob/b2a95623369b49d74163b93f76e054212c91c332/notebooks/s13_combine_patch.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import squidpy as sq
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
# will be avialable on pip

from STHD import patchify
from STHD import color_palette
from STHD import train

# %%
path = '../testdata/crop10large/'
refile = '../testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt'
patchify.merge(path, refile)

# %%
sthdata_with_pdata = train.load_data_with_pdata(path + '/all_region')

# %%
crop_coor = sthdata_with_pdata.get_sequencing_data_region()

cmap = color_palette.get_config_colormap( name = 'colormap_coloncatlas_98')
data_palette = color_palette.prepare_palette(cmap, sthdata_with_pdata.adata)

f, ax = plt.subplots(2, 1, figsize=(20,20))

sq.pl.spatial_scatter(sthdata_with_pdata.adata, 
                      color='STHD_pred_ct', 
                      crop_coord = [crop_coor],
                      legend_fontsize=8,
                      palette = data_palette,
                      ax = ax[1]
                     )
