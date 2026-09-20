# mined from: https://github.com/yi-zhang/STHD/blob/b2a95623369b49d74163b93f76e054212c91c332/notebooks/s03_run_STHD_model_patch.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numba
import squidpy as sq
from numba import jit, njit, prange

import pandas as pd
import matplotlib.pyplot as plt
import squidpy as sq
import sys
sys.path.append('../STHD')
import model
import refscrna
import sthdio
import train
import qcmask
import color_palette
import config

# %%
patch_path = '../analysis/20240527_update_sthd/crop10_large/'
refile = '/hpc/home/yz922/yizhanglab/yz922/proj/STHD/STHD_data/colon_cancer_celltype_average_expr_genenorm_rctd_style_0525_0.000125_log2.0.5_4618gs.txt'
# preset parameters
n_iter =  20
step_size = 1
beta = 0.1

# %%
sthdata = train.load_data(patch_path)
print(sthdata.adata.shape)
sthdata.adata = qcmask.background_detector(sthdata.adata, threshold = 50, n_neighs =4, n_rings = 2)
print(sthdata.adata.shape)
qcmask.visualize_background(sthdata)

# %%
sthdata = train.load_data(patch_path)
print(sthdata.adata.shape)
sthdata.adata = qcmask.background_detector(sthdata.adata, threshold = 50, n_neighs =4, n_rings = 2)
print(sthdata.adata.shape)
#qcmask.visualize_background(sthdata)
sthdata_filtered = qcmask.filter_background(sthdata, threshold = 50 )
print(sthdata.adata.shape)
sthdata_filtered, genemeanpd_filtered = train.sthdata_match_refgene(sthdata_filtered, refile)
P_filtered = train.train(sthdata_filtered, n_iter, step_size, beta)
P = train.fill_p_filtered_to_p_full(P_filtered, sthdata_filtered, genemeanpd_filtered, sthdata )
sthdata = train.predict(sthdata, P, genemeanpd_filtered, mapcut= 0.8)
pdata = train.save_prediction_pdata(sthdata, file_path = patch_path)

# %%
'''
### Or, train simply without masking any region.

sthdata = load_data(patch_path)
sthdata, genemeanpd_filtered = sthdata_match_refgene(sthdata, refile, matchrefgene=True, ref_renorm = False)
P = train.train(sthdata, n_iter, step_size, beta)
sthdata = train.predict(sthdata, P, genemeanpd_filtered, mapcut= 0.8) 
pdata = train.save_prediction_pdata(sthdata, file_path = patch_path, prefix = '')
'''

# %%
cmap = color_palette.get_config_colormap( name = 'colormap_coloncatlas_98')
data_palette = color_palette.prepare_palette(cmap, sthdata.adata)

sq.pl.spatial_scatter(sthdata.adata, 
                      color='STHD_pred_ct', 
                      crop_coord = [sthdata.get_sequencing_data_region()],
                      legend_fontsize=8,
                      palette = data_palette,
                      figsize=(12,12)
                    
                     )
#plt.savefig('../analysis/figure/fig1c_crop10_pred_ct.pdf',dpi=300)

# %%
sthdata = train.load_data_with_pdata(file_path = patch_path)

# %%
# -----------------------Quick Visualization----------------------------
def analysis(sthd_data, color_list1, color_list2):
    adata = sthd_data.adata.copy()
    crop_coor = sthdata.get_sequencing_data_region()
    
    f, ax = plt.subplots(2, 3, figsize=(40, 20)) # row, col
    
    cur_row = 0
    for i, c in enumerate(color_list1):
        sq.pl.spatial_scatter(adata, color=c, crop_coord = [crop_coor], ax=ax[cur_row, i], vmin=0) #[]todo. we put p=-1 for the filtered spots
        
    cur_row = 1
    for i, c in enumerate(color_list2):
        sq.pl.spatial_scatter(adata, color=c, crop_coord = [crop_coor], ax=ax[cur_row, i], vmin=0)
    return(adata)

# %%
## coloring a few cell types's probablity
color_list1 = ['p_ct_Tumor cE03 (Stem/TA-like prolif)','p_ct_Tumor cE05 (Enterocyte 2)', 'p_ct_Tumor cE01 (Stem/TA-like)']
color_list2 = ['p_ct_cP2 (Plasma IgG)', 'p_ct_cM02 (Macrophage-like)', 'STHD_pred_ct', ]
adata = analysis(sthdata, color_list1, color_list2)

# %%
crop_coor = sthdata.get_sequencing_data_region()

# %%
#fig, ax = plt.subplots( figsize=(6,6))
sq.pl.spatial_scatter(adata, color=color_list1[0], crop_coord = [crop_coor],  vmin=0)
plt.savefig('../analysis/figure/crop10_prob_cE03.pdf',dpi=300)

# %%
#fig, ax = plt.subplots( figsize=(6,6))
sq.pl.spatial_scatter(adata, color=color_list1[1], crop_coord = [crop_coor],  vmin=0)
plt.savefig('../analysis/figure/crop10_prob_cE05.pdf',dpi=300)

# %%
#fig, ax = plt.subplots( figsize=(6,6))
sq.pl.spatial_scatter(adata, color=color_list1[2], crop_coord = [crop_coor],  vmin=0)
plt.savefig('../analysis/figure/crop10_prob_cE01.pdf',dpi=300)

# %%
#fig, ax = plt.subplots( figsize=(6,6))
sq.pl.spatial_scatter(adata, color=color_list2[0], crop_coord = [crop_coor],  vmin=0)
plt.savefig('../analysis/figure/crop10_prob_plasma.pdf',dpi=300)

# %%
#fig, ax = plt.subplots( figsize=(6,6))
sq.pl.spatial_scatter(adata, color=color_list2[1], crop_coord = [crop_coor],  vmin=0)
plt.savefig('../analysis/figure/crop10_prob_macro.pdf',dpi=300)
