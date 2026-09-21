# mined from: https://github.com/gao-lab/SLAT/blob/dfb2b01d6fd52d5eeb88b8f17a69be6daaacdad4/benchmark/label_data/ploty_label.ipynb
# symbols: squidpy.datasets.visium_hne_adata, squidpy.datasets.visium_hne_image

# %%
import matplotlib.pyplot as plt
import scanpy as sc
import cv2

import plotly.graph_objs as go
import plotly.offline as py

import pandas as pd
import numpy as np
from ipywidgets import interactive, HBox, VBox

# %%
adata = sc.read_h5ad('../../../data/stereo_seq/counts/E11.5/count_E11.5_E1S1.MOSTA.h5ad')
# adata = sc.pp.subsample(adata, n_obs=500, copy=True)

# %%
plt.rcParams["figure.figsize"] = [8, 8]
png = sc.pl.spatial(adata, spot_size=3, color='annotation',return_fig=True)

# %%
def var2color(df:pd.DataFrame,col:str)->pd.DataFrame:
    r"""
    Random add color by the class of a column in dataframe
    """
    import random
    celltypes = list(set(df[col]))
    c_map = {}
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(celltypes))]
    for i, celltype in enumerate(celltypes):
        c_map[celltype] = color[i]
    df['color'] = [*map(c_map.get, df[col])]
    return df

# %%
# Create dataframe of cell location and cell type
df = pd.DataFrame({'cell_name':'adata1-' + adata.obs.index, 
                    'x':adata.obsm['spatial'][:,0],
                    'y':adata.obsm['spatial'][:,1],
                    'cell_type':adata.obs['annotation']})
df = var2color(df,'cell_type')

# %%
df

# %%
# ref https://stackoverflow.com/questions/58102719/creating-pandas-dataframe-from-the-data-points-selected-on-the-plotly-scatter-pl 
py.init_notebook_mode()
f = go.FigureWidget([go.Scatter(y=df['x'], x=df['y'], mode='markers', marker_color=df['color'], marker_size=3)])
f.update_layout(
    autosize=False,
    width=1200,
    height=1200)

scatter = f.data[0]
scatter.x = df['x']
scatter.y = df['y']
scatter.marker.opacity = 0.5

t = go.FigureWidget([go.Table(
    header=dict(values=['cell_name','x','y','cell_type'],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[df[col] for col in ['cell_name','x','y','cell_type']],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))])

_select_i = {}
_i = 0

def selection_fn(trace,points,selector):
    global _i
    global _select_i
    _i = int(_i)
    print(f'Selecting {str(_i)}th region')
    # print(points.point_inds)
    t.data[0].cells.values = [df.iloc[points.point_inds][col] for col in ['cell_name','x','y','cell_type']]
    _select_i['select_'+str(_i)] = df.iloc[points.point_inds]
    _i += 1

scatter.on_selection(selection_fn)

# Put everything together
VBox((f,t))

# %%
df_dics = t.to_dict()
df_dics.keys()

# %%
df_dics['data']

# %%
import napari
from skimage import data
import squidpy as sq

# %%
adata = sq.datasets.visium_hne_adata()
img = sq.datasets.visium_hne_image()

# %%
viewer = img.interactive(adata)
