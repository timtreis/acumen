# mined from: https://github.com/NBISweden/workshop-spatial/blob/0cb2bfbda0b88cd042173dc007135ed5b8a95fc4/labs/03_1_segmentation_free_supervised.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
# These seem to be necessary for linux:

# %pip install ipywidgets==7.7.1 jupyterlab-widgets==1.1.1 planktonspace=0.0.9

# %%


# widens the screen:
from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# imports:
import sys
import os
import plankton.plankton as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# convenience function to create new figures:
def figure(width=8,height=8):
    plt.figure(figsize=(width,height))

# %%
# Data folder location:
data_root = '../data/in_situ_sequencing'
assert os.path.exists(data_root)

# Define um_p_px parameter for the data set coordinates:
um_p_px = 0.325

# Read coordinate/gene data from .csv file
coordinates = pd.read_csv(os.path.join(data_root,'S2T1_pcw6.csv'))

# Extract x,y coordinates and gene labels 
x =  coordinates.Global_x_pos.values 
y =  coordinates.Global_y_pos.values 
g =  coordinates.Gene.values

# %%
# Create a plankton-SpatialData object with the coordinates:

sdata = pl.SpatialData(
                        coordinates.Gene,
                        coordinates.Global_x_pos*um_p_px,
                        coordinates.Global_y_pos*um_p_px,
                        )

# display data in the notebook:
sdata

# %%
# inspect basic statistics at gene level:

sdata.stats

# %%
# Plot a bar graph of the gene counts in the data set:

figure(22,5)
sdata.counts.sort_values().plot.bar()

# %%
# Plot data set overview/summary:

sdata.plot_overview()

plt.gcf().set_size_inches(17.5, 9.5)

# %%
# use the 'scatter' function to get familiar with the data set:

figure(10,10)

sdata.scatter(alpha=0.2)

# %%
# Load staining image as .jpg:

figure(7,14)
bg = -plt.imread(os.path.join(data_root,'background.jpg')).mean(-1)
bg = (bg-bg.min())/(bg.max()-bg.min())

plt.subplot(121)

plt.title('original')

plt.imshow(bg,cmap='Greys')

# Create PixelMap

bg_map = pl.PixelMap(pixel_data=bg,
                     cmap='Greys',
                     px_p_um = 0.504/um_p_px)

plt.subplot(122)

plt.title('PixelMap with affine transform (rescale:)')
bg_map.imshow()

# %%
sdata = pl.SpatialData(
                        coordinates.Gene,
                        coordinates.Global_x_pos*um_p_px,
                        coordinates.Global_y_pos*um_p_px,
                        pixel_maps={'DAPI':bg_map}
                        )

# %%
plt.figure(figsize=(19,8))

plt.subplot(1,3,1)
plt.title('coordinates')
plt.scatter(*sdata.coordinates[:,:].T*np.array([[1],[-1]]),c=sdata.var.c_genes[sdata.gene_ids],marker='.',alpha=0.1)

plt.subplot(1,3,2)
plt.title('plankton')
sdata.scatter(alpha=0.1)

ax=plt.subplot(1,3,3)
plt.title('DAPI stain')
bg_map.imshow(axd=ax)

# %%
plt.figure(figsize=(20,7))


# Slice using array notation:
plt.subplot(1,4,1)
plt.title('subsampled by 200:')
sdata[::200].scatter()

# Subsample using boolean mask:
plt.subplot(1,4,2)
plt.title('subsampled for HGF,WNT2:')
sdata[sdata.g.isin(['HGF','WNT2'])].scatter(legend=True)

# Crop using spatial view:
plt.subplot(1,4,3)
plt.title('subsampled in space:')
sdata.spatial[100:2800,1000:].scatter(alpha=0.1)

# %%
figure(9,9)

low_count_gene_mask = (sdata.counts<200)

sdata[low_count_gene_mask[sdata.gene_ids]].scatter(marker='x',legend=True)

# %%
import anndata

# Read provided count table:
cellwise_counts = pd.read_csv(os.path.join(data_root,'S2T1_pcw6_complex_celltypes_formatted.csv'),index_col=0)

# Group cell-subtypes for ease of interpretability:
celltypes = cellwise_counts['cell type'].values
for i,c in enumerate(celltypes):
    if c[-1].isdigit():
        celltypes[i]=c[:-2]

# create scanpy/AnnData object from molecule count matrix, 
# containing 7997 cells and 141 genes:
adata = anndata.AnnData(X = cellwise_counts.iloc[:,78:],)

# add celltype labels to the individual cells
adata.obs['celltype'] = celltypes

# %%
# create new sdata object with added single-cell data:
sdata = pl.SpatialData(
                        genes=g,
                        x_coordinates=x*um_p_px,
                        y_coordinates=y*um_p_px,
                        pixel_maps={'DAPI':bg_map},
                        scanpy=adata
                        ).clean()

# show signature matrix
sdata.scanpy

# %%
# Generate signature matrix
signatures = sdata.scanpy.generate_signatures()

signatures

# %%
from plankton.utils import ssam

# Create a celltype map using the ssam algorithm:

kernel_bandwidth = 5   # Bandwidth for the Gaussian KDE smoothing kernel
patch_length = 2000     # length of the individual data batches 
threshold_corr = 0.1    # Threshold for expression-signature correlation 
threshold_exp = 0.3    # Threshold for total signal norm

ctmap = ssam(sdata,signatures=signatures,kernel_bandwidth=kernel_bandwidth,
            patch_length=patch_length,threshold_cor=threshold_corr,threshold_exp=threshold_exp)

# %%
# sample the map's values at all molecule locations:
values_at_xy = ctmap.get_value(sdata.x,sdata.y)

# assign tissue label to sampled values:
celltype_labels = np.array(signatures.index)[values_at_xy]
celltype_labels[ctmap.get_value(sdata.x,sdata.y)==-1]='other'

# add 'celltype' annotation to each molecule of the sdata frame:
sdata['celltype']= celltype_labels
sdata['celltype'] = sdata.celltype.astype('category')

sdata

# %%
from matplotlib.cm import get_cmap

# Colored scatter points to create the legend:
labels = sdata.celltype.cat.categories
cm = get_cmap('nipy_spectral')
tissue_colors = [cm((i+1)/(len(labels)-1)) for i in range(len(labels)-1)]


# Show celltype map:

figure(15,20)


ctmap.imshow(cmap='nipy_spectral',interpolation='none')

handles = [plt.scatter([],[],color=tissue_colors[i]) for i in range(len(labels)-1)]
plt.legend(handles,labels,)

# %%
figure(25,25)

for i,g in enumerate(signatures.index):
    
    plt.subplot(7,7,i+1)
    
    plt.title(g)
    
    (ctmap==i).imshow(cmap='Reds')

# %%
import squidpy as sq

sq.gr.spatial_neighbors(sdata, key_added='spatial')
sq.gr.nhood_enrichment(sdata,'celltype')

# %%
sq.pl.nhood_enrichment(sdata,'celltype')

# %%
# compute co-occurrence indicator for each class-class-pair:
cooc = sdata.stats.co_occurrence(resolution=5,max_radius=200,linear_steps=40,category='celltype')

# %%
autos = cooc.diagonal()

figure(20,10)

for i,c in enumerate(tissue_colors):
    _=plt.plot(autos[:,i]/autos[0,i], c = c)

plt.legend(handles,labels,)

plt.xticks(np.arange(0,40,5),np.arange(0,40,5)*5)
plt.title('Auto-co-occurrence curves for all molecules, by SSAM-assigned cell types:')

# %%
figure(18,6)

genes_to_plot=['Airway fibroblast','Erythrocyte','Mesothelial']

plt.subplot(141)
plt.title('Auto co-occurrence')
plt.xticks(np.arange(0,40,5),np.arange(0,40,5)*5)

for i,c in enumerate(genes_to_plot):
    
    tissue_index = np.where(signatures.index==c)[0][0]
    color = tissue_colors[tissue_index]
    
    plt.subplot(1,4,1)
    plt.plot(autos[:,tissue_index]/autos[0,tissue_index],color=color)
    
    plt.subplot(1,4,i+2)
    sdata[sdata.celltype==c].scatter(color = color)
    plt.title(c)

# %%
figure(18,6)

genes_to_plot=['Airway fibroblast','Ciliated epithelial','Mesothelial']

plt.subplot(141)
plt.title('Co-occurrence with "other":')
plt.xticks(np.arange(0,40,5),np.arange(0,40,5)*5)

for i,c in enumerate(genes_to_plot):
    
    tissue_index = np.where(signatures.index==c)[0][0]
    color = tissue_colors[tissue_index]
    
    plt.subplot(1,4,1)
    plt.plot(cooc[-1,tissue_index]/autos[0,tissue_index],color=color)
    
    plt.subplot(1,4,i+2)
    sdata[sdata.celltype==c].scatter(color = color)
    plt.title(c)

# %%

