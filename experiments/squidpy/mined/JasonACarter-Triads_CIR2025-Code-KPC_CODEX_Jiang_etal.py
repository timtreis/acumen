# mined from: https://github.com/JasonACarter/Triads_CIR2025/blob/7d2e9cdf838442e443905baf989b1d24779fd6c5/Code/KPC_CODEX_Jiang_etal.ipynb
# symbols: squidpy.tl.var_by_distance

# %%
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq

import numpy as np
import pandas as pd
from scipy import stats 
import squidpy as sq
import os
import anndata
import glob
import scanpy.external as sce
import anndata as ad

# %%
adata = ad.read_h5ad('../Data/KPC CODEX/KPC_CODEX_Jiang.h5ad')

# %%
adata

# %%
sample,count=np.unique(adata.obs.Slide,return_counts=1)
pd.DataFrame(np.vstack((sample,count)).T,columns=['Sample','Cell count'])

# %%
sc.set_figure_params(dpi_save=600,frameon=False,transparent=True,figsize=(7,5))
sc.pl.matrixplot(adata,groupby='Celltype',
               var_names=['PanCk','CD44','SMA','CD31',
                          'CD45','CD11b','CD169',
                          'CD11c','MHCII',
                          'CD3','CD4','FoxP3','CD8a','CD19'],
              categories_order=['Epithelial','Fibroblast','Endothelial','Macrophage','DC','Th','Treg','CTL','B cell'],
              standard_scale='group',use_raw=False,vmax=1,swap_axes=True)

# %%
sc.set_figure_params(dpi_save=600,frameon=False,transparent=True,figsize=(7,5))
celltype_dict={
    'Epithelial':'dimgrey',
    'Endothelial':'blueviolet',
    'Fibroblast':'#17becf',
    'B cells':'#ffbb78',
    'Plasma cells':'#ff7f0e',
    'Plasma cell':'#ff7f0e',
    'Th':'Blue',
    'Treg':'Dodgerblue',
    'CTL':'Red',
    'DC':'Green',
    'Macrophage':'Gold',
    'Cholangiocyte':'lightgray',
    'B cell':'Orange'
}

df = adata.obs[['Celltype', 'Slide']]
counts = df.groupby(['Slide', 'Celltype']).size().reset_index(name='count')

total_counts = df.groupby('Slide').size().reset_index(name='total')
counts = counts.merge(total_counts, on='Slide')
counts['proportion'] = counts['count'] / counts['total']
counts['proportion'] = counts['proportion']+0.001
order=counts.groupby('Celltype')['proportion'].mean().sort_values(ascending=0).index

plt.figure(figsize=(4,5.5))
sns.set_style('white')
sns.set_style('ticks')
sns.boxplot(data=counts, y='Celltype', x='proportion',order=order,palette=celltype_dict,showfliers=False)
sns.stripplot(data=counts, y='Celltype', x='proportion',order=order,color='k')
plt.xlim([0.001,1])
plt.ylabel('')
plt.xlabel('')
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xscale('log')
sns.despine()
plt.tight_layout()
plt.show()
plt.close()

# %%
threshold=20 #adata.obsm['spatial'] given in microns

df1=pd.DataFrame(np.empty((adata.obs.Slide.unique().shape[0],4)),
               index=adata.obs.Slide.unique(),
               columns=['Isolated','CD4','CD8','Triad']).T

for x in adata.obs.Slide.unique():
    adata_temp=adata[adata.obs.Slide==x]

    sq.tl.var_by_distance(
        adata=adata_temp,
        groups=['DC','Th','CTL'],
        cluster_key="Celltype"
    )
    
    df=adata_temp.obsm['design_matrix']
    isolated=df[(df.Celltype=='DC') & (df['Th_raw']>threshold) & (df['CTL_raw']>threshold)].shape[0]/df[(df.Celltype=='DC')].shape[0],
    cd4=df[(df.Celltype=='DC') & (df['Th_raw']<=threshold) & (df['CTL_raw']>threshold)].shape[0]/df[(df.Celltype=='DC')].shape[0],
    cd8=df[(df.Celltype=='DC') & (df['Th_raw']>threshold) & (df['CTL_raw']<=threshold)].shape[0]/df[(df.Celltype=='DC')].shape[0],
    triad=df[(df.Celltype=='DC') & (df['Th_raw']<=threshold) & (df['CTL_raw']<=threshold)].shape[0]/df[(df.Celltype=='DC')].shape[0]
    order=['Isolated','CD4','CD8','Triad']
    df1.loc[order,x]=np.hstack((isolated[0],cd4[0],cd8[0],triad))
df1['Celltype']=df1.index
df1=df1.melt(id_vars='Celltype')

sns.set_style('white')
sns.set_style('ticks')
plt.figure(figsize=(3.25,5.5))
sns.boxplot(y='Celltype',x='value',data=df1,zorder=0,color='white',
           order=order,palette=['Green','royalblue','#FF0800','darkorchid'],saturation=0.8,showfliers=False)
sns.scatterplot(y='Celltype',x='value',data=df1,color='k',zorder=1,s=50)
plt.legend('',frameon=False)
sns.despine()
plt.yticks(fontsize=20)
plt.ylabel('')
plt.xlabel('')
plt.xscale('log')
plt.xticks([0.1,1],fontsize=21)
plt.xlim([0.05,1])
plt.tight_layout()
plt.show()
plt.close()

# %%

