# mined from: https://github.com/zhangqf-lab/SPACE/blob/6fc15f9eb84267e8eaa2f2a2c34f1508a0d5db09/scripts/SES_slice153.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import space as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import colors
from termcolor import colored

# %%
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

# %%
plt.rcParams['axes.unicode_minus']=False
plt.rc('font', family='Helvetica')
plt.rcParams['pdf.fonttype'] = 42
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.set_figure_params(dpi=120,facecolor='w',frameon=True,figsize=(4,4)) 
# %config InlineBackend.figure_format='retina'
# %matplotlib inline

# %%
featurePlotCols=["lightgrey","whitesmoke","#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026","#800026"]
mymap2 = colors.LinearSegmentedColormap.from_list('my_colormap',featurePlotCols,N=512)

# %%
sp.__version__

# %%
adata=sc.read('/data1/liyuzhe/data/SPACE_revision/slice153-1_cc_adata.h5ad')
adata

# %%
sq.gr.spatial_neighbors(adata, n_neighs=20,coord_type="generic")
adj=adata.obsp['spatial_connectivities'].toarray()
df=pd.DataFrame(index=adata.obs.index,columns=adata.obs.subclass_preprocessed.cat.categories)
for i in range(len(adj)):
    tmp=adata[adj[i].nonzero()]
    df_tmp=tmp.obs.subclass_preprocessed.value_counts(normalize=False)
    df.iloc[i][df_tmp.index]=df_tmp
df=df.fillna(0)
df_sum=df.sum(axis=1)
df=df.div(df_sum,axis=0)

# %%
meta=adata.obs.copy()

# %%
meta.Cell_Communities=meta.cell_comm.astype('category')

# %%
df_cor=pd.DataFrame(cosine_similarity(df),index=df.index,columns=df.index)

# %%
df1=pd.DataFrame(index=df_cor.index,columns=['corr','group1','group2']) # celltype:0 ; community:0
df2=pd.DataFrame(index=df_cor.index,columns=['corr','group1','group2']) # celltype:0 ; community:1
df3=pd.DataFrame(index=df_cor.index,columns=['corr','group1','group2']) # celltype:1 ; community:0
df4=pd.DataFrame(index=df_cor.index,columns=['corr','group1','group2']) # celltype:1 ; community:1
df1['group1']='g1'
df2['group1']='g2'
df3['group1']='g3'
df4['group1']='g4'
df1['group2']=df1.index.copy()
df2['group2']=df2.index.copy()
df3['group2']=df3.index.copy()
df4['group2']=df4.index.copy()

# %%
for i in df.index:
    cell_com=meta.loc[i,'Cell_Communities']
    celltype=meta.loc[i,'subclass_preprocessed']
    idx1=meta[(meta.Cell_Communities!=cell_com)&(meta.subclass_preprocessed!=celltype)].index.copy()
    idx2=meta[(meta.Cell_Communities!=cell_com)&(meta.subclass_preprocessed==celltype)].index.copy()
    idx3=meta[(meta.Cell_Communities==cell_com)&(meta.subclass_preprocessed!=celltype)].index.copy()
    idx4=meta[(meta.Cell_Communities==cell_com)&(meta.subclass_preprocessed==celltype)].index.copy()
    
    if len(idx1)>0:
        df1.loc[i,'corr']=(df_cor.loc[i,idx1].sum())/(len(idx1))
    else:
        break
    
    if len(idx2)>0:
        df2.loc[i,'corr']=(df_cor.loc[i,idx2].sum())/(len(idx2))
    else:
        break
   
    if len(idx3)>0:
        df3.loc[i,'corr']=(df_cor.loc[i,idx3].sum())/(len(idx3))
    else:
        break
   
    if len(idx4)>1:
        df4.loc[i,'corr']=(df_cor.loc[i,idx4].sum()-1)/(len(idx4)-1)
    else:
        df4.loc[i,'corr']=1

# %%
df1['cell_com']=meta['Cell_Communities'].copy()
df2['cell_com']=meta['Cell_Communities'].copy()
df3['cell_com']=meta['Cell_Communities'].copy()
df4['cell_com']=meta['Cell_Communities'].copy()
df1['celltype']=meta['subclass_preprocessed'].copy()
df2['celltype']=meta['subclass_preprocessed'].copy()
df3['celltype']=meta['subclass_preprocessed'].copy()
df4['celltype']=meta['subclass_preprocessed'].copy()

# %%
df_all=pd.concat([df1,df2,df3,df4])

# %%
sns.boxplot(data=df_all, x="group1", y="corr")

# %%

