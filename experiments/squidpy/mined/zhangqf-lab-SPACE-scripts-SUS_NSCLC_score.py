# mined from: https://github.com/zhangqf-lab/SPACE/blob/6fc15f9eb84267e8eaa2f2a2c34f1508a0d5db09/scripts/SUS_NSCLC_score.ipynb
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
import scipy.stats as stats
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

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
adata=sc.read('/home/liyuzhe/Notebook/SPACE_rebuttal/NSCLC/NSCLC_cc_adata.h5ad')
adata

# %%
len(adata.obs.celltype.value_counts())

# %%
df=pd.DataFrame(index=adata.obs.cell_comm.cat.categories,columns=['SES'])
for i in adata.obs.cell_comm.cat.categories:
    adata_=adata[adata.obs.cell_comm==i].copy()
    sq.gr.spatial_neighbors(adata_, n_neighs=20,coord_type="generic")
    adj=adata_.obsp['spatial_connectivities'].toarray()
    df_=pd.DataFrame(index=adata_.obs.index,columns=adata_.obs.celltype.cat.categories)
    for j in range(len(adj)):
        tmp=adata_[adj[j].nonzero()]
        df_tmp=tmp.obs.celltype.value_counts(normalize=True)
        df_.iloc[j][df_tmp.index]=df_tmp
    df_=df_.fillna(0)
    df_ref=pd.DataFrame(adata_.obs.celltype.value_counts(normalize=True))
    df_ref=df_ref.loc[df_.columns].copy()
    df_cor=pd.DataFrame(cosine_similarity(df_,df_ref.transpose()),index=df_.index,columns=['SES'])
    score=df_cor.mean().values[0]
    df.loc[i,'SES']=score

# %%
adata_=sc.read('/home/liyuzhe/Notebook/SPACE_rebuttal/NSCLC/NSCLC_cc_adata.h5ad')
sq.gr.spatial_neighbors(adata_, n_neighs=20,coord_type="generic")
adj=adata_.obsp['spatial_connectivities'].toarray()
df_=pd.DataFrame(index=adata_.obs.index,columns=adata_.obs.celltype.cat.categories)
for j in range(len(adj)):
    tmp=adata_[adj[j].nonzero()]
    df_tmp=tmp.obs.celltype.value_counts(normalize=True)
    df_.iloc[j][df_tmp.index]=df_tmp
df_=df_.fillna(0)
df_ref=pd.DataFrame(adata_.obs.celltype.value_counts(normalize=True))
df_ref=df_ref.loc[df_.columns].copy()
df_cor=pd.DataFrame(cosine_similarity(df_,df_ref.transpose()),index=df_.index,columns=['SES'])
score=df_cor.mean().values[0]
df.loc['all','SES']=score

# %%
df['group']=df.index.copy()
df=df.reset_index()
df

# %%
df.to_csv('results/NSCLC_SES_uniform_score.csv')

# %%
plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(5,4))  

sns.barplot(df,x='group',y='SES',width=0.75,
            palette={'CC1': '#1f77b4','CC2': '#ff7f0e','CC3': '#279e68',
                     'CC4': '#d62728','CC5': '#aa40fc','CC6': '#8c564b',
                     'all':'grey'})
ax=plt.gca()
ax.set_ylabel('SES')
ax.set_xlabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('figures/NSCLC_SES_uniform_score.pdf', bbox_inches = 'tight')
plt.show()

# %%


# %%

