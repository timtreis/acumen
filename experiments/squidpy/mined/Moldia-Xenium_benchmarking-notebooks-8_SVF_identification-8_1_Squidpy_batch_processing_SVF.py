# mined from: https://github.com/Moldia/Xenium_benchmarking/blob/50249f7d03d9cfc1453c4b81ef3b1053c4d23052/notebooks/8_SVF_identification/8_1_Squidpy_batch_processing_SVF.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import squidpy as sq
import scanpy as sc
import os

# %%
maindir='../../data/unprocessed_adata/'
output_dir='../../data/unprocessed_adata_nuclei/'
files=os.listdir(maindir)
files=['ms_brain_multisection1.h5ad','human_brain.h5ad', 'ms_brain_multisection2.h5ad','ms_brain_multisection3.h5ad',
'realmouse_1.h5ad','realmouse_2.h5ad','realmouse_3.h5ad','realmouse_4.h5ad','hbreast_ilc_addon_set2.h5ad',
 'hbreast_ilc_addon_set4.h5ad', 'hbreast_ilc_entiresample_set3.h5ad', 'healthy_lung.h5ad',
 'human_alzheimers.h5ad', 'human_gbm.h5ad', 'human_spinal_chord_active.h5ad', 'human_spinal_chord_inactive.h5ad',
 'h_breast_1.h5ad', 'h_breast_2.h5ad', 'lung_cancer.h5ad', 'ms_brain_fullcoronal.h5ad', 'ms_brain_partialcoronal.h5ad' ]

# %%
sq.gr.spatial_autocorr(adata1, mode="geary")
hs_results=adata1.uns['gearyC']
hs_results['rank']=list(hs_results['C'].rank(ascending=False))
sq.gr.spatial_autocorr(adata1, mode="geary")
hs_results=adata1.uns['gearyC']
hs_results['rank']=list(hs_results['C'].rank())
hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
hs_results.columns=['Pval','FDR','rank']

# %%
for f in files[:]:
    try:
        print(f)
        adata1=sc.read(output_dir+f)
        adata1.obsm["spatial"]=np.array([adata1.obs.x_centroid,adata1.obs.y_centroid]).transpose().astype('float64')
        adata1.obsm['spatial']=np.array(adata1.obs.loc[:,['x_centroid','y_centroid']])
        adata1=sc.AnnData(adata1.X,obs=adata1.obs,var=adata1.var,obsm=adata1.obsm)
        adata1.layers['raw']=adata1.X
        sq.gr.spatial_neighbors(adata1,radius=50.0,coord_type ='generic')
        sq.gr.spatial_autocorr(adata1, mode="moran")
        hs_results=adata1.uns["moranI"]
        hs_results['rank']=list(hs_results['I'].rank())
        hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
        hs_results.columns=['Pval','FDR','rank']
        hs_results.to_csv('../../figures/SVF/'+str(f.split('.')[0])+'__squidpy_MoranI.csv')
        sq.gr.spatial_autocorr(adata1, mode="geary")
        hs_results=adata1.uns['gearyC']
        hs_results['rank']=list(hs_results['C'].rank(ascending=False))
        hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
        hs_results.columns=['Pval','FDR','rank']
        hs_results.to_csv('../../figures/SVF/'+str(f.split('.')[0])+'__squidpy_gearyC.csv')
        sc.pp.highly_variable_genes(adata1)
        adata1.var['rank']=adata1.var['dispersions'].rank()
        hvg=adata1.var
        hvg=hvg.loc[:,['highly_variable','dispersions_norm','rank']]
        hvg.to_csv('../../figures/SVF/'+str(f.split('.')[0])+'__highly_variable_scanpy.csv')
    except:
        print('not possible')

# %%
for f in files[:]:
    try:
        print(f)
        adata1=sc.read(output_dir+f)
        adata1.obsm["spatial"]=np.array([adata1.obs.x_centroid,adata1.obs.y_centroid]).transpose().astype('float64')
        adata1.obsm['spatial']=np.array(adata1.obs.loc[:,['x_centroid','y_centroid']])
        adata1=sc.AnnData(adata1.X,obs=adata1.obs,var=adata1.var,obsm=adata1.obsm)
        adata1.layers['raw']=adata1.X
        import random
        some=random.sample(range(0,adata1.shape[0]),5000)
        adata1=adata1[some,:]
        sq.gr.spatial_neighbors(adata1,radius=50.0,coord_type ='generic')
        sq.gr.spatial_autocorr(adata1, mode="moran")
        hs_results=adata1.uns["moranI"]
        hs_results['rank']=list(hs_results['I'].rank())
        hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
        hs_results.columns=['Pval','FDR','rank']
        hs_results.to_csv('../../figures/SVF/'+str(f.split('.')[0])+'__squidpy5000_MoranI.csv')
        sq.gr.spatial_autocorr(adata1, mode="geary")
        hs_results=adata1.uns['gearyC']
        hs_results['rank']=list(hs_results['C'].rank(ascending=False))
        hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
        hs_results.columns=['Pval','FDR','rank']
        hs_results.to_csv('../../figures/SVF/'+str(f.split('.')[0])+'__squidpy5000_gearyC.csv')
        sc.pp.highly_variable_genes(adata1)
        adata1.var['rank']=adata1.var['dispersions'].rank()
        hvg=adata1.var
        hvg=hvg.loc[:,['highly_variable','dispersions_norm','rank']]
        hvg.to_csv('../../figures/SVF/'+str(f.split('.')[0])+'__highly_variable5000_scanpy.csv')
    except:
        print('not possible')

# %%
import time
f='ms_brain_multisection1.h5ad'
cells=[500,1000,5000,10000,50000,100000]

times_hvg=[]
times_I=[]
times_G=[]
for cel in cells:
    print(f)
    start_time = time.time()
    adata1=sc.read(output_dir+f)
    adata1.obsm["spatial"]=np.array([adata1.obs.x_centroid,adata1.obs.y_centroid]).transpose().astype('float64')
    adata1.obsm['spatial']=np.array(adata1.obs.loc[:,['x_centroid','y_centroid']])
    adata1=sc.AnnData(adata1.X,obs=adata1.obs,var=adata1.var,obsm=adata1.obsm)
    adata1.layers['raw']=adata1.X
    import random
    some=random.sample(range(0,adata1.shape[0]),int(cel))
    adata1=adata1[some,:]
    sq.gr.spatial_neighbors(adata1,radius=50.0,coord_type ='generic')
    sq.gr.spatial_autocorr(adata1, mode="moran")
    hs_results=adata1.uns["moranI"]
    hs_results['rank']=list(hs_results['I'].rank())
    hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
    hs_results.columns=['Pval','FDR','rank']
    end_time = time.time()
    times_I.append(end_time - start_time)
    
    ###gearyc
    print(f)
    start_time = time.time()
    adata1=sc.read(output_dir+f)
    adata1.obsm["spatial"]=np.array([adata1.obs.x_centroid,adata1.obs.y_centroid]).transpose().astype('float64')
    adata1.obsm['spatial']=np.array(adata1.obs.loc[:,['x_centroid','y_centroid']])
    adata1=sc.AnnData(adata1.X,obs=adata1.obs,var=adata1.var,obsm=adata1.obsm)
    adata1.layers['raw']=adata1.X
    import random
    some=random.sample(range(0,adata1.shape[0]),int(cel))
    adata1=adata1[some,:]
    sq.gr.spatial_neighbors(adata1,radius=50.0,coord_type ='generic')
    sq.gr.spatial_autocorr(adata1, mode="geary")
    hs_results=adata1.uns['gearyC']
    hs_results['rank']=list(hs_results['C'].rank(ascending=False))
    hs_results=hs_results.loc[:,['pval_norm','pval_norm_fdr_bh','rank']]
    hs_results.columns=['Pval','FDR','rank']
    end_time = time.time()
    times_G.append(end_time - start_time)
    
    # hvg
    print(f)
    start_time = time.time()
    adata1=sc.read(output_dir+f)
    adata1.obsm["spatial"]=np.array([adata1.obs.x_centroid,adata1.obs.y_centroid]).transpose().astype('float64')
    adata1.obsm['spatial']=np.array(adata1.obs.loc[:,['x_centroid','y_centroid']])
    adata1=sc.AnnData(adata1.X,obs=adata1.obs,var=adata1.var,obsm=adata1.obsm)
    adata1.layers['raw']=adata1.X
    import random
    some=random.sample(range(0,adata1.shape[0]),int(cel))
    adata1=adata1[some,:]
    sc.pp.highly_variable_genes(adata1)
    adata1.var['rank']=adata1.var['dispersions'].rank()
    hvg=adata1.var
    hvg=hvg.loc[:,['highly_variable','dispersions_norm','rank']]
    end_time = time.time()
    times_hvg.append(end_time - start_time)
    print(times_hvg)

# %%
timesres=pd.DataFrame([cells,times_hvg,times_I,times_G],index=['cells','times_hvg','times_I','times_G']).transpose()

# %%
plt.plot(timesres['cells'],timesres['times_hvg'])
plt.plot(timesres['cells'],timesres['times_I'])
plt.plot(timesres['cells'],timesres['times_G'])

# %%
timesres.to_csv('../../figures/times_svf/squidpy_times.csv')
