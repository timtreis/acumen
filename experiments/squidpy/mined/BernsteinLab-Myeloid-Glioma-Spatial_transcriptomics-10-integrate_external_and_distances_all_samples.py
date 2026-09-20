# mined from: https://github.com/BernsteinLab/Myeloid-Glioma/blob/118e4ce8c41c6a1e121ea1f86f8445da8b74b719/Spatial_transcriptomics/10-integrate_external_and_distances_all_samples.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import os
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from scipy import linalg, sparse
from tqdm import tqdm
from itertools import compress

import warnings
warnings.filterwarnings("ignore")

# %%
adata_dir = 'adata_env'
samples = [f[5:-5] for f in os.listdir(adata_dir)] 

# %%
adata_output_dir = 'adata_integrated_all'
if not os.path.exists(adata_output_dir):
    os.mkdir(adata_output_dir)

# %%
# exclude IDH mutants
keep = ['IDHMutant' not in sample for sample in samples]
samples = list(compress(samples, keep))

# %%
# set max measure for spatial distances
ring_max = 40

# %%
# set path for external info
CNV_info_dir = '/Users/cpc45/Data/GBM/Henrik_spatial/CNV_info'
RCTD_info_dir = '/Users/cpc45/Data/GBM/Henrik_spatial/RCTD_csv'

# %%
adata = sc.read('adata_env.h5ad')
adata = adata[adata.obs['patient'].isin(samples)]

# %%
adatas = []
d = []
chromosomes = ['Chr7','Chr10']

first1 = True
first2 = True
for sample in tqdm(samples):
    _adata = sc.read(os.path.join(adata_dir,'adata%s.h5ad'%sample))
    sq.gr.spatial_neighbors(_adata, n_rings = ring_max)
    d.append(np.asarray(_adata.obsp['spatial_distances'].todense()))
    
    # find number to append to indices to match adata
    index_num_to_append = adata[adata.obs['patient']==sample,:].obs.index[0].split('-')[-1]
    
    # integrate cnv data
    try:
        CNV_file = os.path.join(CNV_info_dir,'cnv_%s.csv'%sample[3::])

        cnv = pd.read_csv(CNV_file, index_col=0)
        cnv = cnv.iloc[:,7:-8]
        
        cnv_original = cnv.copy()
        cnv.index = cnv.index + '-%s'%index_num_to_append
        index = list(set(cnv.index) & set(adata.obs.index))
        
        # all pixels
        if first1:
            for ch in chromosomes:
                adata.obs[ch] = cnv.loc[index,ch]
            first1 = False
        else:
            for ch in chromosomes:
                adata.obs.loc[index,ch] = cnv.loc[index,ch]
        
        # per sample
        for ch in chromosomes:
            _adata.obs[ch] = cnv_original.loc[:,ch]
            
    except FileNotFoundError:
        print('CNV file not found for %s'%sample)
        
    # integrate RCTD data
    try:
        RCTD_file = os.path.join(RCTD_info_dir,'RCTD_%s.csv'%sample)
        rctd = pd.read_csv(RCTD_file, index_col=0)
        
        # correct names of malignant programs
        malignant_dict = {'Malignant2':'NPC','Malignant3':'AC','Malignant4':'MES2','Malignant6':'MES1','Malignant7':'OPC'}
        for m in malignant_dict:
            rctd.columns = list(map(lambda x: x.replace(m, malignant_dict[m]), rctd.columns))
        #
        
        rctd.columns = ['rctd_%s'%col for col in rctd.columns]
        
        rctd_original = rctd.copy()
        rctd.index = rctd.index + '-%s'%index_num_to_append
        index = list(set(rctd.index) & set(adata.obs.index))
        
        # all pixels
        if first2:
            rctd_columns = rctd.columns
            first2 = False
            for r in rctd_columns:
                adata.obs[r] = rctd.loc[index,r]
        else:
            for r in rctd_columns:
                adata.obs.loc[index,r] = rctd.loc[index,r]
        
        # per sample
        for r in rctd_columns:
            _adata.obs[r] = rctd_original.loc[:,r]
            
        _adata.write(os.path.join(adata_output_dir, 'adata%s.h5ad'%sample))
        
    except FileNotFoundError:
        print('RCTD file not found for %s'%sample)
        

# %%
D = linalg.block_diag(*d)
adata.obsp['spatial_distances'] = sparse.csr_matrix(D)

# %%
adata.write('adata_integrated_with_normal.h5ad')

# %%

