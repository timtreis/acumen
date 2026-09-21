# mined from: https://github.com/gkanfer/SpatialOmicsToolkit/blob/a41fd656300d917c1b94af94afead5b9da63eec5/notebooks/Cancer_cell_8april_2024_.ipynb
# symbols: squidpy.im.ImageContainer

# %%
import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
import numpy as np

# %%
path = '/data/kanferg/Sptial_Omics/playGround/Data/GSE247629_RAW'
os.listdir('/data/kanferg/Sptial_Omics/playGround/Data/GSE247629_RAW')

# %%
andatap1 = sc.read_10x_mtx(path = path,prefix = 'GSM7898157_P1_')
andatap2 = sc.read_10x_mtx(path = path,prefix = 'GSM7898158_P2_')
andatap3 = sc.read_10x_mtx(path = path,prefix = 'GSM7898159_P3_')
andatap4 = sc.read_10x_mtx(path = path,prefix = 'GSM7898160_P4_')

# %%
def print_matrix(andata):
    print(f'spots: {andata.n_obs} genes: {andata.n_vars}')
print_matrix(andatap1)
print_matrix(andatap2)
print_matrix(andatap3)
print_matrix(andatap4)

# %%
def read_barcode_features(path,bc_file,ft_file):
    bc = os.path.join(path,bc_file)
    ft = os.path.join(path,ft_file)
    def unzip_read(file_path):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
        return df
    def report_df(df,text_table_type):
        print(f'{str(text_table_type)} Rows number: {len(df)} Columns number {len(df.columns)}')
    df_bc = unzip_read(bc)
    report_df(df_bc,"barcode")
    df_ft = unzip_read(ft)
    report_df(df_ft,"features")
    return df_bc, df_ft

path = path
bc_file = 'GSM7898157_P1_barcodes.tsv.gz'
ft_file = 'GSM7898159_P3_features.tsv.gz'
df_bc, df_ft = read_barcode_features(path,bc_file,ft_file)

# %%
from scipy.io import mmread

# %%
matrix_path = os.path.join(path,'GSM7898157_P1_matrix.mtx.gz')
matrix = mmread(matrix_path)
dense_matrix = matrix.todense()
df = pd.DataFrame(dense_matrix)
df

# %%
path = '/data/kanferg/Sptial_Omics/playGround/Data/GSE263303_RAW'
os.listdir('/data/kanferg/Sptial_Omics/playGround/Data/GSE263303_RAW')

# %%
def read_barcode_features(path,bc_file,ft_file):
    bc = os.path.join(path,bc_file)
    ft = os.path.join(path,ft_file)
    def unzip_read(file_path):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
        return df
    def report_df(df,text_table_type):
        print(f'{str(text_table_type)} Rows number: {len(df)} Columns number {len(df.columns)}')
    df_bc = unzip_read(bc)
    report_df(df_bc,"barcode")
    df_ft = unzip_read(ft)
    report_df(df_ft,"features")
    return df_bc, df_ft

path = path
bc_file = 'GSM8189358_K75-1-FMFC-barcodes.tsv.gz'
ft_file = 'GSM8189359_K75-2-FMFC-features.tsv.gz'
df_bc, df_ft = read_barcode_features(path,bc_file,ft_file)

# %%
df_ft

# %%
andatap1 = sc.read_10x_mtx(path = path,prefix = 'GSM8189358_K75-1-FMFC-')

# %%
andatap1

# %%
path_034_C1d1 = '/data/kanferg/Sptial_Omics/playGround/Data/GSE225691_RAW/01_034_C1d1/outs'
path_039_C1d1 = '/data/kanferg/Sptial_Omics/playGround/Data/GSE225691_RAW/01_039_C1d1/outs'
path_034_C3d1 = '/data/kanferg/Sptial_Omics/playGround/Data/GSE225691_RAW/01_034_C3d1/outs'
path_039_C3d1 = '/data/kanferg/Sptial_Omics/playGround/Data/GSE225691_RAW/01_039_C3d1/outs'

# %%
import warnings
warnings.filterwarnings('ignore')
def read_nature_files(path):
    andata = sc.read_visium(path = path)
    print(f'spots: {andata.n_obs} genes: {andata.n_vars}')
    return andata
adata_034_c1d1 = read_nature_files(path_034_C1d1)
adata_039_C1d1 = read_nature_files(path_039_C1d1)
adata_034_C3d1 = read_nature_files(path_034_C3d1)
adata_039_C3d1 = read_nature_files(path_039_C3d1)

# %%
warnings.filterwarnings('default')

# %%
np.shape(adata_034_c1d1.obsm['spatial'])

# %%
adata_034_c1d1.obsm['spatial']

# %%
df = adata_034_c1d1.to_df()
row_sums = df.sum(axis=1)

# Plotting the histogram of the row sums
plt.figure(figsize=(10, 6))
plt.hist(row_sums, bins=50, alpha=0.75, color='blue')
plt.title('Histogram of Total Counts per Spot')
plt.xlabel('Total Counts')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# %%
np.shape(adata_034_c1d1.uns['spatial']['01_034_C1d1']['images']['hires'])

# %%
df["x"] = adata_034_c1d1.obsm['spatial'][:,0]
df["y"] = adata_034_c1d1.obsm['spatial'][:,1]
df.insert(0,"CellID",np.arange(len(df)))

# %%
df.to_csv("adata_034_c1d1.csv",index=False)

# %%
df.head()

# %%
img034_c1d1 = sq.im.ImageContainer(
    adata_034_c1d1.uns['spatial']['01_034_C1d1']['images']['hires'],
    scale=adata_034_c1d1.uns['spatial']['01_034_C1d1']['images']['hires']
)

# %%


# %%
adata_034_c1d1.uns['spatial']['01_034_C1d1']['images']['hires']

# %%
adata_034_c1d1.uns['spatial']['01_034_C1d1']

# %%


# %%

