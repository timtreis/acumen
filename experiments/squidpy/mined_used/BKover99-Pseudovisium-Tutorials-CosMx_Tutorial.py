# mined from: https://github.com/BKover99/Pseudovisium/blob/518a0ad415290dcc21e4840d501e60b88354b296/Tutorials/CosMx_Tutorial.ipynb
# symbols: squidpy.pl.spatial_scatter, squidpy.read.nanostring, squidpy.read.visium

# %%
# !wget https://dg1oqa.bl.files.1drv.com/y4m2CulsqPOHi88uM3y4R3bpe_E0oV-5QWcibRtQAa6_OEwaeDJp2szwSfQLHeZ9hukrY97lU2Mzzghwvm77jm3jqzjsYHWWmwcn-_lRMn7e1R1tOBrFkHba5uaoOv2yQcDobXlqrpmChbiHihhY_4uT2L3ayWbLGj-OBQIIulTUqBSMDshOXZmZhfGQfa3i3GHgzgYf6-2z0sWlKjMlQhv-w
# !unzip /content/y4m2CulsqPOHi88uM3y4R3bpe_E0oV-5QWcibRtQAa6_OEwaeDJp2szwSfQLHeZ9hukrY97lU2Mzzghwvm77jm3jqzjsYHWWmwcn-_lRMn7e1R1tOBrFkHba5uaoOv2yQcDobXlqrpmChbiHihhY_4uT2L3ayWbLGj-OBQIIulTUqBSMDshOXZmZhfGQfa3i3GHgzgYf6-2z0sWlKjMlQhv-w
# !pip install pympler
# !pip install Pseudovisium

# %%
import time
from Pseudovisium.pseudovisium_generate import generate_pv
from pympler import asizeof
# !head "/content/pancreas/Pancreas_tx_file.csv"

# %%
csv_file="/content/pancreas/Pancreas_tx_file.csv"
output_path = "/content"
hexagon_size = 25 #change to 50 for faster execution. Any higher would
#be inappropriate for a very small tissue section like this
start = time.time()
pseudovisium_path = generate_pv(csv_file=csv_file,
                                hexagon_size=hexagon_size,
                                output_path=output_path,
                                batch_size=2000000,
                                technology="CosMx",
                                max_workers=10,
                                project_name='cosmx_pancreas',
                                coord_to_um_conversion=0.12028 #this is the
                                #pxl to um conversion factor used
                                #in this data,but may
                                #differ for other cosmx dataset

                                )
end = time.time()

# %%
from Pseudovisium.pseudovisium_qc import generate_qc_report
folders= ["/content/pseudovisium/cosmx_pancreas/"]
output_folder="/content/"
gene_names=["Ace2"]
generate_qc_report(folders,
                   output_folder,
                   gene_names,
                   include_morans_i=False,
                   max_workers=10,
                   normalisation=True,
                   save_plots=True)

# %%
import squidpy as sq
import numpy as np
adata_pv = sq.read.visium("/content/pseudovisium/cosmx_pancreas/", library_id="library_id")
adata_pv.obs["sum"]= np.array(np.sum(adata_pv.X,axis=1)).flatten()
sq.pl.spatial_scatter(
    adata_pv,color="sum",img=False
)

# %%
sq.pl.spatial_scatter(
    adata_pv,color="INS",img=False
)

# %%
import pandas as pd
def load_in_fullres(folder,ctu=0.12028):
      try:
          adata_fullres = sq.read.nanostring(
              path=folder,
              counts_file="Pancreas_exprMat_file.csv",
              meta_file="Pancreas_metadata_file.csv",
              fov_file="Pancreas_fov_positions_file.csv",
          )


      except:
          #first command -> locate to folder folder
          file= "/Pancreas_fov_positions_file.csv"
          new_file = "/Pancreas_fov_positions_new_file.csv"
          # extract the third column from the CSV file
          df = pd.read_csv(folder + file)
          #create a column fov which is same as FOV
          df["fov"]=df["FOV"]
          #make it the index
          df.set_index("fov", inplace=True)

          df.to_csv(folder + new_file, index=True)

          adata_fullres = sq.read.nanostring(
                  path=folder,
                  counts_file="Pancreas_exprMat_file.csv",
                  meta_file="Pancreas_metadata_file.csv",
                  fov_file="Pancreas_fov_positions_new_file.csv",
              )
          adata_fullres.obsm["spatial"] = adata_fullres.obs[["CenterX_global_px","CenterY_global_px"]].to_numpy()
          adata_fullres.obsm["spatial"]=adata_fullres.obsm["spatial"]*ctu

      sc.pp.calculate_qc_metrics(adata_fullres, percent_top=(50, 100, 200, 300), inplace=True)
      adata_fullres.obs["sum"]= np.array(np.sum(adata_fullres.X,axis=1)).flatten()
      return adata_fullres

# %%
import scanpy as sc
adata_fullres = load_in_fullres("/content/pancreas",ctu=0.12028)

# %%
import matplotlib.pyplot as plt
plt.scatter(adata_fullres.obsm["spatial"][:,0],adata_fullres.obsm["spatial"][:,1],c=adata_fullres.obs["sum"],s=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %%
#plot Insulin gene
plt.scatter(adata_fullres.obsm["spatial"][:,0],adata_fullres.obsm["spatial"][:,1],c=adata_fullres[:,"INS"].X.toarray().flatten(),cmap="viridis")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %%
print("Total size of the object:", asizeof.asizeof(adata_fullres)/ (1024 * 1024), "Mb")

# %%
print("Total size of the object:", asizeof.asizeof(adata_pv)/ (1024 * 1024), "Mb")

# %%
import subprocess

# Run the pip freeze command to get a list of installed packages
output = subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip().split('\n')

print("Installed packages:")
for package in output:
    print(package)

# %%

