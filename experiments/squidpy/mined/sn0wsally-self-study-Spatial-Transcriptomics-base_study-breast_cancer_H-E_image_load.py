# mined from: https://github.com/sn0wsally/self-study-Spatial-Transcriptomics/blob/626345e6c034a51e2014bb684a2eccab9a03a93b/base_study/breast_cancer_H&E_image_load.ipynb
# symbols: squidpy.im.ImageContainer

# %%
# !pip install scanpy
# !pip install squidpy

# %%
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

# %%
# 1. 10x Genomics 유방암 ST 데이터 로드
print("🚀 데이터 로드 중... (시간이 조금 걸릴 수 있습니다)")
adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
adata.var_names_make_unique()
library_id = "V1_Breast_Cancer_Block_A_Section_1"

# %%
# 2. H&E 이미지 데이터 추출
# 공간 전사체 데이터에서 실제 이미지는 adata.uns 딕셔너리 깊은 곳에 저장되어 있습니다.
hires_image = adata.uns["spatial"][library_id]["images"]["hires"]
print(f"원본 고해상도 이미지 배열 형태: {hires_image.shape}")

# %%
# 3. Squidpy ImageContainer로 변환 (딥러닝 전처리 및 패치 추출용)
# 단순 Numpy 배열을 넘어, 좌표 기반 자르기(Crop)를 지원하는 객체로 만듭니다.
img = sq.im.ImageContainer(hires_image)

# %%
# 4. H&E 원본 이미지 시각화
fig, ax = plt.subplots(figsize=(8, 8))
img.show(layer="image", ax=ax)
plt.title("H&E Stained Tissue Image (High Resolution)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

print(f"✅ ImageContainer 변환 완료: {img.shape}")
