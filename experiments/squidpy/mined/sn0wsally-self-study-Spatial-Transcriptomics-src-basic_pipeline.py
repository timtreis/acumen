# mined from: https://github.com/sn0wsally/self-study-Spatial-Transcriptomics/blob/626345e6c034a51e2014bb684a2eccab9a03a93b/src/basic_pipeline.ipynb
# symbols: squidpy.datasets.visium, squidpy.im.ImageContainer

# %%
# !pip install scanpy
# !pip install squidpy

# %%
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision.models as models
import squidpy as sq

from tqdm import tqdm

# %%
class STDataset(Dataset):
    def __init__(
            self, adata, image_container,
            patch_size=224, transform=None, gene_transform=None
            ):
        self.adata = adata
        self.img = image_container
        self.patch_size = patch_size
        self.transform = transform
        self.gene_transform = gene_transform

        library_id = list(adata.uns["spatial"].keys())[0]
        scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
        self.coords = adata.obsm["spatial"] * scale

        # gene expression
        if hasattr(adata.X, "toarray"):
            self.genes = adata.X.toarray()
        else:
            self.genes = adata.X

    def __len__(self):
        return len(self.coords)

    def _extract_patch(self, x, y):
        half = self.patch_size // 2

        img = np.asarray(self.img["image"].values).squeeze()

        h, w, _ = img.shape

        x = int(np.clip(x, half, w - half))
        y = int(np.clip(y, half, h - half))

        patch = img[
            y - half : y + half,
            x - half : x + half
        ]

        return patch

    def __getitem__(self, idx):
        x, y = self.coords[idx]

        patch = self._extract_patch(x, y)
        # print("DEBUG patch:", patch.shape)
        patch = torch.tensor(patch).permute(2, 0, 1).float() / 255.0

        if self.transform:
            patch = self.transform(patch)

        gene = torch.tensor(self.genes[idx]).float()

        if self.gene_transform:
            gene = self.gene_transform(gene)

        coord = torch.tensor([x, y]).float()

        return {
            "image": patch,
            "gene": gene,
            "coord": coord
        }

# %%
### ----- Base CNN Encoder ----- ###
class ImageEncoder(nn.Module):
    """
    CNN backbone for histology feature extraction
    """
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        if backbone == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            out_dim = 512

        elif backbone == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            out_dim = 512

        elif backbone == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            out_dim = 2048

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # remove classifier
        model.fc = nn.Identity()

        self.encoder = model
        self.out_dim = out_dim

    def forward(self, x):
        return self.encoder(x)


### ----- Gene Prediction Head ----- ###
class GeneHead(nn.Module):
    """
    MLP head for gene prediction
    """
    def __init__(self, in_dim, num_genes):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, num_genes)
        )

    def forward(self, x):
        return self.mlp(x)


### ----- Full Model ----- ###
class STModel(nn.Module):
    """
    Spatial Transcriptomics Model
    Image -> Gene Expression
    """
    def __init__(
        self,
        num_genes,
        backbone="resnet18",
        pretrained=True
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained
        )

        self.head = GeneHead(
            in_dim=self.encoder.out_dim,
            num_genes=num_genes
        )

    def forward(self, x):
        features = self.encoder(x)
        genes = self.head(features)
        return genes

# %%
### ----- Seed ----- ###
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


### ----- Device ----- ###
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


### ----- Model ----- ###
def get_model(num_genes, device):
    model = STModel(num_genes=num_genes)
    return model.to(device)


### ----- Loss ----- ###
def get_loss():
    return nn.MSELoss()


### ----- Optimizer ----- ###
def get_optimizer(model, lr=1e-4, wd=1e-5):
    return AdamW(model.parameters(), lr=lr, weight_decay=wd)


### ----- Scheduler ----- ###
def get_scheduler(optimizer):
    return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


### ----- Train ----- ###
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        img = batch["image"].to(device)
        gene = batch["gene"].to(device)

        pred = model(img)
        loss = criterion(pred, gene)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


### ----- Validation ----- ###
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            gene = batch["gene"].to(device)

            pred = model(img)
            loss = criterion(pred, gene)

            total_loss += loss.item() * img.size(0)

    return total_loss / len(loader.dataset)

# %%
set_seed()
device = get_device()
print("Device:", device)

batch_size = 2
epochs = 5
patch_size = 224

# Load 10x Visium dataset
library_id = "V1_Breast_Cancer_Block_A_Section_1"
adata = sq.datasets.visium(sample_id=library_id)
adata.var_names_make_unique()

hires = adata.uns["spatial"][library_id]["images"]["hires"]
img = sq.im.ImageContainer(hires)

dataset = STDataset(
    adata=adata,
    image_container=img,
    patch_size=patch_size
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = STModel(num_genes=dataset.genes.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")

    train_loss = train_one_epoch(
        model, loader, optimizer, criterion, device
    )

    val_loss = validate(
        model, loader, criterion, device
    )

    scheduler.step()

    print(f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")
