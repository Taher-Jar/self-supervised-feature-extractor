import sys
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def compute_knn(backbone, data_loader_train, data_loader_val):
    """Get CLS embeddings and use KNN classifier on them.

    We load all embeddings in memory and use sklearn. Should
    be doable.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer whose head is just an identity
        mapping.

    data_loader_train, data_loader_val : torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any
        augmentations. Just casting to tensor and then normalizing.

    Returns
    -------
    val_accuracy : float
        Validation accuracy.
    """
    device = next(backbone.parameters()).device

    data_loaders = {
        "train": data_loader_train,
        "val": data_loader_val,
    }
    lists = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": [],
    }

    for name, data_loader in data_loaders.items():
        for imgs, y in tqdm(data_loader, desc=f"Collecting features {name}"):
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier()
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc


def main(path_dataset, path_backbone):

    path_dataset = Path(path_dataset)
    path_backbone = Path(path_backbone)

    backbone = torch.load(path_backbone)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    backbone.to(device)

    path_train_dataset = path_dataset / "train" if (path_dataset / "train").is_dir() else None
                
    if (path_dataset / "val").is_dir():   
        path_val_dataset = path_dataset / "val"
    elif (path_dataset / "validation").is_dir():
            path_val_dataset = path_dataset / "validation"
    else:
        path_val_dataset = None

    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(backbone.in_dimensions()[1:]),
        ]
    )

    try:
        dataset_train = ImageFolder(path_train_dataset, transform=transform_plain)
        dataset_val = ImageFolder(path_val_dataset, transform=transform_plain)
    except:
        print("Make sure the dataset is saved according to https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html as well as train e val sets are present.")
        sys.exit(1)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=64,
        num_workers=8,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=64,
        num_workers=8,
    )

    acc = compute_knn(
        backbone.eval(),
        data_loader_train,
        data_loader_val,
    )

    print(f"KNN accuracy: {acc}")
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UMAP')
    
    parser.add_argument(
        '--path_dataset',
        type=str,
        required=True,
        help='Path to dataset'
    )
    
    parser.add_argument(
        '--path_backbone',
        type=str,
        required=True,
        help='Path to nackbone'
    )
    args = parser.parse_args()
    main(**vars(args))