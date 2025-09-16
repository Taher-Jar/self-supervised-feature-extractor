import argparse
from pathlib import Path

import seaborn as sns
import torch
import torch.nn as nn
import umap
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import pandas as pd

import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class UMAP:
    def __init__(self, color_palette: str = "hls"):
        """Offline UMAP helper.

        Args:
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
        """

        self.color_palette = color_palette

    def plot(
        self,
        backbone: nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the model.
        **Note: the model should produce features for the forward() function.

        Args:
            backbone (nn.Module): current model.
            dataloader (torch.utils.data.Dataloader): current dataloader containing data.
        """

        device = next(backbone.parameters()).device
        data = []
        Y = []

        # set module to eval model and collect all feature representations
        backbone.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Collecting features"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats = backbone(x)
                data.append(feats.cpu())
                Y.append(y.cpu())

        data = torch.cat(data, dim=0).numpy()
        Y = torch.cat(Y, dim=0)
        num_classes = len(torch.unique(Y))
        Y = Y.numpy()

        print("Creating UMAP")
        data = umap.UMAP(n_components=2).fit_transform(data)

        # passing to dataframe
        df = pd.DataFrame()
        df["feat_1"] = data[:, 0]
        df["feat_2"] = data[:, 1]
        labels = dataloader.dataset.classes
        Y = [labels[idx] for idx in Y]
        df["Y"] = Y
        plt.figure(figsize=(9, 9))
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=df,
            legend="full",
            alpha=0.3,
        )
        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # manually improve quality of imagenet umaps
        if num_classes > 100:
            anchor = (0.5, 1.8)
        else:
            anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
        plt.tight_layout()

        # save plot locally as well
        plt.savefig("umap.pdf")
        plt.close()


def main(path_dataset, path_backbone):

    path_dataset = Path(path_dataset)
    path_backbone = Path(path_backbone)

    backbone = torch.load(path_backbone)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    backbone.to(device)

    path_dataset_val = pathlib.Path(path_dataset)
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(backbone.in_dimensions()[1:]),
        ]
    )
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)
    dataloader = DataLoader( dataset_val_plain, batch_size=64, num_workers=8 )
    
    umap = UMAP()
    umap.plot(backbone, dataloader)


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