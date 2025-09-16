import sys
from pathlib import Path
from typing import List
import logging

import pytorch_lightning as pl

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, ConcatDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, path_datasets: List, model_name: str, batch_size: int, num_workers: int, transform, **kwargs):
        super().__init__()
        self.path_train_datasets = path_datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform 
        self.model_name = model_name

    def setup(self, stage= None):
        datasets = []
        for dst in self.path_train_datasets:
            datasets.append(ImageFolder(root=Path(dst), transform=self.transform))
            
        n_images = 4
        for i in range(0,len(datasets[0]),n_images):
            datasets[0] = list(datasets[0])
            datasets[0][i] = list(datasets[0][i])
            for j in range(1,n_images):
                datasets[0][i][0].extend(datasets[0][i+j][0])
            datasets[0][i] = tuple(datasets[0][i])
                
                
        datasets[0] = [datasets[0][i] for i in range(len(datasets[0])) if i%n_images==0]
        
        dataset = ConcatDataset(datasets)
        
        dataset_size = len(dataset)
        train_size = int(dataset_size * (1 - 0.1))
        val_size = dataset_size - train_size
        self.dataset_train, self.dataset_val = random_split(dataset, [train_size, val_size])

        #print(len(self.dataset_train))
        #for i in self.dataset_train:
        #    print(i)
        #    break
        

    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers, pin_memory=True,)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers, pin_memory=True,) if self.dataset_val else None


