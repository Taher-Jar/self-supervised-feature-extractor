from pathlib import Path

import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics
import torchvision.transforms as transforms

from src.backbones.mobilenet import Backbone

class Head(nn.Module):
  
  def __init__(
            self,
            in_dim,  
            n_layers,
            hidden_dim, 
            dropout,
            n_class 
    ):
    super().__init__()
    n_layers = max(n_layers, 1)
    layers = []
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.GELU())

    for _ in range(n_layers - 1):
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
    
    last_layer = nn.Linear(hidden_dim, n_class)
    layers.append(last_layer)
    
    self.head = nn.Sequential(*layers)

    self.apply(self._init_weights)

  def _init_weights(self, m):
    """Initialize learnable parameters."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
  def forward(self, x):
    y = self.head(x)  

    return y

class Supervised_Model(nn.Module):
    def __init__(self, backbone, head, n_class):
        super().__init__()

        path_backbone = Path(backbone.timm_backbone)
        
        if path_backbone.suffix in [".pth", ".pt", ".ckpt"]:
            self.backbone =  torch.load(path_backbone)
        else:
            self.backbone = Backbone(timm_backbone=str(path_backbone), pretrain=backbone.pretrain)

        if backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.input_size = self.backbone.in_dimensions()[1:]
        self.normalize = self.backbone.get_normalize()

        input_dim_head = self.backbone.out_dimensions()
        self.head = Head( in_dim=input_dim_head, **head, n_class=n_class )
    
    def forward(self, x):
        embeddings = self.backbone(x)
        y = self.head(embeddings)

        return y

    def get_input_size(self):
        return self.input_size
    
    def get_normalize(self):
        return self.normalize
    
    def get_prod_model(self):
        return {"model" : self}

class Supervised_Trainer(pl.LightningModule):
    
    def __init__(self, backbone, head, n_class, **kwargs):
        super().__init__()
        self.model = Supervised_Model(backbone, head, n_class)
       
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.transform = transforms.Compose(
        [
            transforms.Resize(self.model.get_input_size()),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize(*self.model.get_normalize()),
        ])

        self.save_hyperparameters()
        
    def forward(self, x):
        y = self.model(x)

        return y

    def training_step(self, batch, batch_idx):
        X, y = batch 
        out = self(X)
        loss = self.loss(out, y)
        self.train_accuracy(out.softmax(dim=-1), y)
        self.log("loss", loss, on_step=True) 
        
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        X, y = batch 
        out = self(X)
        self.val_accuracy(out.softmax(dim=-1), y)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_transform(self):
        return self.transform

    def save_modules(self):
        #check is a dict
        return self.model.get_prod_model() 

