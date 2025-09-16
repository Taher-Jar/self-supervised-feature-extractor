import sys

import torch
import torch.nn as nn
import timm


class Backbone(nn.Module):
  
  def __init__( self,timm_backbone, pretrain, **kwargs ):
    super().__init__()
    
    try:
      model = timm.create_model(timm_backbone, pretrained=pretrain)
    except RuntimeError:
      print(f"Make sure to select a right backbone from : {timm.list_models(pretrained=pretrain)}")
      raise

    
    self.in_dim = model.default_cfg["input_size"]
    self.normalize = (model.default_cfg["mean"], model.default_cfg["std"])
    self.out_dim= model.get_classifier().in_features
    name_last_layer = model.default_cfg["classifier"]
    setattr(model, name_last_layer, nn.Identity())
    self.backbone = model

  def forward(self, x):
    feats = self.backbone(x)
    return feats

  def in_dimensions(self):
    return self.in_dim
  
  def get_normalize(self):
    return self.normalize

  def out_dimensions(self):
    return self.out_dim
