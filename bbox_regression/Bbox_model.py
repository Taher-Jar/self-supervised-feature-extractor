import torch
import torch.nn as nn

class Bbox(nn.Module):
  
  def __init__(
            self,
            backbone,  
            head):
      super().__init__()

      self.backbone = backbone
      self.head = head
    
  def forward(self, x):
    embeddings = self.backbone(x)
    y = self.head(embeddings)

    return y