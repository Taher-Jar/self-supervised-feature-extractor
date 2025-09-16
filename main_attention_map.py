import argparse
from pathlib import Path

import PIL
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

def get_last_selfattention(backbone, img):
    """Get the attention weights of CLS from the last self-attention layer.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated Vision Transformer. Note that we will in-place
        take the `head` attribute and replace it with `nn.Identity`.

    img : torch.Tensor
        Image of shape `(1, 3, size, size)`.

    Returns
    -------
    torch.Tensor
        Attention weights `(n_samples, n_heads, n_patches)`.
    """
    attn_module = backbone.blocks[-1].attn
    n_heads = attn_module.num_heads

    # define hook
    inp = None
    def fprehook(self, inputs):
        nonlocal inp
        inp = inputs[0]

    # Register a hook
    handle = attn_module.register_forward_pre_hook(fprehook)

    # Run forward pass
    _ = backbone(img)
    handle.remove()

    B, N, C = inp.shape
    qkv = attn_module.qkv(inp).reshape(B, N, 3, n_heads, C // n_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)

    return attn[0, :, 0, 1:]

def threshold(attn, k=30):
    """threshold attetion map to visualize only the k highst attention pixels

    Parameters
    ----------
    attentions : List[torch.Tensor]
       List of heads attention maps.

    Returns
    -------
    attentions : List[torch.Tensor]
       List of heads attention maps.
    """
    n_heads = len(attn)
    indices = attn.argsort(dim=1, descending=True)[:, k:]#k number of high attention we're going to visualize
    for head in range(n_heads):
        attn[head, indices[head]] = 0

    attn /= attn.sum(dim=1, keepdim=True)

    return attn

def get_attn_map(img, backbone):
    """Get attention heads maps given an image and a backbone 

    Parameters
    ----------
    img : torch.Tensor
        Image of shape `(1, 3, size, size)`.

    backbone : timm.models.vision_transformer.VisionTransformer
        The vision transformer.

    Returns
    -------
    attentions : List[torch.Tensor]
       List of heads attention maps.
    """
    patch_size = backbone.patch_embed.proj.kernel_size[0]
    n_patches = img.shape[2:][0]// patch_size #square image
    attentions = get_last_selfattention(backbone, img) 
    attentions = attentions / attentions.sum(dim=1, keepdim=True)  # (n_heads, n_patches)
    attentions = threshold(attentions, k=30)
    attentions = attentions.reshape(-1, n_patches, n_patches)
    attentions = F.interpolate( attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest" )[0]

    return attentions

def save_attn_map(img, attn_map, path_file_out):
    """Save anntention maps to path_file_out

    Parameters
    ----------
    img : torch.Tensor
        Image of shape `(1, 3, size, size)`.

    attn_map : List[torch.Tensor]
       List of heads attention maps.
           
    path_file_out : torch.Tensor
        path to save figures.

    Returns
    -------
    torch.Tensor
        Attention weights `(n_samples, n_heads, n_patches)`.
    """
    n_heads = len(attn_map) 
    plt.figure(figsize=(n_heads+3, 3), dpi=150)
    plt.subplot(1, n_heads+1, 1)
    plt.title("input")
    plt.imshow(np.array((img)))
    plt.axis("off")
    for i in range(n_heads):
        plt.subplot(1, n_heads+1, i+2)
        plt.title("Head #"+str(i))
        plt.imshow(attn_map[i])
        plt.axis("off")
    plt.savefig(path_file_out)
    plt.clf()


def main(path_dir_images, path_backbone):

    path_backbone = Path(path_backbone)
    backbone = torch.load(path_backbone).backbone
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone.to(device)

    path_dir_images = Path(path_dir_images)
    image_paths = list(path_dir_images.glob("**/*.JPEG"))

    transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.Resize((224,224)),] )

    for idx, path_img in tqdm(enumerate(image_paths), desc=f"Looping over images"):

        img = PIL.Image.open(path_img)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        attn_map = get_attn_map(img_tensor, backbone).cpu().numpy()

        path_dir_out = Path("attn_map")
        path_dir_out.mkdir(parents=True, exist_ok=True)
        save_attn_map(img, attn_map, path_dir_out / f"img_{idx}.png")
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention map')
    
    parser.add_argument(
        '--path_dir_images',
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
