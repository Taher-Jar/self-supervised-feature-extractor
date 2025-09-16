[README.md](https://github.com/user-attachments/files/22370380/README.md)
# Self-supervised

This repository contains self-supervised methods for unsupervised visual representation learning powered by PyTorch, PyTorch Lightning and MLFlow for experiments tracking. It aims at learning a generic backbone that can construct a more robust features space. The <a href="https://github.com/rwightman/pytorch-image-models">Pytorch timm</a> library is used to initialize the backbone.

Self-supervised methods implemented:
 - <a href="https://arxiv.org/pdf/2104.14294.pdf">Dino</a> 


## Project structure
```
.
├──config
│   ├── datamodules
│   │   |   |- imaginette.yaml  
│   │   ├── experiments
│   │   |      |- default.yaml
│   │   └── models
│   │   |      |- dino.py
|   |   |      |- supervised.yaml
│   └── train.yaml
├── src.py
│   ├── backbones
│   │   |   |- timm.py  
│   │   ├── datamodules
│   │   |      |- default.yaml
│   │   └── models
│   │   |      |- dino.py
|   |   |      |- supervised.yaml
│   └── train.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── main_train.py
├── main_knn.py
├── main_umap.py
├── main_attention_map.py
```
- `config/`: Hydra configuations
- `main_train.py`: Train the specified model in hydra configs.
- `main_knn.py`: Evaluates a backbone in knn accuracy given a labeled datasets.
- `main_umap.py`: Use UMAP algorithm to visualize feature space given a dataset
- `main_attention.py`: Plots attentions maps given a backbone and a list of images, in case ViT is used as backbone.

<!-- ## Usage
It was tested on Python 3.9.10.
```shell
$ python -m venv /path/to/env`
$ source /path/to/env/bin/activate`
$ pip install -r requirements.txt
``` -->
## To Do

- Add requirments.txt
- Write a more clear README
- Clean Code
- Experiments with new datasets (AH for example)
- Implement new self-supervised models
- ...

