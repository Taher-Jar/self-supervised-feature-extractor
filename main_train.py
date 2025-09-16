import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import matplotlib as plt

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    #MODEL

    model = hydra.utils.instantiate(cfg.models, n_class=cfg.datamodules.n_class, batch_size=cfg.experiments.batch_size)
    #DATAMODULE
    data_module = hydra.utils.instantiate(cfg.datamodules, model_name=cfg.models.name, batch_size=cfg.experiments.batch_size, num_workers=cfg.experiments.num_workers, transform=model.get_transform())
    #MLFLOW
    experiment_name = f"{cfg.datamodules.name}__{cfg.models.name}"
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri="file:./mlruns")
    #TRAINER PL
    seed_everything(cfg.experiments.seed, workers=True)
    dir_model_path = Path("model_dir")
    shutil.rmtree(dir_model_path, ignore_errors=True)
    dir_model_path_checkpoints = dir_model_path / "PL_checkpoints"
    checkpoint_callback = ModelCheckpoint( **cfg.models.checkpoint, dirpath=dir_model_path_checkpoints )
    # #TRAINER STRATEGY
    if cfg.experiments.gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(process_group_backend="nccl") # gloo windows backend | nccl linux backend
        trainer = Trainer(strategy=strategy, devices=cfg.experiments.gpus, num_nodes=1, max_epochs=cfg.experiments.max_epochs, deterministic=True, accelerator='gpu', logger=mlf_logger, callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", min_delta=0.1, patience=10)])
    else:
        trainer = Trainer(devices=cfg.experiments.gpus, max_epochs=cfg.experiments.max_epochs, deterministic=True, accelerator='gpu', logger=mlf_logger, callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", min_delta=0.1, patience=10)])
        #trainer.tune(model, data_module)
        #lr_finder = trainer.tuner.lr_find(model, data_module,min_lr=1e-15, max_lr=1e-4, early_stop_threshold=None, num_training=100, mode="exponential")
        #new_lr = lr_finder.suggestion()
        #print(new_lr)
        #lr_finder.plot(show=True, suggest=True)


        

        

    #LOG HYPERPARAMETERS TO MLFLOW
    trainer.logger.log_hyperparams(cfg)
    #FIT MODEL
    trainer.fit(model, data_module)

    #LOG MODELS TO MLFLOW
    modules = model.save_modules()
    for key, value in modules.items():
        dir_model_path_backbone = dir_model_path / str(key)
        dir_model_path_backbone.mkdir(parents=True, exist_ok=True)
        torch.save(value, dir_model_path_backbone / f"model.pth")

    mlf_logger.experiment.log_artifact(mlf_logger.run_id, dir_model_path)

if __name__ == "__main__":
    main()

    