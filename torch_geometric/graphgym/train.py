import logging

import pytorch_lightning as pl

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule


def train(model: GraphGymModule, train_dataloaders, val_dataloaders=None,
          test_dataloaders=None):
    """
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    resume_from_checkpoint = None
    if cfg.train.auto_resume:
        # TODO: @aniketmaurya load from a specific epoch
        resume_from_checkpoint = cfg.run_dir

    logger = LoggerCallback()
    trainer = pl.Trainer(enable_checkpointing=True,
                         resume_from_checkpoint=resume_from_checkpoint,
                         callbacks=[logger], max_epochs=cfg.optim.max_epoch)
    trainer.fit(model, train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders)

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))
