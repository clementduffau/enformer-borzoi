import os

import cv2
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import precision_recall_curve
from grelu.lightning.metrics import MSE, BestF1, PearsonCorrCoef
from torchmetrics import AUROC, Accuracy, AveragePrecision, MetricCollection
from torch import Tensor, nn, optim
from typing import Callable, List, Optional, Tuple, Union
import logging
import pandas as pd
import grelu.lightning
import pyBigWig
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from enformer_pytorch.data import str_to_seq_indices
from grelu.io.genome import get_genome
from intervals import (
    get_defined_intervals,
    expression_bw,
    create_interval_n
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
class LightningModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.eval_every_n_steps = args.eval_every_n_steps
        self.model = load_enformer_model(num_tracks=len(args.signal_bw_paths))
        self.loss = nn.MSELoss()
        self.activation = nn.Identity()
        self.val_losses = []
    
    def forward(self, x):
        x = self.model(x)
        return self.activation(x)
    
    def training_step(self, batch, batch_idx):

        x, y = batch["input_ids"], batch["labels"]
        logits = self(x)
        loss = self.loss(logits, y)
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if (self.global_step + 1) % self.eval_every_n_steps == 0:
            x, y = batch["input_ids"], batch["labels"]
            logits = self(x)
            loss = self.loss(logits, y)
            self.log("val_loss", loss, prog_bar=True, logger=True)
            self.val_losses.append(loss)
            return loss
        return None
    
    def on_validation_epoch_end(self):
        if self.val_losses:
            mean_loss = torch.stack(self.val_losses).mean()
            self.log("mean_val_loss", mean_loss, prog_bar=True, logger=True)
            self.val_losses.clear()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def initialize_metrics(self):
        
        metrics = MetricCollection(
            {
                "mse": MSE(num_outputs=self.model.head.n_tasks, average=False),
                "pearson": PearsonCorrCoef(
                    num_outputs=self.model.head.n_tasks, average=False
                ),
            }
        )
        return metrics

    def update_metrics(self, metrics: dict, y_hat: Tensor, y: Tensor) -> None:
        
        metrics.update(y_hat, y)
    
    def format_input(self, x: Union[Tuple[Tensor, Tensor], Tensor]) -> Tensor:

        # if x is a tuple of sequence, label, return the sequence
        if isinstance(x, Tensor):
            if x.ndim == 3:
                return x
            else:
                return x.unsqueeze(0)
        elif isinstance(x, Tuple):
            return x[0]
        else:
            raise Exception("Cannot perform forward pass on the given input format.")

"""



model_params = {
    'model_type': 'EnformerPretrainedModel',
    'n_tasks': 1,  
    'crop_len': 0,
    'n_transformers': 1,
}

train_params = {
    'task':'regression',
    'loss': 'poisson',
    'lr': 1e-4,
    'logger': 'wandb',
    'batch_size': 2,
    'num_workers': 1,
    'devices': 0,
    'save_dir': "./experiment",
    'optimizer': 'adam',
    'max_epochs': 5,
    'checkpoint': True,
    'val_check_interval': 500,
}


def prepare_batch(batch, genome, signal_bw_paths,mappability_bw_path, bin_size=128, ignore_low_mappa=True, ignore_low_mappa_threshold=0.9):
    signal_bws = [pyBigWig.open(str(bw_path)) for bw_path in signal_bw_paths]
    mappa_bw = pyBigWig.open(mappability_bw_path) if mappability_bw_path else None
    seqs, targets, ignore_masks = [], [], []
    
    for i in range(len(batch["start"])):
        chrom, start, end = batch["chrom"][i], batch["start"][i], batch["end"][i]
        seq = genome.get_seq(chrom, start + 1, end).seq.upper()
        seqs.append(seq)
        
        target = []
        for signal_bw in signal_bws:
            signal_nt = signal_bw.values(chrom, start, end, numpy=True)
            signal_b = expression_bw(np.nan_to_num(signal_nt), bin_size, statistic="mean")
            target.append(signal_b)
        target = np.stack(target, axis=1)
        targets.append(target)
        
        ignore_mask = np.isnan(target) | (target == 0)
        
        if ignore_low_mappa and mappa_bw:
            mappa_nt = mappa_bw.values(chrom, start, end, numpy=True)
            mappa_b = expression_bw(np.nan_to_num(mappa_nt), bin_size, statistic="mean")
            low_mappa_mask = (mappa_b <= ignore_low_mappa_threshold)
            ignore_mask = ignore_mask | low_mappa_mask
        
        ignore_masks.append(ignore_mask)
    
    if mappa_bw:
        mappa_bw.close()
    
    return {"input_ids": str_to_seq_indices(seqs), "labels": targets, "ignore_mask": ignore_masks}


def load_data(args):
    genome = get_genome(args.assembly)
    intervals = get_defined_intervals(genome, args.train_chroms + args.eval_chroms, min_length=args.seq_length)
    windows = create_interval_n(intervals, args.seq_length, args.target_length_b * args.bin_size)
    
    
    ds = DatasetDict({
        "train": Dataset.from_pandas(windows[windows.chrom.isin(args.train_chroms)]),
        "eval": Dataset.from_pandas(windows[windows.chrom.isin(args.eval_chroms)]),
        "test" : Dataset.from_pandas(windows[windows.chrom.isin(args.test_chroms)])
    })
    
    ds = ds.map(lambda batch: prepare_batch(batch, genome, args.signal_bw_paths, args.bin_size), batched=True)
    return ds

def main():
    class Args:
        assembly = "hg38"
        signal_bw_paths = "SRR18582971.signal.bl_s8_exz.bw"
        mappability_bw_path = "k36.Umap.MultiTrackMappability.bw"
        seq_length, bin_size, target_length_b = 196608, 128, 896
        train_chroms, eval_chroms, test_chroms = [f"chr{i}" for i in range(1, 14)], ["chr14"], ["chr15"]
        lr = 1e-4
        weight_decay = 1e-5
        eval_every_n_steps = 500
    
    args = Args()
    dataset = load_data(args)
    print(dataset)
    model_params['n_tasks'] = len(args.signal_bw_paths) 

    experiment='experiment'
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    model = grelu.lightning.LightningModel(model_params=model_params, train_params=train_params)
    trainer = model.train_on_dataset(
        train_dataset=dataset["train"],
        val_dataset=dataset["eval"],
    )
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(best_checkpoint)

    model = grelu.lightning.LightningModel.load_from_checkpoint(best_checkpoint)

    test_metrics = model.test_on_dataset(
    test_ds = dataset["test"],
    batch_size=4,
    devices=0,
    num_workers=1
    )

    test_metrics

if __name__ == "__main__":
    main()

   
    