import os
import wandb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
from grelu.lightning.metrics import MSE, BestF1, PearsonCorrCoef
from torchmetrics import AUROC, Accuracy, AveragePrecision, MetricCollection
from torch import Tensor, nn, optim
from typing import Callable, List, Optional, Tuple, Union
import logging
from tangermeme.utils import one_hot_encode
import pandas as pd
from enformer_pytorch.finetune import HeadAdapterWrapper
import grelu.lightning
from grelu.data.dataset import LabeledSeqDataset
from pytorch_lightning.loggers import WandbLogger
import pyBigWig
import scipy.stats
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from enformer_pytorch.data import str_to_seq_indices
from enformer_pytorch import Enformer
from grelu.io.genome import get_genome
from intervals import (
    get_defined_intervals,
    expression_bw,
    create_interval_n,
    expression_val_histone,
    get_mapability,
    process_sequences,
    get_value_bw
)
from model import load_enformer_model
from pytorch_lightning import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb_logger = WandbLogger(project="genome-prediction")

train_params = {
    'lr': 1e-4,
    'batch_size': 2,
    'num_workers': 31,
    'devices': 0,
    'save_dir': "./experiment",
    'max_epochs': 5,
    'checkpoint': True,
    'val_check_interval' : 100
}

class DataModule(pl.LightningDataModule):
    def __init__(self, args, train_params):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.train_params = train_params
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage = None):
        if stage == "fit":
            genome = get_genome(self.args.assembly)
            intervals = get_defined_intervals(genome, self.args.train_chroms + self.args.eval_chroms + self.args.test_chroms, min_length=self.args.seq_length)
            windows = create_interval_n(intervals, self.args.seq_length, self.args.bin_size, self.args.target_length_b * self.args.bin_size)

            df_train = pd.concat([windows[chrom] for chrom in self.args.train_chroms], axis=0)
            df_val = pd.concat([windows[chrom] for chrom in self.args.eval_chroms], axis=0)

            train_data, val_data, _ = get_value_bw(self.args.signal_bw_paths, self.args.mappability_bw_path, df_train, df_val, pd.DataFrame,stage, self.args.bin_size)
            
            train_ds = Dataset.from_pandas(train_data)
            train_ds.set_format("torch")
            val_ds = Dataset.from_pandas(val_data)
            val_ds.set_format("torch")

            self.train_ds = process_sequences(train_ds, genome)
            self.val_ds = process_sequences(val_ds, genome)

        if stage == "test" :
            genome = get_genome(self.args.assembly)
            df_test = pd.concat([windows[chrom] for chrom in self.args.test_chroms], axis=0)

            _, _, test_data = get_value_bw(
                self.args.signal_bw_paths,
                self.args.mappability_bw_path,
                pd.DataFrame(), 
                pd.DataFrame(),
                df_test,
                self.args.bin_size,
            )

            self.test_ds = process_sequences(Dataset.from_pandas(test_data), genome)


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_params['batch_size'], num_workers=self.train_params['num_workers'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.train_params['batch_size'], num_workers=self.train_params['num_workers'], shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.train_params['batch_size'], num_workers=self.train_params['num_workers'], shuffle = False)





class Enformerlightning(pl.LightningModule):
    def __init__(self, train_params):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = load_enformer_model()
        self.model = HeadAdapterWrapper(enformer = self.backbone,num_tracks=1)
        self.loss_fn = torch.nn.functional.poisson_nll_loss 
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        #self.train_pearson = torchmetrics.PearsonCorrCoef()
        #self.val_pearson = torchmetrics.PearsonCorrCoef()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y , z = batch["input_oh"], batch["labels"], batch["map"]
        x = x.permute(0, 2, 1).to(dtype=torch.float16)
        preds = self(x)
        mask_map = (z != 0)            
        mask = (y !=0)  
        ignore_mask = mask_map | mask 
        f_preds = preds[ignore_mask] 
        f_labels = y[ignore_mask]
        f_preds = f_preds.squeeze(-1)
        loss = self.loss_fn(f_preds.squeeze(), f_labels.squeeze(), reduction = 'mean')
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        #self.log("train_mse", self.train_mse(f_preds, f_labels), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch["input_oh"], batch["labels"], batch["map"]
        x = x.permute(0, 2, 1).to(dtype=torch.float16)
        preds = self(x)
        mask_map = (z != 0)            
        mask = (y !=0)  
        ignore_mask = mask_map | mask  
        f_preds = preds[ignore_mask].squeeze(-1)
        f_labels = y[ignore_mask]
        
        loss = self.loss_fn(f_preds.squeeze(), f_labels.squeeze(), reduction = 'mean')

        if f_preds.numel() > 1 and f_labels.numel() > 1:
            pearson_score = scipy.stats.pearsonr(f_preds.cpu().numpy().flatten(), 
                                             f_labels.cpu().numpy().flatten())[0]
        else:
            pearson_score = float('nan')

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("val_mse", self.val_mse(f_preds, f_labels), on_epoch=True,on_step=True, prog_bar=True, logger= True)
        self.log("val_pearson", pearson_score, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss, "val_pearson": pearson_score}
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(float('nan'))).item()
        avg_pearson = self.trainer.callback_metrics.get("val_pearson", torch.tensor(float('nan'))).item()

        print(f" Validation Epoch End - Loss: {avg_loss:.4f}, Pearson: {avg_pearson:.4f}")

        self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("avg_val_pearson", avg_pearson, prog_bar=True, logger=True)


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.train_params['lr'])


class Args:
        assembly = "hs1"
        signal_bw_paths = ["SRR18582971.signal.bl_s8_exz.bw"]
        mappability_bw_path = ["k36.Umap.MultiTrackMappability.bw"]
        seq_length, bin_size, target_length_b = 196608, 128, 896
        train_chroms, eval_chroms, test_chroms = [f"chr{i}" for i in range(1, 2)], ["chr14"], ["chr15"]

def main():

    args = Args()
    data_module  = DataModule(args, train_params=train_params)
    data_module.setup(stage = "fit")
    model = Enformerlightning(train_params=train_params)
    
    trainer = pl.Trainer(
        max_epochs=train_params['max_epochs'],
        val_check_interval=train_params['val_check_interval'],
        logger=wandb_logger,
        log_every_n_steps=100,
        accelerator="gpu",
    )

    trainer.fit(model, data_module)


    wandb.log({"mse": trainer.logged_metrics['mse'],
               "pearson": trainer.logged_metrics['pearson'],
               "loss": trainer.logged_metrics['loss']})
    

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(best_checkpoint)

    

if __name__ == "__main__":
    main()
    wandb.finish()
   
    