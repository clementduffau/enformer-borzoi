import os
import wandb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics.functional import pearson_corrcoef
from transformers import get_scheduler
from accelerate import Accelerator
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
from get_data_model import (
    get_defined_intervals,
    expression_bw,
    create_interval_n,
    expression_val_histone,
    get_mapability,
    process_sequences,
    get_value_bw,
)
from test_model import inference_model
from model import load_enformer_model, load_flashzoi_model
from loss import PoissonMultinomialLoss
from pytorch_lightning import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



train_params = {
    'lr': 1e-4,
    'batch_size': 2,
    'num_workers': 16,
    'devices': 0,
    'max_epochs': 2,
    'checkpoint': False,
    'val_check_interval' : 50
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
            intervals = get_defined_intervals(genome, self.args.train_chroms + self.args.eval_chroms, min_length=self.args.seq_length)
            windows = create_interval_n(intervals, self.args.seq_length, self.args.bin_size, self.args.target_length_b,  self.args.n)

            df_train = pd.concat([windows[chrom] for chrom in self.args.train_chroms], axis=0)
            print(df_train)
            df_val = pd.concat([windows[chrom] for chrom in self.args.eval_chroms], axis=0)
            print(df_val)
            print(self.args.bin_size)
            train_data, val_data, _ = get_value_bw(self.args.signal_bw_paths, self.args.mappability_bw_path, df_train, df_val, pd.DataFrame,stage, self.args.bin_size)
            
            train_ds = Dataset.from_pandas(train_data)
            train_ds.set_format("torch")
            val_ds = Dataset.from_pandas(val_data)
            val_ds.set_format("torch")

            self.train_ds = process_sequences(train_ds, genome)
            self.val_ds = process_sequences(val_ds, genome)

        if stage == "test" :
            genome = get_genome(self.args.assembly)
            intervals = get_defined_intervals(genome, self.args.test_chroms, min_length=self.args.seq_length)
            windows = create_interval_n(intervals, self.args.seq_length, self.args.bin_size, self.args.target_length_b,  self.args.n)
            df_test = pd.concat([windows[chrom] for chrom in self.args.test_chroms], axis=0)
            print(df_test)
            _, _, test_data = get_value_bw(
                self.args.signal_bw_paths,
                self.args.mappability_bw_path,
                pd.DataFrame(), 
                pd.DataFrame(),
                df_test,
                stage,
                self.args.bin_size,
            )
            test_ds = Dataset.from_pandas(test_data)
            test_ds.set_format("torch")
            self.test_ds = process_sequences(test_ds, genome)


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_params['batch_size'], num_workers=self.train_params['num_workers'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.train_params['batch_size'], num_workers=self.train_params['num_workers'], shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.train_params['batch_size'], num_workers=self.train_params['num_workers'], shuffle = False)





class Modelelightning(pl.LightningModule):
    def __init__(self, model_name, train_params, config_mask_map, num_tracks, dropout_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.config_mask_map = config_mask_map
        self.train_params = train_params
        self.num_tracks = num_tracks
        if model_name == "enformer":
            self.model = load_enformer_model(num_tracks)
        elif model_name == "flashzoi":
            self.model = load_flashzoi_model(num_tracks, dropout_rate)
        
        #self.loss_fn = torch.nn.MSELoss()
        #self.loss_fn = torch.nn.functional.poisson_nll_loss 
        self.loss_fn = PoissonMultinomialLoss(multinomial_weight=5,reduction = 'mean',multinomial_axis= "length")
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.pearson = torchmetrics.PearsonCorrCoef()

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x= batch["input_oh"]
        labels_cols = [batch[f"labels_{i}"] for i in range(self.num_tracks)]
        y = torch.stack(labels_cols, dim=-1)

        z= batch["map"]

        if self.model_name == "enformer":
            x = x.permute(0, 2, 1).to(dtype=torch.float16)
            preds = self(x)
        else:
            x = x.to(dtype=torch.float16)
            preds = self(x)
            preds = preds["logits"]
        
        mask_map = (z != 0).unsqueeze(-1)          
        mask = (y !=0) 
        if self.config_mask_map is True: 
            ignore_mask = mask_map & mask 
        else :
            ignore_mask = mask
        f_preds = preds[ignore_mask] 
        f_labels = y[ignore_mask]
        f_preds = f_preds.squeeze(-1)
        #poisson
        #loss = self.loss_fn(f_preds.squeeze(), f_labels.squeeze(), reduction = 'mean')
        #mse
        #loss = self.loss_fn(f_preds.squeeze(), f_labels.squeeze())
        #multinomiale
        loss = self.loss_fn(f_preds.unsqueeze(1), f_labels.unsqueeze(1))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x = batch["input_oh"]
        labels_cols = [batch[f"labels_{i}"] for i in range(self.num_tracks)]
        y = torch.stack(labels_cols, dim=-1)
        z= batch["map"]
        if self.model_name == "enformer":
            x = x.permute(0, 2, 1).to(dtype=torch.float16)
            preds = self(x)
        else:
            x = x.to(dtype=torch.float16)
            preds = self(x)
            preds = preds["logits"]
        mask_map = (z != 0).unsqueeze(-1)           
        mask = (y !=0)  
        if self.config_mask_map is True: 
            ignore_mask = mask_map & mask 
        else :
            ignore_mask = mask  


        f_preds_all = [None] * self.num_tracks
        f_labels_all = [None] * self.num_tracks

        for i in range(self.num_tracks):
            if ignore_mask[..., i].sum() < 10:
                continue
            pred_i = preds[..., i]
            label_i = y[..., i]
            mask_i = ignore_mask[..., i]

            f_preds_all[i] = pred_i[mask_i]
            f_labels_all[i] = label_i[mask_i]
    
        pearsons = []
        for i in range(self.num_tracks):
            if (
                f_preds_all[i] is None or 
                f_labels_all[i] is None or 
                f_preds_all[i].numel() == 0 or 
                f_labels_all[i].numel() == 0 or 
                f_preds_all[i].std() < 1e-6 or 
                f_labels_all[i].std() < 1e-6
            ):
                pearson_i = torch.tensor(0.0)
            else:
                pearson_i = pearson_corrcoef(f_preds_all[i], f_labels_all[i])
            self.log(f"val_pearson_track_{i}", pearson_i, on_epoch=True, on_step=False, logger=True)
            pearsons.append(pearson_i)

        if len(pearsons) > 0:
            mean_pearson = torch.stack(pearsons).mean()
        else:
            mean_pearson = torch.tensor(0.0)

        self.log("val_pearson_mean", mean_pearson, on_epoch=True, on_step= False, prog_bar=True, logger=True)

        losses = []
        for i in range(self.num_tracks):
            if f_preds_all[i] is None or f_labels_all[i] is None or f_preds_all[i].numel() == 0:
                continue
            #loss_i = self.loss_fn(f_preds_all[i], f_labels_all[i], reduction='mean')
            loss_i = self.loss_fn(f_preds_all[i], f_labels_all[i])
            self.log(f"val_loss_track_{i}", loss_i, on_epoch=True, on_step=False, logger=True)
            losses.append(loss_i)

        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0)

        self.log("val_loss", loss, on_epoch=True,on_step = False, prog_bar=True, logger=True)

        try:
            f_preds_cat = torch.cat([t for t in f_preds_all if t is not None])
            f_labels_cat = torch.cat([t for t in f_labels_all if t is not None])
            self.log("val_mse", self.val_mse(f_preds_cat, f_labels_cat), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        except Exception:
            self.log("val_mse", torch.tensor(0.0), on_epoch=True,on_step=False, prog_bar=True, logger=True)

        return {"val_loss": loss }


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.train_params['lr'], weight_decay=1e-5)
        num_training_steps = (
            len(self.trainer.datamodule.train_dataloader()) 
            * self.hparams.train_params["max_epochs"] 
            // self.trainer.accumulate_grad_batches
        )

        lr_scheduler = get_scheduler(
            name="linear",  
            optimizer=optimizer,
            num_warmup_steps=int(0.5 * num_training_steps),
            num_training_steps=num_training_steps,
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",  
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    


class Config:
        model_name = "flashzoi"
        config_mask_map = True
        assembly = "hg38"
        dropout_rate = 0
        signal_bw_paths = ["DATA/ENCFF919ISB.bigWig", "DATA/ENCFF688ARC.bigWig", "DATA/ENCFF078OIB.bigWig","DATA/ENCFF224YHG.bigWig"]
        #signal_bw_paths = ["DATA/ENCFF688ARC.bigWig"]
        num_tracks = len(signal_bw_paths)
        mappability_bw_path = ["DATA/k36.Umap.MultiTrackMappability.bw"]
        if model_name == "enformer":
            seq_length, bin_size, target_length_b , n= 196608, 128, 896, 320
        else:
            seq_length, bin_size, target_length_b, n = 524_288, 32, 6144, 5120
        train_chroms, eval_chroms, test_chroms = [f"chr{i}" for i in range(5, 10)], ["chr14"], ['chr10']


def main():

    config = Config()
    wandb_logger = WandbLogger(project="genome-prediction",  name=f"{config.model_name}-mask_map_{config.config_mask_map}-num_tarcks_{config.num_tracks}")
    accelerator = Accelerator(mixed_precision="bf16")
    data_module  = DataModule(config, train_params=train_params)
    data_module.setup(stage = "fit")
    model = Modelelightning(model_name=config.model_name, 
                            train_params=train_params, 
                            config_mask_map = config.config_mask_map,
                            num_tracks = config.num_tracks,
                            dropout_rate = config.dropout_rate
            )
    
    model = accelerator.prepare(model)

    wandb_logger.watch(model, log="gradients", log_freq=100)

    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename=f"{config.model_name}-mask_map_{config.config_mask_map}-num_tarcks_{config.num_tracks}-loss_multinomiale",
    monitor="val_loss", 
    save_top_k=1,
    mode="min"        
)
    trainer = pl.Trainer(
        max_epochs=train_params['max_epochs'],
        val_check_interval=train_params['val_check_interval'],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=50,
        accelerator="gpu",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )

    trainer.fit(model, data_module)

    
    best_checkpoint = checkpoint_callback.best_model_path
    """

    model = Modelelightning.load_from_checkpoint(
    checkpoint_path="checkpoints/flashzoi-mask_map_False-num_tarcks_4-loss_mse.ckpt",
    model_name=config.model_name,
    config_mask_map=config.config_mask_map,
    train_params=train_params,
    num_tracks=config.num_tracks,
    )
    model.eval()
    
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    model,test_loader = accelerator.prepare(model, test_loader)
    inference_model(model, test_loader, config)

    """

if __name__ == "__main__":
    main()
    wandb.finish()
   
    