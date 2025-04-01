import pandas as pd
import numpy as np
import pyBigWig
import bioframe as bf
import pyfaidx
import logging
from tangermeme.utils import one_hot_encode
import torch
from scipy.stats import pearsonr
from grelu.io.genome import get_genome
from torchmetrics.functional import pearson_corrcoef



def inference_model(model, test_loader, config):
    all_preds = [[] for _ in range(config.num_tracks)]
    all_labels = [[] for _ in range(config.num_tracks)]
    all_map = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_oh"].to(dtype=torch.float16)
            labels_cols = [batch[f"labels_{i}"] for i in range(config.num_tracks)]
            y = torch.stack(labels_cols, dim=-1) 
            z = batch["map"]  
        
            if config.model_name == "enformer":
                x = x.permute(0, 2, 1)
                preds = model(x)
            else:
                preds = model(x)["logits"] 

            for i in range(config.num_tracks):
                all_preds[i].append(preds[..., i].cpu().numpy())
                all_labels[i].append(y[..., i].cpu().numpy())

            all_map.append(z.cpu().numpy()) 
            
    for i in range(config.num_tracks):
        all_preds[i] = np.concatenate(all_preds[i], axis=0)    
        all_labels[i] = np.concatenate(all_labels[i], axis=0) 
    all_map = np.concatenate(all_map, axis=0)  

    pearsons = []
    for i in range(config.num_tracks):
        pred_i = all_preds[i].flatten()
        label_i = all_labels[i].flatten()

        if (
            len(pred_i) == 0 or
            np.std(pred_i) < 1e-6 or
            np.std(label_i) < 1e-6
        ):
            pearsons.append(0.0)
        else:
            pearsons.append(np.corrcoef(pred_i, label_i)[0, 1]) 
    pearson_per_pos = []
    for i in range(config.num_tracks):
        N, L = all_preds[i].shape
        pos_corrs = np.zeros(L)
        for pos in range(L):
            p = all_preds[i][:, pos]
            l = all_labels[i][:, pos]
            if np.std(p) < 1e-6 or np.std(l) < 1e-6:
                pos_corrs[pos] = np.nan
            else:
                pos_corrs[pos] = pearsonr(p, l)[0]
        pearson_per_pos.append(pos_corrs)

    np.savez_compressed(
        f"checkpoints/predictions_maskmap_True_by_track_chr10_{config.model_name}_loss_multinomiale.npz",
        **{f"preds_track_{i}": all_preds[i] for i in range(config.num_tracks)},
        **{f"labels_track_{i}": all_labels[i] for i in range(config.num_tracks)},
        **{f"pearson_track_{i}": pearsons[i] for i in range(config.num_tracks)},
        **{f"pearson_per_pos_{i}": pearson_per_pos[i] for i in range(config.num_tracks)},
        mapability=all_map
    )