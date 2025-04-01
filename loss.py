
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
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional



class PoissonMultinomialLoss(nn.Module):
    def __init__(
        self,
        multinomial_weight: float = 5,
        mse_weight: float = 1,
        eps: float = 1e-7,
        log_input: bool = False,
        reduction: str = "mean",
        multinomial_axis: str = "length",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.multinomial_weight = multinomial_weight
        self.mse_weight = mse_weight
        self.log_input = log_input
        self.reduction = reduction
        if multinomial_axis == "length":
            self.axis = 2
        elif multinomial_axis == "task":
            self.axis = 1

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input: (B, T, L)
            target: (B, T, L)
            mask: (B, T, L) — optional binary mask
        """
        input = input.to(torch.float32)
        target = target.to(torch.float32)

        if self.log_input:
            input = torch.exp(input)
        else:
            input = input + self.eps

        if mask is not None:
            input = input * mask
            target = target * mask

        total_input = input.sum(dim=self.axis, keepdim=True) + self.eps
        total_target = target.sum(dim=self.axis, keepdim=True) + self.eps

        # --- Poisson ---
        poisson_loss = F.poisson_nll_loss(total_input, total_target, log_input=False, reduction='none')
        poisson_loss /= input.shape[self.axis]

        # --- Multinomial ---
        log_p_input = torch.log(input / total_input + self.eps)
        multinomial = -(target * log_p_input).mean(dim=self.axis, keepdim=True)
        
        # --- MSE ---
        if mask is not None:
            mse = F.mse_loss(input[mask.bool()], target[mask.bool()], reduction='mean')
        else:
            mse = F.mse_loss(input, target, reduction='mean')
        """
        # --- Total ---
        total_loss = (
            self.multinomial_weight * multinomial.mean() +
            poisson_loss.mean() +
            self.mse_weight * mse
        )
        """
        total_loss = (
            self.multinomial_weight * multinomial.mean() +
            poisson_loss.mean())
        
        return total_loss

        