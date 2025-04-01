from accelerate import Accelerator
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch import Enformer
import enformer_pytorch
from typing import Optional
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from torch import nn
import torch

def load_enformer_model(num_tracks):
    model = enformer_pytorch.from_pretrained("EleutherAI/enformer-official-rough")
    return HeadAdapterWrapper(enformer=model, num_tracks = num_tracks)


def load_flashzoi_model(num_tracks, dropout_rate):
    config = FlashZoiHeadAdapterConfig(num_tracks, dropout_rate)
    return FlashZoiHeadAdapter(config)

def load_flashzoi_model_or():
    model = Borzoi.from_pretrained('johahi/flashzoi-replicate-0') # 'johahi/flashzoi-replicate-[0-3]'
    return model

class FlashZoiHeadAdapterConfig(PretrainedConfig):
    model_type = "flashzoi_head_adapter"

    def __init__(
        self,
        num_tracks, 
        dropout_rate,
        freeze_flashzoi: bool = False, 
        finetune_last_n_layers_only: Optional[int] = None,  
        loss_kind: str = "poisson", 
        pretrained_model_name: str = "johahi/flashzoi-replicate-0",
        flashzoi_config = BorzoiConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_tracks = num_tracks
        self.dropout_rate = dropout_rate
        self.freeze_flashzoi = freeze_flashzoi
        self.finetune_last_n_layers_only = finetune_last_n_layers_only
        self.loss_kind = loss_kind
        self.pretrained_model_name = pretrained_model_name
        self.flashzoi_config = flashzoi_config



class FlashZoiHeadAdapter(PreTrainedModel):

    config_class = FlashZoiHeadAdapterConfig

    def __init__(self, config: FlashZoiHeadAdapterConfig):
        super().__init__(config)
        self.config = config
        self.flashzoi = Borzoi.from_pretrained(config.pretrained_model_name)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        flashzoi_hidden_dim = 1920
        self.head = nn.Sequential(
            nn.Linear(flashzoi_hidden_dim, config.num_tracks),
            nn.Softplus()
        )
        for name, param in self.flashzoi.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}") 
    
    def forward(
        self,
        input_oh: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        map : Optional[torch.Tensor] = None
    ):
        #print(f"input_oh: {input_oh.shape}")
        
        embeddings = self.flashzoi.get_embs_after_crop(input_oh)
        embeddings = self.flashzoi.final_joined_convs(embeddings)
        #print(f"Shape de embeddings après final_joined_convs : {embeddings.shape}")
        embeddings = embeddings.permute(0, 2, 1)

        logits = self.head(embeddings)

        return {"logits": logits}
