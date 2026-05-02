from typing import List, Optional
from pathlib import Path

from einops import rearrange
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from chemeleon.text_encoder import MODEL_NAMES, ARTIFACT_PATHS
from chemeleon.text_encoder.crystal_clip import CrystalClip
from chemeleon.utils.diff_utils import prob_mask_like
import wandb


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_name: str = "lfoppiano/MatTPUSciBERT",
        text_embed_dim: int = 768,
        max_text_len: int = 256,
        text_dim: int = 512,
        trainable_text_encoder: bool = False,
        pretrained_clip_model: Optional[CrystalClip] = None,
    ):
        super(TextEncoder, self).__init__()
        # configs
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = text_embed_dim
        self.max_text_len = max_text_len
        self.text_dim = text_dim

        # embedding layers
        self.text_emb = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.text_embed_dim),
            nn.LayerNorm(self.text_embed_dim),
            nn.GELU(),
            nn.Linear(self.text_embed_dim, self.text_dim),
        )
        self.null_text_embeds = nn.Parameter(torch.randn(1, self.text_embed_dim))
        # set up text encoder
        if pretrained_clip_model is not None:
            self.clip_model = pretrained_clip_model
            self.text_encoder = self.clip_model.text_encoder
            self.tokenizer = self.clip_model.tokenizer
        else:
            self.clip_model = None
            self.text_encoder = None
            self.tokenizer = None
            self.text_encoder, self.tokenizer = self._setup_text_encoder(
                trainable_text_encoder
            )
        assert (
            self.text_encoder is not None or self.clip_model is not None
        ), "Text encoder is not loaded properly!"

    def _setup_text_encoder(self, trainable_text_encoder: bool = False):
        assert (
            self.text_encoder_name in MODEL_NAMES
        ), f"Invalid model name. Must be one of {MODEL_NAMES}"
        if self.text_encoder_name.startswith("chemeleon/"):
            # download model from wandb
            api = wandb.Api()
            artifact_path = ARTIFACT_PATHS[self.text_encoder_name]
            model_id = artifact_path.split("/")[-1]
            artifact = api.artifact(artifact_path)
            # check if the model is already downloaded
            if not Path(f".cache/artifacts/{model_id}").exists():
                artifact.download(f".cache/artifacts/{model_id}")
            # load clip model
            model_path = Path(f".cache/artifacts/{model_id}/model.ckpt")
            self.clip_model = CrystalClip.load_from_checkpoint(  # pylint: disable=E1120
                model_path
            )
            model = self.clip_model.text_encoder
            tokenizer = self.clip_model.tokenizer
        elif self.text_encoder_name.startswith("t5"):
            model = T5EncoderModel.from_pretrained(self.text_encoder_name)
            tokenizer = T5Tokenizer.from_pretrained(self.text_encoder_name)
            print(f"Initalized T5 model for text encoder from {self.text_encoder_name}")
        elif self.text_encoder_name.startswith("microsoft"):
            model = AutoModelForCausalLM.from_pretrained(
                self.text_encoder_name, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.text_encoder_name, trust_remote_code=True
            )
            # set eos token as pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # set model to output hidden states
            model.config.output_hidden_states = True
            print(
                f"Initalized Microsoft model for text encoder from {self.text_encoder_name}"
            )
        elif self.text_encoder_name.startswith("meta-llama"):
            model = AutoModelForCausalLM.from_pretrained(self.text_encoder_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            # set eos token as pad token
            tokenizer.pad_token = tokenizer.eos_token
            # set model to output hidden states
            model.config.output_hidden_states = True
            print(
                f"Initalized Meta-LLaMa model for text encoder from {self.text_encoder_name}"
            )
        else:
            model = BertModel.from_pretrained(self.text_encoder_name)
            tokenizer = BertTokenizer.from_pretrained(self.text_encoder_name)
            print(
                f"Initalized Bert model for text encoder from {self.text_encoder_name}"
            )

        if trainable_text_encoder:
            for param in model.parameters():
                param.requires_grad = True
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        return model, tokenizer

    def text_encode(self, batch_text: List[str], device: str) -> torch.Tensor:
        tokenized = self.tokenizer.batch_encode_plus(
            batch_text,
            padding="longest",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        )
        # set device
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        self.text_encoder.to(device)

        if self.text_encoder_name.startswith("t5"):
            # embeddings are the mean of last hidden states
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            last_hidden_state = outputs.last_hidden_state
            embeddings = last_hidden_state.masked_fill(
                ~rearrange(attention_mask, "... -> ... 1").bool(), 0.0
            )  # [B, L, D]
            embeddings = embeddings.mean(dim=1)  # [B, D]
        elif self.text_encoder_name.startswith("microsoft"):
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            last_hidden_state = outputs.hidden_states[-1]  # [B, L, D]
            embeddings = last_hidden_state.masked_fill(
                ~rearrange(attention_mask, "... -> ... 1").bool(), 0.0
            )
            embeddings = embeddings.mean(dim=1)  # [B, D]

        elif self.text_encoder_name.startswith("meta-llama"):
            # embeddings are the mean of last hidden states
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            last_hidden_state = outputs.hidden_states[-1]  # [B, L, D]
            embeddings = last_hidden_state.masked_fill(
                ~rearrange(attention_mask, "... -> ... 1").bool(), 0.0
            )
            embeddings = embeddings.mean(dim=1)  # [B, D]
        else:
            # embeddings are the class token embeddings
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state[:, 0, :]  # [B, D]

        # projection layer when using contrastive learning model
        if self.clip_model is not None:  # contrastive learning model
            self.clip_model.text_proj = self.clip_model.text_proj.to(device)
            embeddings = self.clip_model.text_proj(embeddings)

        return embeddings

    def get_text_embeds(
        self, batch_text: List[str], cond_drop_prob: float, device: str
    ):
        batch_size = len(batch_text)
        self.text_emb.to(device)
        self.null_text_embeds.to(device)
        # get text embeddings from text_encoder
        text_embeds = self.text_encode(batch_text, device)  # [B, D]
        # drop embeddings with cond_drop_prob
        text_keep_mask = prob_mask_like(
            (batch_size), 1.0 - cond_drop_prob, device
        )  # [B]
        text_embeds = torch.where(
            text_keep_mask[:, None],
            text_embeds,
            self.null_text_embeds.repeat(batch_size, 1),
        )  # [B, D]
        # last embedding to match dimension of hidden_dim
        text_embeds = self.text_emb(text_embeds)  # [B, hidden_dim]
        return text_embeds
