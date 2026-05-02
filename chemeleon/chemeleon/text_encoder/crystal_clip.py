from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from transformers import BertModel, BertTokenizer

from chemeleon.text_encoder import MODEL_NAMES
from chemeleon.modules.cspnet import CSPNet
from chemeleon.utils.scatter import scatter_mean, scatter_sum


class CrystalClip(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters(_config)
        self.clip_dim = _config["clip_dim"]
        self.label_smoothing = _config["label_smoothing"]
        # text encoder
        self.text_encoder_name = _config["text_encoder"]
        self.max_text_len = _config["max_text_len"]
        self.text_embed_dim = _config["text_embed_dim"]
        assert (
            self.text_encoder_name in MODEL_NAMES
        ), f"Invalid model name. Must be one of {MODEL_NAMES}"
        self.tokenizer = BertTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = BertModel.from_pretrained(self.text_encoder_name).to(
            self.device
        )
        self.text_encoder.train()

        # graph encoder
        _config["time_dim"] = 0
        _config["text_dim"] = 0
        assert _config["time_dim"] == 0 and _config["text_dim"] == 0
        self.graph_encoder = CSPNet(
            hidden_dim=_config["hidden_dim"],
            time_dim=_config["time_dim"],
            text_dim=_config["text_dim"],
            num_layers=_config["num_layers"],
            max_atoms=_config["max_atoms"],
            act_fn=_config["act_fn"],
            dis_emb=_config["dis_emb"],
            num_freqs=_config["num_freqs"],
            edge_style=_config["edge_style"],
            cutoff=_config["cutoff"],
            max_neighbors=_config["max_neighbors"],
            ln=_config["ln"],
            ip=_config["ip"],
            smooth=_config["smooth"],
            pred_atom_types=_config["pred_atom_types"],
        )
        self.graph_pooling = _config["graph_pooling"]
        self.graph_embed_dim = _config["hidden_dim"]
        if self.graph_pooling == "mean":
            self.pool_fn = scatter_mean
        elif self.graph_pooling == "sum":
            self.pool_fn = scatter_sum
        # projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.text_embed_dim),
            nn.LayerNorm(self.text_embed_dim),
            nn.GELU(),
            nn.Linear(self.text_embed_dim, self.clip_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_embed_dim, self.graph_embed_dim),
            nn.LayerNorm(self.graph_embed_dim),
            nn.GELU(),
            nn.Linear(self.graph_embed_dim, self.clip_dim),
        )

        # optimizer
        self.lr = _config["lr"]
        self.graph_encoder_lr = _config["graph_encoder_lr"]
        self.text_encoder_lr = _config["text_encoder_lr"]
        self.weight_decay = _config["weight_decay"]
        self.patience = _config["patience"]

    def get_text_embeds(self, text: List[str]):
        tokenized = self.tokenizer.batch_encode_plus(
            text,
            padding="longest",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",  # Returns torch.tensor instead of python integers
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        class_token_embeds = outputs.last_hidden_state[:, 0, :]  # [B, text_embed_dim]
        text_embeds = self.text_proj(class_token_embeds)  # [B, clip_dim]
        return text_embeds

    def get_graph_embeds(self, batch: Batch):
        outputs = self.graph_encoder(
            t=None,
            atom_types=batch.atom_types,
            frac_coords=batch.frac_coords,
            lattices=batch.lattices,
            num_atoms=batch.natoms,
            node2graph=batch.batch,
        )
        node_features = outputs.node_features  # [B_n, hidden_dim]
        graph_features = self.pool_fn(
            node_features, batch.batch, dim=0
        )  # [B, hidden_dim]
        graph_embeds = self.graph_proj(graph_features)  # [B, clip_dim]
        return graph_embeds

    def forward(self, batch: Batch):
        # text encoder
        text_embeds = self.get_text_embeds(batch.text)  # [B, clip_dim]
        # graph encoder
        graph_embeds = self.get_graph_embeds(batch)  # [B, clip_dim]
        return text_embeds, graph_embeds

    def compute_contrastive_loss(
        self, text_embeds: torch.Tensor, graph_embeds: torch.Tensor
    ):
        # gather all embeddings
        all_text_embeds = self.all_gather(text_embeds, sync_grads=True).view(
            -1, self.clip_dim
        )  # [B * k, clip_dim]
        all_graph_embeds = self.all_gather(graph_embeds, sync_grads=True).view(
            -1, self.clip_dim
        )  # [B * k, clip_dim]
        # get targets
        graph_similarity = all_graph_embeds @ all_graph_embeds.T  # [B * k, B * k]
        text_similarity = all_text_embeds @ all_text_embeds.T  # [B * k, B * k]
        all_targets = F.softmax((graph_similarity + text_similarity) / 2, dim=-1)
        # get logits
        all_logits = all_text_embeds @ all_graph_embeds.T  # [B * k, B * k]
        # calculate loss
        graph_loss = F.cross_entropy(
            all_logits.T,
            all_targets.argmax(dim=-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # [B * k]
        text_loss = F.cross_entropy(
            all_logits,
            all_targets.argmax(dim=0),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # [B * k]
        loss = (graph_loss + text_loss) / 2  # [B * k]
        loss = loss.mean()
        return loss

    def training_step(self, batch: Batch, *args, **kwargs):
        text_embeds, graph_embeds = self.forward(batch)
        loss = self.compute_contrastive_loss(text_embeds, graph_embeds)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Batch, *args, **kwargs):
        text_embeds, graph_embeds = self.forward(batch)
        loss = self.compute_contrastive_loss(text_embeds, graph_embeds)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch: Batch, *args, **kwargs):
        text_embeds, graph_embeds = self.forward(batch)
        loss = self.compute_contrastive_loss(text_embeds, graph_embeds)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        parameters = [
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {"params": self.graph_encoder.parameters(), "lr": self.graph_encoder_lr},
            {"params": self.text_proj.parameters(), "lr": self.lr},
            {"params": self.graph_proj.parameters(), "lr": self.lr},
        ]
        optimizer = torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", min_lr=1e-6, factor=0.8, patience=self.patience
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning rate",
            "monitor": "val/loss",
        }

        return ([optimizer], [lr_scheduler])
