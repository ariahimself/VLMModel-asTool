"""
vlm_model.py

A minimal "real" VLM for MNIST:
- Vision encoder: your pretrained ViT (from main2.py), used to extract a CLS embedding.
- Text decoder: a small TransformerDecoder that generates the answer text.
- Conditioning: the decoder cross-attends to a single "memory token" coming from the image embedding.

Training objective:
Given text: "question: ...\nanswer: " and an image, the decoder learns to generate the answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from main2 import VisionTransformer


@dataclass
class VLMConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_len: int = 256


class ViTEncoder(nn.Module):
    """
    Wraps your VisionTransformer and exposes a feature embedding (CLS after norm).
    """
    def __init__(self, vit: VisionTransformer):
        super().__init__()
        self.vit = vit

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,28,28)
        x = self.vit.patch_embed(x)           # (B, patches+1, embed_dim)
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)
        cls = x[:, 0]                         # (B, embed_dim)
        return cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class MNISTVLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: VLMConfig, vit_embed_dim: int = 64):
        super().__init__()
        self.cfg = cfg

        # Vision side
        vit = VisionTransformer()
        self.vision = ViTEncoder(vit)
        self.vision_proj = nn.Linear(vit_embed_dim, cfg.d_model)

        # Text side
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.num_layers)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size)

    def load_vit_weights(self, vit_state_dict: dict):
        """
        Load pretrained weights from vit_mnist_model.pth into our internal ViT.
        """
        self.vision.vit.load_state_dict(vit_state_dict)

    def freeze_vision(self):
        for p in self.vision.parameters():
            p.requires_grad = False

    def unfreeze_vision(self):
        for p in self.vision.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        images: (B,1,28,28)
        input_ids: (B,T) decoder input tokens (teacher forcing)
        returns logits: (B,T,vocab)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Vision memory: (B,1,d_model)
        vis = self.vision.forward_features(images)                   # (B,64)
        memory = self.vision_proj(vis).unsqueeze(1)                  # (B,1,d_model)

        # Token + positional embeddings
        tok = self.token_emb(input_ids)                              # (B,T,d_model)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos = self.pos_emb(pos_ids)
        tgt = tok + pos

        # Causal mask so each position can't see the future
        # shape expected by TransformerDecoder (T,T) if batch_first=True
        causal = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()

        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)  # (B,T,d_model)
        logits = self.lm_head(out)                                   # (B,T,vocab)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        eos_id: int,
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """
        Greedy decoding. input_ids should already include BOS and the prompt part.
        Returns full sequence (prompt + generated).
        """
        self.eval()
        device = input_ids.device
        seq = input_ids

        for _ in range(max_new_tokens):
            # Keep within max_len for positional embeddings
            if seq.shape[1] >= self.cfg.max_len:
                break

            logits = self.forward(images, seq)             # (B,T,vocab)
            next_logits = logits[:, -1, :]                 # (B,vocab)
            next_id = torch.argmax(next_logits, dim=-1)    # (B,)
            seq = torch.cat([seq, next_id.unsqueeze(1)], dim=1)

            # stop if all batches hit EOS
            if torch.all(next_id == eos_id):
                break

        return seq
