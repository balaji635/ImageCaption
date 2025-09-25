#!/usr/bin/env python3
"""
Transformer Decoder caption model for region features
"""
import torch
import torch.nn as nn
import numpy as np

class TransformerCaptionModel(nn.Module):
    def __init__(self, feat_dim=2048, num_regions=36, d_model=512, nhead=8, num_layers=4, vocab_size=30000, max_len=30):
        super().__init__()
        self.num_regions = num_regions
        self.d_model = d_model
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_regions, d_model))
        self.token_emb = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, region_feats, tgt_seq):
        # region_feats: [B, R, D]
        # tgt_seq: [B, T]
        B = region_feats.size(0)
        R = region_feats.size(1)
        enc = self.feat_proj(region_feats) + self.pos_embed[:, :R, :]  # [B, R, d]
        enc = enc.permute(1,0,2)  # [R, B, d]
        tgt_emb = self.token_emb(tgt_seq) * np.sqrt(self.d_model)  # [B, T, d]
        tgt_emb = tgt_emb.permute(1,0,2)  # [T, B, d]
        tgt_len = tgt_emb.size(0)
        device = region_feats.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        out = self.transformer_decoder(tgt_emb, enc, tgt_mask=tgt_mask)
        out = out.permute(1,0,2)  # [B, T, d]
        logits = self.fc_out(out)  # [B, T, V]
        return logits
