#!/usr/bin/env python3
"""
Train the TransformerCaptionModel on extracted region features.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import RegionCaptionDataset
from model import TransformerCaptionModel
import numpy as np
from tqdm import tqdm

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    # ensure special tokens exist
    added = False
    special_tokens = []
    if '<start>' not in tokenizer.get_vocab():
        special_tokens.append('<start>')
    if '<end>' not in tokenizer.get_vocab():
        special_tokens.append('<end>')
    if special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        added = True

    ds = RegionCaptionDataset(args.features, tokenizer, max_len=args.max_len)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    sample_feats = ds[0][0]
    feat_dim = sample_feats.shape[1]
    num_regions = sample_feats.shape[0]

    vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    model = TransformerCaptionModel(feat_dim=feat_dim, num_regions=num_regions,
                                    d_model=args.d_model, nhead=args.nhead,
                                    num_layers=args.num_layers, vocab_size=vocab_size,
                                    max_len=args.max_len)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        it = 0
        for feats, seqs in tqdm(dataloader, desc=f"Epoch {epoch}"):
            feats = feats.to(device)                # [B, R, D]
            seqs = seqs.to(device)                  # [B, T]
            optimizer.zero_grad()
            input_tokens = seqs[:, :-1]             # [B, T-1]
            target_tokens = seqs[:, 1:]             # [B, T-1]
            logits = model(feats, input_tokens)     # [B, T-1, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            it += 1
        avg = total_loss / max(1, it)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")
        ckpt_path = os.path.join(args.out, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved", ckpt_path)
        # save tokenizer too (so special tokens persist)
        tokenizer.save_pretrained(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to .npz features file")
    parser.add_argument("--out", default="outputs/rcnn_transformer_checkpoints", help="Checkpoint output dir")
    parser.add_argument("--tokenizer_model", default="bert-base-uncased", help="HF tokenizer model to base on")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--d-model", dest="d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()
    main(args)
