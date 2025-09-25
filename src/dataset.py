#!/usr/bin/env python3
"""
Dataset wrappers for region features + captions
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class RegionCaptionDataset(Dataset):
    def __init__(self, npz_file, tokenizer, max_len=30):
        data = np.load(npz_file, allow_pickle=True)
        self.features = data['features']  # [N, R, D]
        self.captions = data['captions']  # array of strings
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return int(self.features.shape[0])

    def encode_caption(self, caption):
        # tokenizer should already have special tokens '<start>' and '<end>'
        toks = self.tokenizer.encode(caption, add_special_tokens=False)
        start = self.tokenizer.convert_tokens_to_ids('<start>')
        end = self.tokenizer.convert_tokens_to_ids('<end>')
        pad = self.tokenizer.pad_token_id
        seq = [start] + toks[:(self.max_len-2)] + [end]
        if len(seq) < self.max_len:
            seq += [pad]*(self.max_len - len(seq))
        return np.array(seq, dtype=np.int64)

    def __getitem__(self, idx):
        feats = self.features[idx].astype('float32')  # [R, D]
        cap = self.captions[idx].item() if hasattr(self.captions[idx], 'item') else self.captions[idx]
        seq = self.encode_caption(cap)
        return torch.from_numpy(feats), torch.from_numpy(seq)
