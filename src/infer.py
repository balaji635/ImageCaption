#!/usr/bin/env python3
"""
Generate caption for a user-provided image using Mask R-CNN + ResNet features
and a trained Transformer captioning model.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from model import TransformerCaptionModel
from extract_features import extract_region_features_from_pil

def greedy_decode(model, tokenizer, region_feats, max_len=30, device=torch.device('cpu')):
    model.eval()
    region_feats = torch.from_numpy(region_feats).unsqueeze(0).float().to(device)  # [1,R,D]
    start = tokenizer.convert_tokens_to_ids('<start>')
    end = tokenizer.convert_tokens_to_ids('<end>')
    pad = tokenizer.pad_token_id
    generated = [start]
    with torch.no_grad():
        for t in range(max_len-1):
            inp = torch.tensor(generated).unsqueeze(0).to(device)  # [1, L]
            logits = model(region_feats, inp)  # [1, L, V]
            next_logits = logits[0, -1]  # [V]
            next_id = int(torch.argmax(next_logits).item())
            generated.append(next_id)
            if next_id == end:
                break
    txt = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return txt

def main(args):
    device = torch.device('cpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # Build detector + resnet encoder
    detector = maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()

    resnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

    transform_crop = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load user image
    pil_image = Image.open(args.image).convert("RGB")

    # Extract features
    region_feats = extract_region_features_from_pil(
        pil_image, detector, resnet, transform_crop,
        max_regions=args.max_regions, device=device
    )

    # Build model (must match training config)
    vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
    feat_dim = region_feats.shape[-1]
    num_regions = region_feats.shape[0]

    model = TransformerCaptionModel(feat_dim=feat_dim, num_regions=num_regions,
                                    d_model=args.d_model, nhead=args.nhead,
                                    num_layers=args.num_layers, vocab_size=vocab_size,
                                    max_len=args.max_len)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    # Generate caption
    caption = greedy_decode(model, tokenizer, region_feats, max_len=args.max_len, device=device)
    print("Generated caption:")
    print(caption)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory where tokenizer was saved")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--max-regions", type=int, default=36)
    parser.add_argument("--d-model", dest="d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()
    main(args)
