#!/usr/bin/env python3
"""
Extract region features from Flickr30k using torchvision Mask R-CNN for boxes
and ResNet50 (ImageNet) to encode region crops.

Saves a compressed .npz with arrays:
- features: [N, R, D]
- captions: array of N caption strings
- image_ids: array of N image ids (or indexes)
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Image as HFImage
from PIL import Image
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models import resnet50
from torchvision import transforms


def extract_region_features_from_pil(
    pil_image, detector, resnet_encoder, transform_crop,
    max_regions=36, device=torch.device('cpu')
):
    img_np = np.array(pil_image.convert('RGB'))
    img_tensor = transforms.ToTensor()(pil_image).to(device)

    with torch.no_grad():
        preds = detector([img_tensor])[0]

    boxes = preds.get("boxes", torch.empty((0, 4))).cpu().numpy()
    masks = preds.get("masks", torch.empty((0, 1, *img_np.shape[:2]))).cpu().numpy()
    scores = preds.get("scores", torch.empty((0,))).cpu().numpy()

    order = np.argsort(-scores) if scores.size else np.array([], dtype=int)
    boxes = boxes[order][:max_regions]
    masks = masks[order][:max_regions]

    crops = []
    h, w = img_np.shape[:2]

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        if i >= max_regions:
            break
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        mask_bin = (mask[0] > 0.5).astype(np.uint8) * 255
        mask_crop = mask_bin[y1:y2, x1:x2]
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0 or mask_crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            crop = cv2.bitwise_and(crop, crop, mask=mask_crop)
            try:
                crop = cv2.resize(crop, (224, 224))
            except Exception:
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
        crop_pil = Image.fromarray(crop)
        t = transform_crop(crop_pil).unsqueeze(0).to(device)
        crops.append(t)

    while len(crops) < max_regions:
        crops.append(torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device))

    batch = torch.cat(crops[:max_regions], dim=0)

    with torch.no_grad():
        feats = resnet_encoder(batch).squeeze(-1).squeeze(-1)

    return feats.cpu().numpy()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = torch.device('cpu')
    print("Loading dataset...")

    # load and decode images directly as PIL
    ds = load_dataset("AnyModal/flickr30k")
    ds = ds.cast_column("image", HFImage(decode=True))

    split = args.split
    detector = maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    detector.to(device).eval()

    resnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device).eval()

    transform_crop = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    feats_list = []
    captions_list = []
    image_ids = []
    total = len(ds[split]) if args.max_images is None else min(args.max_images, len(ds[split]))
    print(f"Processing {total} images from split '{split}' (one entry per caption).")

    # ✅ iterate over rows properly
    for cnt, ex in enumerate(tqdm(ds[split].select(range(total)), total=total)):
        pil = ex["image"]   # already a PIL.Image
        feats = extract_region_features_from_pil(
            pil, detector, resnet, transform_crop,
            max_regions=args.max_regions, device=device
        )

        caps = ex.get("alt_text", [])
        if not isinstance(caps, list):
            caps = [caps]

        for c in caps:
            feats_list.append(feats)
            captions_list.append(c)
            image_ids.append(ex.get("img_id", cnt))

    if len(feats_list) == 0:
        print("No features extracted. Exiting.")
        return

    features = np.stack(feats_list, axis=0)
    np.savez_compressed(
        args.out,
        features=features,
        captions=np.array(captions_list, dtype=object),
        image_ids=np.array(image_ids)
    )
    print("Saved:", args.out, "shape:", features.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--max-regions", type=int, default=36)
    parser.add_argument("--out", default="outputs/flickr30k_maskrcnn_region_features.npz")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    main(args)
