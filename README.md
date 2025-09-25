# 🖼️ Image Captioning with Mask R-CNN + Transformer

This project implements an end-to-end **image captioning system** using:
- **Mask R-CNN** for object detection & region feature extraction  
- **ResNet backbone** for feature encoding  
- **Transformer decoder** for caption generation  
- Trained & tested on **Flickr30k dataset**

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-name>


project/
│── src/
│   ├── extract_features.py
│   ├── train.py
│   ├── infer.py
│   └── utils/...
│
│── outputs/                # Features, checkpoints, results
│── requirements.txt
│── README.md
│── .gitignore
│── venv/                   # (not pushed to git)


# Extract features
python extract_features.py --split train --max-images 200 --max-regions 36 --out outputs/features.npz

# Train model
python train.py --features outputs/features.npz --out outputs/checkpoints --epochs 5

# Caption an image
python infer.py --checkpoint outputs/checkpoints/model_epoch5.pth --checkpoint-dir outputs/checkpoints --image example.jpg
