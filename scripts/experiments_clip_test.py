import torch
import numpy as np
from PIL import Image
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained="dfndr2b")
model = model.to(device).eval()

# Dummy crop tensor [3, 100, 100] with range [0, 1]
crop = torch.rand(3, 100, 100)
crop_np = crop.permute(1, 2, 0).numpy()

# Option A: Passing numpy float32 directly
try:
    input_a = preprocess(crop_np).unsqueeze(0).to(device)
    feat_a = model.encode_image(input_a)
    print("Option A (numpy) succeeded! Shape:", feat_a.shape)
except Exception as e:
    print("Option A (numpy) failed with error:", e)

# Option B: Convert to PIL image first
try:
    crop_pil = Image.fromarray((crop_np * 255).astype(np.uint8))
    input_b = preprocess(crop_pil).unsqueeze(0).to(device)
    feat_b = model.encode_image(input_b)
    print("Option B (PIL) succeeded! Shape:", feat_b.shape)
    
    # Check similarity
    sim = torch.cosine_similarity(feat_a, feat_b).item()
    print("Cosine similarity between Option A and Option B features:", sim)
except Exception as e:
    print("Option B (PIL) failed with error:", e)
