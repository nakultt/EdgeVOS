# scripts/test_checkpoint.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, '.')

from models.backbone import BackboneWithRoI
from models.mask_head import MLPMaskHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
ckpt = torch.load('checkpoints/best_distill.pt', map_location=device)
print(f"Checkpoint epoch : {ckpt['epoch']}")
print(f"Checkpoint loss  : {ckpt['loss']:.4f}")

# Build model
backbone  = BackboneWithRoI().to(device).eval()
mask_head = MLPMaskHead().to(device).eval()

# Load weights
state = ckpt['model']
backbone_state  = {k.replace('backbone.',''):v for k,v in state.items() if k.startswith('backbone')}
maskhead_state  = {k.replace('mask_head.',''):v for k,v in state.items() if k.startswith('mask_head')}
backbone.load_state_dict(backbone_state,  strict=False)
mask_head.load_state_dict(maskhead_state, strict=False)

# Pick a random cached target to test with
targets = list(Path('data/sam3_targets/train2017').glob('*.pt'))
t = torch.load(targets[100], weights_only=True)

print(f"\nTeacher logit range : {t['soft_logits'].float().min():.2f} / {t['soft_logits'].float().max():.2f}")
print(f"Box                 : {t['boxes']}")

# Find the actual COCO image
from pycocotools.coco import COCO
coco = COCO('data/COCO/annotations/instances_train2017.json')
ann  = coco.loadAnns(int(t['ann_id']))[0]
img_info = coco.loadImgs(ann['image_id'])[0]
img_path = Path('data/COCO/train2017') / img_info['file_name']

img = Image.open(img_path).convert('RGB')
img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).to(device)

bbox = t['boxes'].to(device)

# Forward pass
with torch.no_grad():
    roi_feat = backbone.get_roi_feature(img_tensor, bbox.unsqueeze(0))
    pred_logits = mask_head(roi_feat)  # [1, 1, 256, 256]

pred_mask    = pred_logits.sigmoid().squeeze().cpu().numpy()
teacher_mask = t['soft_logits'].float().sigmoid().squeeze().numpy()

print(f"\nPred   mask — min/max : {pred_mask.min():.3f} / {pred_mask.max():.3f}")
print(f"Teacher mask — min/max: {teacher_mask.min():.3f} / {teacher_mask.max():.3f}")

# IoU between student and teacher (binarised at 0.5)
pred_bin    = pred_mask > 0.5
teacher_bin = teacher_mask > 0.5
inter = (pred_bin & teacher_bin).sum()
union = (pred_bin | teacher_bin).sum()
iou   = inter / (union + 1e-6)
print(f"Student vs Teacher IoU : {iou:.3f}  (target after 30 epochs: >0.90)")

# Save visual comparison
H, W = np.array(img).shape[:2]
orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

pred_vis    = cv2.resize((pred_mask    * 255).astype(np.uint8), (W, H))
teacher_vis = cv2.resize((teacher_mask * 255).astype(np.uint8), (W, H))

pred_color    = cv2.applyColorMap(pred_vis,    cv2.COLORMAP_JET)
teacher_color = cv2.applyColorMap(teacher_vis, cv2.COLORMAP_JET)

# Draw bbox on original
x1,y1,x2,y2 = [int(v * max(H,W) / 224) for v in bbox.cpu().tolist()]
orig_box = orig.copy()
cv2.rectangle(orig_box, (x1,y1), (x2,y2), (0,255,0), 2)

combined = np.hstack([orig_box, teacher_color, pred_color])
cv2.imwrite('test_output.jpg', combined)
print(f"\nSaved: test_output.jpg")
print("Left=original+box | Middle=SAM3.1 teacher | Right=student (2 epochs)")
