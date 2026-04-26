import torch
from pathlib import Path
from tqdm import tqdm

def main():
    target_dir = Path('data/sam3_targets/train2017')
    files = list(target_dir.glob('*.pt'))
    print(f"Patching {len(files)} files...")

    for f in tqdm(files):
        t = torch.load(f, map_location="cpu", weights_only=False)
        t['soft_logits'] = t['soft_logits'].float().clamp(-6, 6).half()
        torch.save(t, f)

    if files:
        t = torch.load(files[0], map_location="cpu", weights_only=False)
        logits = t['soft_logits'].float()
        print(f"Done. min/max: {logits.min():.2f} / {logits.max():.2f}")

if __name__ == "__main__":
    main()
