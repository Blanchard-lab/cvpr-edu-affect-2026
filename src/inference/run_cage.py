import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms


def build_model(device: torch.device, weights_path: str):
    model = models.maxvit_t(weights="DEFAULT")
    block_channels = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(block_channels),
        nn.Linear(block_channels, block_channels),
        nn.Tanh(),
        nn.Linear(block_channels, 10, bias=False),
    )

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

    model.eval()
    model.to(device)
    return model


def load_rgb(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--weights_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--drop_missing", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = build_model(device, args.weights_path)

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    k = args.k_frames
    frame_cols = [f"frame_{i:02d}" for i in range(1, k + 1)]

    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    extra_cols = ["valence_mean", "arousal_mean", "valence_std", "arousal_std", "used_frames"]
    fieldnames_out = list(rows[0].keys()) + extra_cols

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_out)
        w.writeheader()

        for row in rows:
            paths = [row.get(c, "") for c in frame_cols]
            paths = [p for p in paths if p and os.path.exists(p)]

            if not paths:
                if args.drop_missing:
                    continue
                out_row = dict(row)
                out_row.update({c: "" for c in extra_cols})
                out_row["used_frames"] = 0
                w.writerow(out_row)
                continue

            tensors = []
            for p in paths:
                img = load_rgb(p)
                if img is None:
                    continue
                tensors.append(tfm(img))

            if not tensors:
                if args.drop_missing:
                    continue
                out_row = dict(row)
                out_row.update({c: "" for c in extra_cols})
                out_row["used_frames"] = 0
                w.writerow(out_row)
                continue

            X = torch.stack(tensors, dim=0)
            used = X.shape[0]

            vals = []
            aros = []

            bs = min(args.batch_size, used)
            with torch.no_grad():
                for i in range(0, used, bs):
                    xb = X[i:i + bs].to(device)
                    out = model(xb)
                    v = out[:, 7].detach().float().cpu().numpy()
                    a = out[:, 8].detach().float().cpu().numpy()
                    vals.append(v)
                    aros.append(a)

            vals = np.concatenate(vals, axis=0) if vals else np.array([], dtype=np.float32)
            aros = np.concatenate(aros, axis=0) if aros else np.array([], dtype=np.float32)

            v_mean = float(np.mean(vals)) if vals.size else ""
            a_mean = float(np.mean(aros)) if aros.size else ""
            v_std = float(np.std(vals)) if vals.size else ""
            a_std = float(np.std(aros)) if aros.size else ""

            out_row = dict(row)
            out_row.update({
                "valence_mean": v_mean if v_mean != "" else "",
                "arousal_mean": a_mean if a_mean != "" else "",
                "valence_std": v_std if v_std != "" else "",
                "arousal_std": a_std if a_std != "" else "",
                "used_frames": used,
            })
            w.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()