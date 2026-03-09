import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

class RecorderMeter1:
    pass

class RecorderMeter:
    pass

def parse_labels(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def majority_vote(labels, label_order):
    if not labels:
        return "", -1
    counts = Counter(labels)
    best_count = max(counts.values())
    tied = [lab for lab, c in counts.items() if c == best_count]
    for lab in label_order:
        if lab in tied:
            return lab, label_order.index(lab)
    tied = sorted(tied)
    return tied[0], label_order.index(tied[0]) if tied[0] in label_order else -1


def load_state(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--poster_root", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--labels", default="Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise")
    ap.add_argument("--drop_missing_frames", action="store_true")
    args = ap.parse_args()

    labels = parse_labels(args.labels)
    if len(labels) != 7:
        raise ValueError(f"--labels must contain exactly 7 classes, got {len(labels)}: {labels}")

    sys.path.append(args.poster_root)
    from models.PosterV2_7cls import pyramid_trans_expr2

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    model = load_state(model, args.checkpoint_path, device)
    model.eval()
    model = model.to(device)

    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with index_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    frame_cols = [f"frame_{i:02d}" for i in range(1, args.k_frames + 1)]
    fieldnames_out = list(rows[0].keys()) + ["pred_label", "pred_idx", "neutral_rate", "used_frames"]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()

        for row in rows:
            paths = [row.get(c, "") for c in frame_cols]
            paths = [p for p in paths if p and os.path.exists(p)]

            tensors = []
            for p in paths:
                img = load_image(p)
                if img is None:
                    continue
                tensors.append(tfm(img))

            if not tensors:
                if args.drop_missing_frames:
                    continue
                out_row = dict(row)
                out_row.update({
                    "pred_label": "",
                    "pred_idx": "",
                    "neutral_rate": "",
                    "used_frames": 0,
                })
                writer.writerow(out_row)
                continue

            X = torch.stack(tensors, dim=0)
            used = X.shape[0]

            preds = []
            bs = min(args.batch_size, used)

            with torch.no_grad():
                for i in range(0, used, bs):
                    xb = X[i:i + bs].to(device)
                    out = model(xb)
                    pred_idx = torch.argmax(out, dim=1).detach().cpu().numpy().tolist()
                    preds.extend(pred_idx)

            pred_labels = [labels[i] for i in preds]
            pred_label, pred_idx = majority_vote(pred_labels, labels)

            neutral_rate = ""
            if "Neutral" in labels and pred_labels:
                neutral_rate = sum(1 for x in pred_labels if x == "Neutral") / float(len(pred_labels))

            out_row = dict(row)
            out_row.update({
                "pred_label": pred_label,
                "pred_idx": pred_idx,
                "neutral_rate": neutral_rate,
                "used_frames": used,
            })
            writer.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()