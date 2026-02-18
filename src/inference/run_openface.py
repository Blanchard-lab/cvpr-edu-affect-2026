import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import cv2
import sys


def load_image_rgb(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_tensor_batch(cropped_faces, device: str):
    batch_tensors = []
    for face in cropped_faces:
        t = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
        batch_tensors.append(t)
    if not batch_tensors:
        return None
    return torch.stack(batch_tensors).to(device)


def load_mlt_model(device: str, weights_path: str):
    sys.path.append("/s/babbage/h/nobackup/nblancha/public-datasets/sifat/OpenFace-3.0")
    from model.MLT import MLT

    model = MLT()
    state = torch.load(weights_path, map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()

    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    return model


def softmax_np(x: np.ndarray):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument(
        "--weights_path",
        default="/s/babbage/h/nobackup/nblancha/public-datasets/sifat/OpenFace-3.0/weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth",
    )
    ap.add_argument("--drop_missing_frames", action="store_true")
    args = ap.parse_args()

    device = args.device
    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = load_mlt_model(device=device, weights_path=args.weights_path)

    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    k = args.k_frames
    frame_cols = [f"frame_{i:02d}" for i in range(1, k + 1)]

    fieldnames_out = list(rows[0].keys()) + [
        "pred_label",
        "pred_idx",
        "neutral_rate",
        "used_frames",
    ]
    
    emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_out)
        w.writeheader()

        for row in rows:
            img_paths = [row.get(c, "") for c in frame_cols]
            img_paths = [p for p in img_paths if p]

            faces = []
            for p in img_paths:
                img = load_image_rgb(p)
                if img is None:
                    continue
                faces.append(img)

            if not faces:
                if args.drop_missing_frames:
                    continue
                out_row = dict(row)
                out_row.update(
                    {
                        "pred_label": "",
                        "pred_idx": "",
                        "neutral_rate": "",
                        "used_frames": 0,
                    }
                )
                w.writerow(out_row)
                continue

            batch = to_tensor_batch(faces, device=device)
            with torch.no_grad():
                emotion_logits, _, _, _ = model(batch)

            # emotion_logits: [N, C]
            logits_np = emotion_logits.detach().cpu().numpy()
            probs = np.apply_along_axis(softmax_np, 1, logits_np)
            mean_prob = probs.mean(axis=0)
            pred_idx = int(np.argmax(mean_prob))
            pred_label = emotion_classes[pred_idx] if pred_idx < len(emotion_classes) else str(pred_idx)

            # neutral collapse signal
            neutral_idx = emotion_classes.index("Neutral") if "Neutral" in emotion_classes else 0
            neutral_rate = float((np.argmax(probs, axis=1) == neutral_idx).mean())

            out_row = dict(row)
            out_row.update(
                {
                    "pred_label": pred_label,
                    "pred_idx": pred_idx,
                    "neutral_rate": neutral_rate,
                    "used_frames": len(faces),
                }
            )
            w.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
