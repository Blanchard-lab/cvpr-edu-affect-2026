import argparse
import csv
import os
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import onnxruntime


IDX_TO_CLASS = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Neutral",
    6: "Sadness",
    7: "Surprise",
}


def preprocess_face_for_onnx(face_bgr, img_size=224):
    face_img = cv2.resize(face_bgr, (img_size, img_size))
    face_img = face_img.astype(np.float32) / 255.0
    face_img[:, :, 0] = (face_img[:, :, 0] - 0.485) / 0.229
    face_img[:, :, 1] = (face_img[:, :, 1] - 0.456) / 0.224
    face_img[:, :, 2] = (face_img[:, :, 2] - 0.406) / 0.225
    img_tensor = face_img.transpose(2, 0, 1)[np.newaxis, ...]
    return img_tensor


def safe_read_bgr(path: str):
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--onnx_path", required=True)
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--providers", default="cuda,cpu")
    ap.add_argument("--drop_missing", action="store_true")
    args = ap.parse_args()

    providers = []
    for p in args.providers.split(","):
        p = p.strip().lower()
        if p == "cuda":
            providers.append("CUDAExecutionProvider")
        elif p == "cpu":
            providers.append("CPUExecutionProvider")

    session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    k = args.k_frames
    frame_cols = [f"frame_{i:02d}" for i in range(1, k + 1)]

    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    extra_cols = [
        "valence_mean",
        "arousal_mean",
        "valence_std",
        "arousal_std",
        "emotion_mode",
        "used_frames",
    ]
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
                for c in extra_cols:
                    out_row[c] = ""
                out_row["used_frames"] = 0
                w.writerow(out_row)
                continue

            vals = []
            aros = []
            emos = []

            for p in paths:
                img = safe_read_bgr(p)
                if img is None:
                    continue

                x = preprocess_face_for_onnx(img, img_size=args.img_size)
                outs = session.run(output_names, {input_name: x})

                emotion_logits = outs[1]
                emo = IDX_TO_CLASS[int(np.argmax(emotion_logits))]
                v = float(np.asarray(outs[2]).reshape(-1)[0])
                a = float(np.asarray(outs[3]).reshape(-1)[0])

                emos.append(emo)
                vals.append(v)
                aros.append(a)

            used = len(vals)

            if used == 0:
                if args.drop_missing:
                    continue
                out_row = dict(row)
                for c in extra_cols:
                    out_row[c] = ""
                out_row["used_frames"] = 0
                w.writerow(out_row)
                continue

            v_mean = float(np.mean(vals))
            a_mean = float(np.mean(aros))
            v_std = float(np.std(vals))
            a_std = float(np.std(aros))

            mode_emo = ""
            if emos:
                c = Counter(emos)
                mode_emo = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

            out_row = dict(row)
            out_row.update(
                {
                    "valence_mean": v_mean,
                    "arousal_mean": a_mean,
                    "valence_std": v_std,
                    "arousal_std": a_std,
                    "emotion_mode": mode_emo,
                    "used_frames": used,
                }
            )
            w.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()