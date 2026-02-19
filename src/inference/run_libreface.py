import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


def majority_vote(labels):
    if not labels:
        return ""
    c = Counter(labels)
    top = c.most_common(1)[0][0]
    return top


def normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    return s


def libreface_predict_image(path: str) -> str:
    import libreface
    attrs = libreface.get_facial_attributes(path, device="cuda")
    return normalize_label(attrs.get("facial_expression", ""))


def libreface_predict_array_rgb(img_rgb_224) -> str:
    """
    LibreFace API expects a file path in common usage. We'll write a temp jpg.
    img_rgb_224: numpy array (224,224,3) RGB
    """
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(img_rgb_224, cv2.COLOR_RGB2BGR))
        return libreface_predict_image(tmp.name)


def load_crop_as_rgb_224(path: str):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument("--drop_missing_frames", action="store_true")
    ap.add_argument("--use_tempfile", action="store_true")
    args = ap.parse_args()

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    k = args.k_frames
    frame_cols = [f"frame_{i:02d}" for i in range(1, k + 1)]

    fieldnames_out = list(rows[0].keys()) + [
        "pred_label",
        "neutral_rate",
        "used_frames",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_out)
        w.writeheader()

        for row in rows:
            paths = [row.get(c, "") for c in frame_cols]
            paths = [p for p in paths if p and os.path.exists(p)]

            if not paths:
                if args.drop_missing_frames:
                    continue
                out_row = dict(row)
                out_row.update({"pred_label": "", "neutral_rate": "", "used_frames": 0})
                w.writerow(out_row)
                continue

            preds = []
            for p in paths:
                try:
                    if args.use_tempfile:
                        img = load_crop_as_rgb_224(p)
                        if img is None:
                            continue
                        pred = libreface_predict_array_rgb(img)
                    else:
                        pred = libreface_predict_image(p)
                    pred = normalize_label(pred)
                    if pred:
                        preds.append(pred)
                except Exception:
                    continue

            if not preds:
                if args.drop_missing_frames:
                    continue
                out_row = dict(row)
                out_row.update({"pred_label": "", "neutral_rate": "", "used_frames": len(paths)})
                w.writerow(out_row)
                continue

            pred_label = majority_vote(preds)

            neutral_rate = ""
            neut_key = "neutral"
            neut_hits = sum(1 for p in preds if p.lower() == neut_key)
            neutral_rate = neut_hits / float(len(preds)) if preds else ""

            out_row = dict(row)
            out_row.update(
                {
                    "pred_label": pred_label,
                    "neutral_rate": neutral_rate,
                    "used_frames": len(preds),
                }
            )
            w.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
