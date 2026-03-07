import argparse
import csv
import random
from pathlib import Path


LABEL_MAP = {
    "anger": "Angry",
    "contempt": "Contempt",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "happiness": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "sadness": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
}


def normalize_label(x: str) -> str:
    x = str(x).strip().lower()
    return LABEL_MAP.get(x, x.title())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--affectnet_root", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    affectnet_root = Path(args.affectnet_root)
    labels_csv = Path(args.labels_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    valid = []
    for row in rows:
        rel_path = row["pth"].strip()
        img_path = affectnet_root / "Train" / rel_path
        if not img_path.exists():
            continue
        label = normalize_label(row["label"])
        valid.append(
            {
                "image_path": str(img_path),
                "gt_label": label,
                "rel_path": rel_path,
            }
        )

    if len(valid) < args.n_samples:
        raise ValueError(f"Only found {len(valid)} valid samples, fewer than requested {args.n_samples}")

    sampled = rng.sample(valid, args.n_samples)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "gt_label", "rel_path"])
        writer.writeheader()
        for row in sampled:
            writer.writerow(row)

    print(f"Wrote: {out_csv} ({len(sampled)} samples)")


if __name__ == "__main__":
    main()