import argparse
import csv
from pathlib import Path

import libreface


LABEL_MAP = {
    "Happiness": "Happy",
    "Sadness": "Sad",
    "Anger": "Angry",
    "Neutral": "Neutral",
    "Surprise": "Surprise",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Contempt": "Contempt",
}


def normalize_label(x: str) -> str:
    x = str(x).strip()
    return LABEL_MAP.get(x, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with index_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["pred_label"]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            try:
                attrs = libreface.get_facial_attributes(row["image_path"])
                pred = normalize_label(attrs.get("facial_expression", ""))
            except Exception:
                pred = ""

            out_row = dict(row)
            out_row["pred_label"] = pred
            writer.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()