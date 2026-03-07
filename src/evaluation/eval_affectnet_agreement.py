import argparse
import csv
from pathlib import Path

import numpy as np


LABELS = ["Neutral", "Surprise", "Disgust", "Happy", "Contempt", "Sad", "Angry", "Fear"]


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return row["image_path"]


def build_matrix(openface_rows, libreface_rows):
    of_map = {key(r): r for r in openface_rows}
    lf_map = {key(r): r for r in libreface_rows}

    shared = sorted(set(of_map.keys()) & set(lf_map.keys()))
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

    for k in shared:
        a = of_map[k].get("pred_label", "").strip()
        b = lf_map[k].get("pred_label", "").strip()
        if a not in LABELS or b not in LABELS:
            continue
        i = LABELS.index(a)
        j = LABELS.index(b)
        mat[i, j] += 1

    return mat, shared


def row_normalize(mat):
    mat = mat.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def write_matrix_csv(path: Path, mat, row_labels, col_labels, float_fmt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(col_labels))
        for i, row_lab in enumerate(row_labels):
            row = [row_lab]
            for val in mat[i]:
                row.append(f"{val:.4f}" if float_fmt else int(val))
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openface_csv", required=True)
    ap.add_argument("--libreface_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    of_rows = load_csv(Path(args.openface_csv))
    lf_rows = load_csv(Path(args.libreface_csv))

    mat_raw, shared = build_matrix(of_rows, lf_rows)
    mat_norm = row_normalize(mat_raw)

    write_matrix_csv(out_dir / "cm_openface_vs_libreface_affectnet_raw.csv", mat_raw, LABELS, LABELS, float_fmt=False)
    write_matrix_csv(out_dir / "cm_openface_vs_libreface_affectnet_norm.csv", mat_norm, LABELS, LABELS, float_fmt=True)

    print(f"Wrote AffectNet agreement matrices to: {out_dir}")
    print(f"Shared samples: {len(shared)}")


if __name__ == "__main__":
    main()