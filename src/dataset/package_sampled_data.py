import argparse
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    index_csv = Path(args.index_csv)
    out_root = Path(args.out_root)

    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    out_root.mkdir(parents=True, exist_ok=True)

    with index_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    frame_cols = [f"frame_{i:02d}" for i in range(1, 11)]

    copied = 0
    missing = 0
    seen = set()

    out_index_csv = out_root / "data" / "processed" / index_csv.name
    out_index_csv.parent.mkdir(parents=True, exist_ok=True)

    rewritten_rows = []

    for row in rows:
        new_row = dict(row)

        for col in frame_cols:
            src_str = row.get(col, "").strip()
            if not src_str:
                continue

            src = Path(src_str)
            if not src.exists():
                missing += 1
                continue

            try:
                rel = src.relative_to(Path("."))
            except ValueError:
                rel = Path(src_str.lstrip("/"))

            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src not in seen:
                shutil.copy2(src, dst)
                seen.add(src)
                copied += 1

            new_row[col] = str(rel)

        rewritten_rows.append(new_row)

    with out_index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rewritten_rows:
            writer.writerow(row)

    print(f"Wrote packaged index: {out_index_csv}")
    print(f"Copied images: {copied}")
    print(f"Missing images: {missing}")
    print(f"Package root: {out_root}")


if __name__ == "__main__":
    main()