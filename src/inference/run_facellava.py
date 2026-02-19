import argparse
import csv
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


def majority_vote(labels):
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


def parse_choice(text, choices):
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    for ch in choices:
        if ch.lower() in t.split():
            return ch
    for ch in choices:
        if ch.lower() in t:
            return ch
    return ""


def run_facellava_single(image_path, facellava_root, model_path, prompt):
    cmd = [
        "python",
        "inference.py",
        f"--model_path={model_path}",
        f"--file_path={image_path}",
        f"--prompt={prompt}",
    ]

    print(facellava_root)
    print(image_path)
    result = subprocess.run(
        cmd,
        cwd=facellava_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("WTF")
        return ""

    return result.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--facellava_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument("--drop_missing_frames", action="store_true")
    # ap.add_argument("--device", type=int, default=10)
    args = ap.parse_args()

    index_csv = Path(args.index_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    k = args.k_frames
    frame_cols = [f"frame_{i:02d}" for i in range(1, k + 1)]

    choices = ["Neutral", "Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust"]
    prompt = (
        "Tell me about the emotions in the photo."
    )

    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        rows = rows[:5]

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
            used = 0

            for p in paths:
                used += 1
                print("PROMPT", prompt)
                output_text = run_facellava_single(
                    image_path=p,
                    facellava_root=args.facellava_root,
                    model_path=args.model_path,
                    prompt=prompt,
                )
                print("OUTPUT", output_text)
                pred = parse_choice(output_text, choices)
                print("PRED", pred)
                if pred:
                    preds.append(pred)

            if not preds:
                if args.drop_missing_frames:
                    continue
                out_row = dict(row)
                out_row.update({"pred_label": "", "neutral_rate": "", "used_frames": used})
                w.writerow(out_row)
                continue

            pred_label = majority_vote(preds)
            neutral_rate = sum(1 for p in preds if p.lower() == "neutral") / float(len(preds))

            out_row = dict(row)
            out_row.update(
                {
                    "pred_label": pred_label,
                    "neutral_rate": neutral_rate,
                    "used_frames": used,
                }
            )
            w.writerow(out_row)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
