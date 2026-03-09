import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


EPISTEMIC_7 = {
    "Curious",
    "Confused",
    "Disengaged",
    "Surprised",
    "Conflicted",
    "Optimistic",
    "Frustrated",
}


def load_csv(path: Path) -> List[Dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def norm(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip()


def group_name_from_row(r: Dict) -> str:
    g = norm(r.get("group", ""))
    if g:
        return g
    gid = norm(r.get("groupID", ""))
    if gid:
        return f"group-{int(float(gid))}"
    return ""


def instance_key(r: Dict) -> Tuple:
    return (
        group_name_from_row(r),
        norm(r.get("participantID", "")),
        int(float(r.get("videoTime", "nan"))) if norm(r.get("videoTime", "")) else None,
        norm(r.get("timestamp", "")),
    )


def filter_groups(rows: List[Dict], groups: set) -> List[Dict]:
    return [r for r in rows if group_name_from_row(r) in groups]


def map_pred_label(pred: str) -> str:
    p = norm(pred).lower()
    if not p:
        return ""
    m = {
        "neutral": "Neutral",
        "happy": "Happy",
        "happiness": "Happy",
        "sad": "Sad",
        "sadness": "Sad",
        "angry": "Angry",
        "anger": "Angry",
        "surprise": "Surprise",
        "surprised": "Surprise",
        "fear": "Fear",
        "fearful": "Fear",
        "disgust": "Disgust",
        "disgusted": "Disgust",
        "contempt": "Contempt",
        "unknown": "",
        "none": "",
    }
    return m.get(p, norm(pred))


def map_gt_label(gt: str) -> str:
    g = norm(gt)
    return g if g in EPISTEMIC_7 else ""


def get_groups_for_split(splits_json: Path, split: str) -> set:
    splits = json.loads(splits_json.read_text())
    if split == "all":
        return set(splits.get("train", []) + splits.get("val", []) + splits.get("test", []))
    return set(splits.get(split, []))


def unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen and x != "":
            seen.add(x)
            out.append(x)
    return out


def make_matrix(rows: List[str], cols: List[str], pairs: List[Tuple[str, str]]) -> np.ndarray:
    rix = {r: i for i, r in enumerate(rows)}
    cix = {c: j for j, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for a, b in pairs:
        if a not in rix or b not in cix:
            continue
        M[rix[a], cix[b]] += 1
    return M


def row_normalize(M: np.ndarray) -> np.ndarray:
    denom = M.sum(axis=1, keepdims=True).astype(np.float64)
    denom[denom == 0] = 1.0
    return M.astype(np.float64) / denom


def write_matrix_csv(path: Path, M, rows: List[str], cols: List[str], fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cols)
        for i, rname in enumerate(rows):
            if fmt == "int":
                w.writerow([rname] + [int(x) for x in M[i].tolist()])
            else:
                w.writerow([rname] + [f"{float(x):.4f}" for x in M[i].tolist()])


def join_by_instance(rows_a: List[Dict], rows_b: List[Dict]) -> List[Tuple[Dict, Dict]]:
    a = {instance_key(r): r for r in rows_a}
    b = {instance_key(r): r for r in rows_b}
    keys = sorted(set(a.keys()) & set(b.keys()))
    return [(a[k], b[k]) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_a_csv", required=True)
    ap.add_argument("--model_b_csv", required=True)
    ap.add_argument("--model_a_name", required=True)
    ap.add_argument("--model_b_name", required=True)
    ap.add_argument("--eval_split", default="all", choices=["all", "train", "val", "test"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_a_name = args.model_a_name.lower()
    model_b_name = args.model_b_name.lower()

    groups = get_groups_for_split(Path(args.splits_json), args.eval_split)
    rows_a = filter_groups(load_csv(Path(args.model_a_csv)), groups)
    rows_b = filter_groups(load_csv(Path(args.model_b_csv)), groups)

    joined = join_by_instance(rows_a, rows_b)

    gt_a_pairs = []
    gt_b_pairs = []
    a_b_pairs = []

    gt_all = []
    a_all = []
    b_all = []

    for a_r, b_r in joined:
        gt = map_gt_label(a_r.get("labels", "") or a_r.get("label", ""))
        apred = map_pred_label(a_r.get("pred_label", ""))
        bpred = map_pred_label(b_r.get("pred_label", ""))

        if gt:
            gt_all.append(gt)
        if apred:
            a_all.append(apred)
        if bpred:
            b_all.append(bpred)

        if gt and apred:
            gt_a_pairs.append((gt, apred))
        if gt and bpred:
            gt_b_pairs.append((gt, bpred))
        if apred and bpred:
            a_b_pairs.append((apred, bpred))

    gt_labels = unique_in_order(gt_all)
    pred_labels = ["Neutral", "Surprise", "Disgust", "Happy", "Sad", "Angry", "Fear"]
    if "Contempt" in set(a_all + b_all):
        pred_labels = ["Neutral", "Surprise", "Disgust", "Happy", "Contempt", "Sad", "Angry", "Fear"]

    cm_gt_a = make_matrix(gt_labels, pred_labels, gt_a_pairs)
    cm_gt_b = make_matrix(gt_labels, pred_labels, gt_b_pairs)
    cm_a_b = make_matrix(pred_labels, pred_labels, a_b_pairs)

    write_matrix_csv(out_dir / f"cm_gt_vs_{model_a_name}_raw.csv", cm_gt_a, gt_labels, pred_labels, "int")
    write_matrix_csv(out_dir / f"cm_gt_vs_{model_a_name}_row_norm.csv", row_normalize(cm_gt_a), gt_labels, pred_labels, "float")

    write_matrix_csv(out_dir / f"cm_gt_vs_{model_b_name}_raw.csv", cm_gt_b, gt_labels, pred_labels, "int")
    write_matrix_csv(out_dir / f"cm_gt_vs_{model_b_name}_row_norm.csv", row_normalize(cm_gt_b), gt_labels, pred_labels, "float")

    write_matrix_csv(out_dir / f"cm_{model_a_name}_vs_{model_b_name}_raw.csv", cm_a_b, pred_labels, pred_labels, "int")
    write_matrix_csv(out_dir / f"cm_{model_a_name}_vs_{model_b_name}_row_norm.csv", row_normalize(cm_a_b), pred_labels, pred_labels, "float")

    summary = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n_joined_instances": len(joined),
        "n_gt_vs_model_a_pairs": len(gt_a_pairs),
        "n_gt_vs_model_b_pairs": len(gt_b_pairs),
        "n_model_a_vs_model_b_pairs": len(a_b_pairs),
        "gt_label_set": gt_labels,
        "pred_label_set": pred_labels,
        "groups": sorted(list(groups)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote matrices to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main() 