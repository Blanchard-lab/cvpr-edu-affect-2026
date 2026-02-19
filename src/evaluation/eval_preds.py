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


def get_test_groups(splits_json: Path) -> set:
    splits = json.loads(splits_json.read_text())
    return set(splits.get("test", []))


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


def filter_test(rows: List[Dict], test_groups: set) -> List[Dict]:
    out = []
    for r in rows:
        if group_name_from_row(r) in test_groups:
            out.append(r)
    return out


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


def join_by_instance(openface_rows: List[Dict], libreface_rows: List[Dict]) -> List[Tuple[Dict, Dict]]:
    a = {instance_key(r): r for r in openface_rows}
    b = {instance_key(r): r for r in libreface_rows}
    keys = sorted(set(a.keys()) & set(b.keys()))
    return [(a[k], b[k]) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--openface_csv", required=True)
    ap.add_argument("--libreface_csv", required=True)
    ap.add_argument("--eval_split", default="all", choices=["all", "train", "val", "test"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = get_groups_for_split(Path(args.splits_json), args.eval_split)
    openface = filter_test(load_csv(Path(args.openface_csv)), groups)
    libreface = filter_test(load_csv(Path(args.libreface_csv)), groups)

    joined = join_by_instance(openface, libreface)

    gt_open_pairs = []
    gt_lib_pairs = []
    of_lf_pairs = []

    gt_all = []
    of_all = []
    lf_all = []

    for of_r, lf_r in joined:
        gt = map_gt_label(of_r.get("labels", "") or of_r.get("label", ""))
        ofp = map_pred_label(of_r.get("pred_label", ""))
        lfp = map_pred_label(lf_r.get("pred_label", ""))

        if gt:
            gt_all.append(gt)
        if ofp:
            of_all.append(ofp)
        if lfp:
            lf_all.append(lfp)

        if gt and ofp:
            gt_open_pairs.append((gt, ofp))
        if gt and lfp:
            gt_lib_pairs.append((gt, lfp))
        if ofp and lfp:
            of_lf_pairs.append((ofp, lfp))

    gt_labels = unique_in_order(gt_all)
    pred_labels = unique_in_order(of_all + lf_all)

    if not pred_labels:
        pred_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust"]

    cm_gt_of = make_matrix(gt_labels, pred_labels, gt_open_pairs)
    cm_gt_lf = make_matrix(gt_labels, pred_labels, gt_lib_pairs)

    cm_of_lf = make_matrix(pred_labels, pred_labels, of_lf_pairs)

    write_matrix_csv(out_dir / "cm_gt_vs_openface_raw.csv", cm_gt_of, gt_labels, pred_labels, "int")
    write_matrix_csv(out_dir / "cm_gt_vs_openface_row_norm.csv", row_normalize(cm_gt_of), gt_labels, pred_labels, "float")

    write_matrix_csv(out_dir / "cm_gt_vs_libreface_raw.csv", cm_gt_lf, gt_labels, pred_labels, "int")
    write_matrix_csv(out_dir / "cm_gt_vs_libreface_row_norm.csv", row_normalize(cm_gt_lf), gt_labels, pred_labels, "float")

    write_matrix_csv(out_dir / "cm_openface_vs_libreface_raw.csv", cm_of_lf, pred_labels, pred_labels, "int")
    write_matrix_csv(out_dir / "cm_openface_vs_libreface_row_norm.csv", row_normalize(cm_of_lf), pred_labels, pred_labels, "float")

    summary = {
        "n_joined_instances": len(joined),
        "n_gt_vs_openface_pairs": len(gt_open_pairs),
        "n_gt_vs_libreface_pairs": len(gt_lib_pairs),
        "n_openface_vs_libreface_pairs": len(of_lf_pairs),
        "gt_label_set": gt_labels,
        "pred_label_set": pred_labels,
        "groups": sorted(list(groups)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote matrices to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
