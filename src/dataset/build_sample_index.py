import argparse
import csv
import json
import random
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_group_dir(group_dir_name: str) -> int:
    m = re.search(r"group[-_]?(\d+)", group_dir_name)
    return int(m.group(1)) if m else -1


def parse_participant_dir(participant_dir_name: str) -> str:
    m = re.search(r"participant[-_]?(.+)", participant_dir_name)
    return m.group(1) if m else participant_dir_name


def list_images(p: Path):
    if not p.exists() or not p.is_dir():
        return []
    return [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix in IMG_EXTS]


def sec_folder_name(sec: int) -> str:
    return f"{sec-1}_{sec}_frames_selected"


def pick_k_frames_for_instance(
    crops_dir: Path,
    group_id: int,
    participant_id: str,
    center_sec: int,
    margin_s: int,
    k_frames: int,
    rng: random.Random,
):
    group_dir = crops_dir / f"group-{group_id}"
    participant_dir = group_dir / f"participant-{participant_id}"

    candidates = []
    for s in range(center_sec - margin_s, center_sec + margin_s + 1):
        candidates.extend(list_images(participant_dir / sec_folder_name(s)))

    if not candidates:
        return []

    if len(candidates) >= k_frames:
        return [str(p) for p in rng.sample(candidates, k_frames)]

    out = [str(p) for p in candidates]
    while len(out) < k_frames:
        out.append(str(rng.choice(candidates)))
    return out


def read_reports(reports_csv: Path):
    with reports_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def make_group_splits(groups_sorted, test_groups: int, val_groups: int):
    test_groups = max(0, min(test_groups, len(groups_sorted)))
    val_groups = max(0, min(val_groups, len(groups_sorted) - test_groups))
    test = groups_sorted[-test_groups:] if test_groups else []
    val = groups_sorted[-(test_groups + val_groups):-test_groups] if val_groups else []
    train = groups_sorted[: len(groups_sorted) - test_groups - val_groups]
    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", default="data/interim/crops")
    ap.add_argument("--reports_csv", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--k_frames", type=int, default=10)
    ap.add_argument("--margin_s", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_groups", type=int, default=2)
    ap.add_argument("--val_groups", type=int, default=1)
    args = ap.parse_args()

    crops_dir = Path(args.crops_dir)
    reports_csv = Path(args.reports_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    rows = read_reports(reports_csv)

    out_rows = []
    missing = 0

    for i, row in enumerate(rows):
        group_id = int(float(row["groupID"]))
        participant_id = str(row["participantID"])
        label = str(row["labels"]).strip()
        video_time = int(float(row["videoTime"]))

        frames = pick_k_frames_for_instance(
            crops_dir=crops_dir,
            group_id=group_id,
            participant_id=participant_id,
            center_sec=video_time,
            margin_s=args.margin_s,
            k_frames=args.k_frames,
            rng=rng,
        )

        if not frames:
            missing += 1
            continue

        out_row = dict(row)
        for j, fp in enumerate(frames, start=1):
            out_row[f"frame_{j:02d}"] = fp
        out_row["n_frames"] = len(frames)
        out_rows.append(out_row)

    out_path = out_dir / f"index_sampled_k{args.k_frames}_m{args.margin_s}.csv"
    fieldnames = list(rows[0].keys()) + [f"frame_{j:02d}" for j in range(1, args.k_frames + 1)] + ["n_frames"]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    groups = sorted({f"group-{int(float(r['groupID']))}" for r in out_rows}, key=parse_group_dir)
    splits = make_group_splits(groups, args.test_groups, args.val_groups)
    splits_path = out_dir / "splits_groups.json"
    splits_path.write_text(json.dumps(splits, indent=2))

    print(f"Wrote {out_path} ({len(out_rows)} rows, dropped_missing={missing})")
    print(f"Wrote {splits_path}")
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
