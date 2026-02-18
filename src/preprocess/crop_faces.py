import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def read_detections_csv(path: Path) -> List[Dict]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def write_crops_manifest(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "crop_id",
        "group",
        "video",
        "frame_path",
        "frame_index",
        "face_idx",
        "det_score",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def load_image_bgr(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def save_image_bgr(path: Path, img) -> None:
    import cv2
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

def pad_box(x1: float, y1: float, x2: float, y2: float, pad_ratio: float) -> Tuple[int, int, int, int]:
    w = x2 - x1
    h = y2 - y1
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    return int(x1 - pad_w), int(y1 - pad_h), int(x2 + pad_w), int(y2 + pad_h)

def crop_and_resize(img, x1: int, y1: int, x2: int, y2: int, out_size: int):
    
    h, w = img.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 1, w)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 1, h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


def make_crop_id(group: str, video: str, frame_index: int, face_idx: int) -> str:
    return f"{group}__{video}__f{frame_index:06d}__face{face_idx:02d}"


def iter_detection_files(detections_dir: Path):
    for group_dir in sorted(detections_dir.glob("group_*")):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for video_dir in sorted(group_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            det_csv = video_dir / "detections.csv"
            if det_csv.exists():
                yield group, video_dir.name, det_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_dir", default="data/interim/detections")
    ap.add_argument("--output_dir", default="data/interim/crops")
    ap.add_argument("--pad_ratio", type=float, default=0.25)
    ap.add_argument("--out_size", type=int, default=224)
    ap.add_argument("--image_ext", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    detections_dir = Path(args.detections_dir)
    output_dir = Path(args.output_dir)

    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections dir not found: {detections_dir}")

    total_videos = 0

    for group, video, det_csv in iter_detection_files(detections_dir):
        total_videos += 1
        out_video_dir = output_dir / group / video
        manifest_path = out_video_dir / "crops.csv"

        if manifest_path.exists() and not args.overwrite:
            print(f"Skipping existing crops: {group}/{video}")
            continue

        detections = read_detections_csv(det_csv)
        if not detections:
            print(f"No detections: {group}/{video}")
            write_crops_manifest(manifest_path, [])
            continue

        print(f"Cropping: {group}/{video} (detections={len(detections)})")

        rows_out: List[Dict] = []
        for d in detections:
            frame_path = Path(d["frame_path"])
            frame_index = int(float(d["frame_index"])) if d["frame_index"] else -1
            face_idx = int(float(d["face_idx"]))
            score = float(d["score"])

            x1 = float(d["x1"])
            y1 = float(d["y1"])
            x2 = float(d["x2"])
            y2 = float(d["y2"])

            crop_x1, crop_y1, crop_x2, crop_y2 = pad_box(x1, y1, x2, y2, args.pad_ratio)

            img = load_image_bgr(frame_path)
            crop = crop_and_resize(img, crop_x1, crop_y1, crop_x2, crop_y2, args.out_size)
            if crop is None:
                continue

            crop_id = make_crop_id(group, video, frame_index, face_idx)
            out_path = out_video_dir / f"face_{face_idx:02d}" / f"frame_{frame_index:06d}.{args.image_ext}"

            if out_path.exists() and not args.overwrite:
                pass
            else:
                save_image_bgr(out_path, crop)

            rows_out.append(
                {
                    "crop_id": crop_id,
                    "group": group,
                    "video": video,
                    "frame_path": str(frame_path),
                    "frame_index": frame_index,
                    "face_idx": face_idx,
                    "det_score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_path": str(out_path),
                }
            )

        write_crops_manifest(manifest_path, rows_out)
        print(f"Wrote: {manifest_path} (rows={len(rows_out)})")

    print(f"Done. Videos processed: {total_videos}")


if __name__ == "__main__":
    main()
