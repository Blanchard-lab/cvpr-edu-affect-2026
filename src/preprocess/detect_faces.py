import argparse
import csv
import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".MP4", ".MOV", ".MKV"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def iter_video_frame_dirs(frames_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    for group_dir in sorted(frames_dir.glob("group_*")):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for video_dir in sorted(group_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            yield group, video_dir.name, video_dir


def list_frames(video_frame_dir: Path) -> List[Path]:
    frames = [p for p in sorted(video_frame_dir.iterdir()) if p.suffix in IMG_EXTS]
    return frames


def parse_frame_index(frame_path: Path) -> int:
    stem = frame_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_detections_csv(out_csv: Path, rows: List[dict]) -> None:
    ensure_parent(out_csv)
    fieldnames = [
        "group",
        "video",
        "frame_path",
        "frame_index",
        "face_idx",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_image_bgr(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def detect_faces_retinaface(frame_path: Path) -> List[Tuple[float, float, float, float, float]]:
    from retinaface import RetinaFace

    img_path = str(frame_path)
    result: Any = RetinaFace.detect_faces(img_path)
    faces: List[Tuple[float, float, float, float, float]] = []

    if not isinstance(result, dict):
        return faces

    for _, v in result.items():
        if not isinstance(v, dict):
            continue
        area = v.get("facial_area", None)
        score = v.get("score", None)
        if area is None or score is None:
            continue
        x1, y1, x2, y2 = area
        faces.append((float(x1), float(y1), float(x2), float(y2), float(score)))

    faces.sort(key=lambda t: t[4], reverse=True)
    return faces


def make_insightface_app(device: str):
    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"]
    if device.lower() in {"cuda", "gpu"}:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=0 if device.lower() in {"cuda", "gpu"} else -1, det_size=(640, 640))
    return app


def detect_faces_insightface(app, frame_path: Path) -> List[Tuple[float, float, float, float, float]]:
    img = load_image_bgr(frame_path)
    faces_out = app.get(img)
    faces: List[Tuple[float, float, float, float, float]] = []
    for f in faces_out:
        bbox = getattr(f, "bbox", None)
        det_score = getattr(f, "det_score", None)
        if bbox is None or det_score is None:
            continue
        x1, y1, x2, y2 = bbox
        faces.append((float(x1), float(y1), float(x2), float(y2), float(det_score)))
    faces.sort(key=lambda t: t[4], reverse=True)
    return faces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/interim/frames")
    ap.add_argument("--output_dir", default="data/interim/detections")
    ap.add_argument("--backend", choices=["retinaface", "insightface"], default="retinaface")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--detect_every_n", type=int, default=1)
    ap.add_argument("--max_faces", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    detect_every_n = max(1, args.detect_every_n)
    max_faces = max(1, args.max_faces)

    insight_app = None
    if args.backend == "insightface":
        insight_app = make_insightface_app(args.device)

    total_videos = 0
    for group, video, video_frame_dir in iter_video_frame_dirs(frames_dir):
        frames = list_frames(video_frame_dir)
        if not frames:
            continue

        out_csv = output_dir / group / video / "detections.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"Skipping existing detections: {group}/{video}")
            continue

        total_videos += 1
        print(f"Detecting faces: {group}/{video} (frames={len(frames)}, backend={args.backend})")

        rows: List[dict] = []
        for idx, frame_path in enumerate(frames):
            if idx % detect_every_n != 0:
                continue

            frame_index = parse_frame_index(frame_path)

            if args.backend == "retinaface":
                faces = detect_faces_retinaface(frame_path)
            else:
                faces = detect_faces_insightface(insight_app, frame_path)

            faces = faces[:max_faces]
            for face_idx, (x1, y1, x2, y2, score) in enumerate(faces):
                rows.append(
                    {
                        "group": group,
                        "video": video,
                        "frame_path": str(frame_path),
                        "frame_index": frame_index,
                        "face_idx": face_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                    }
                )

        write_detections_csv(out_csv, rows)
        print(f"Wrote: {out_csv} (rows={len(rows)})")

    print(f"Done. Videos processed: {total_videos}")


if __name__ == "__main__":
    main()
