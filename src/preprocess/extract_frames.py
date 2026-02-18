import argparse
import os
import subprocess
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".MP4", ".MOV", ".MKV"}


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stderr.strip()}")


def ffmpeg_exists() -> bool:
    try:
        run(["ffmpeg", "-version"])
        return True
    except Exception:
        return False


def extract_video_frames(video_path: Path, out_dir: Path, fps: float, image_ext: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"frame_%06d.{image_ext}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        pattern,
    ]
    run(cmd)


def has_frames(out_dir: Path, image_ext: str) -> bool:
    return any(out_dir.glob(f"frame_*.{image_ext}"))


def iter_group_videos(input_dir: Path):
    for group_dir in sorted(input_dir.glob("group_*")):
        if not group_dir.is_dir():
            continue
        videos = [p for p in sorted(group_dir.iterdir()) if p.suffix in VIDEO_EXTS]
        yield group_dir.name, group_dir, videos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw/videos")
    ap.add_argument("--output_dir", default="data/interim/frames")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--image_ext", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_videos = 0
    extracted = 0
    skipped = 0

    for group_name, _, videos in iter_group_videos(input_dir):
        if not videos:
            print(f"Skipping {group_name} (no videos found)")
            continue

        for vid in videos:
            total_videos += 1
            stem = vid.stem
            out_dir = output_dir / group_name / stem

            if not args.overwrite and has_frames(out_dir, args.image_ext):
                print(f"Skipping existing frames: {group_name}/{stem}")
                skipped += 1
                continue

            print(f"Extracting: {group_name}/{stem} (fps={args.fps})")
            extract_video_frames(vid, out_dir, args.fps, args.image_ext)
            extracted += 1

    print(f"Done. Videos seen: {total_videos}, extracted: {extracted}, skipped: {skipped}")
    print(f"Frames root: {output_dir}")


if __name__ == "__main__":
    main()
