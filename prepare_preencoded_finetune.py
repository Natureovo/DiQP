import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROTOCOL_NAME = "preencoded_hevc_qp37_woLF_oneI"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a video-disjoint DiQP fine-tuning split from aligned Raw and "
            "pre-encoded reconstructed YUV files."
        )
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--encoded-dir", required=True)
    parser.add_argument("--train-ids", type=int, nargs="+", required=True)
    parser.add_argument("--val-ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--test-ids",
        type=int,
        nargs="*",
        default=[],
        help="Reserved test IDs. They are checked for overlap but are not extracted.",
    )
    parser.add_argument("--qp", type=int, default=37)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=536)
    parser.add_argument("--pix-fmt", choices=("yuv420p",), default="yuv420p")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--output-root",
        default=str(BASE_DIR / "data" / "LDV_woLF_oneI_qp37"),
    )
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def yuv420p_frame_bytes(width, height):
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("yuv420p requires positive even width and height values.")
    return width * height * 3 // 2


def resolve_raw_yuv(raw_dir, video_id):
    for name in (f"{video_id}.yuv", f"{video_id:03d}.yuv"):
        path = raw_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find Raw YUV for video {video_id} in {raw_dir}")


def resolve_reconstructed_yuv(encoded_dir, video_id, qp):
    for name in (f"{video_id}_{qp}rec.yuv", f"{video_id:03d}_{qp}rec.yuv"):
        path = encoded_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find reconstructed YUV for video {video_id}, QP {qp} in {encoded_dir}"
    )


def frame_count(path, bytes_per_frame):
    size = path.stat().st_size
    if size % bytes_per_frame:
        raise ValueError(
            f"YUV size is not divisible by a {bytes_per_frame}-byte frame: "
            f"{path} ({size} bytes)"
        )
    return size // bytes_per_frame


def validate_disjoint_ids(train_ids, val_ids, test_ids):
    groups = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise ValueError(f"{left}/{right} video IDs overlap: {overlap}")


def resolve_ffmpeg(value):
    resolved = shutil.which(value)
    if resolved is None:
        path = Path(value).expanduser()
        resolved = str(path.resolve()) if path.is_file() else None
    if resolved is None:
        raise FileNotFoundError(f"Cannot find FFmpeg: {value}")
    result = subprocess.run(
        [resolved, "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg is not executable: {resolved}")
    return resolved


def png_count(directory):
    return sum(1 for _ in directory.glob("*.png")) if directory.is_dir() else 0


def marker_path(output_root, video_id, qp):
    return output_root / ".prepared" / f"{video_id:03d}_QP-{qp}.json"


def marker_matches(path, expected, raw_output, encoded_output):
    if not path.is_file():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        saved == expected
        and png_count(raw_output) == expected["frames"]
        and png_count(encoded_output) == expected["frames"]
    )


def reset_output_directory(path, output_root):
    resolved = path.resolve()
    root = output_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to replace a directory outside output root: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def extract_yuv(ffmpeg, source, output, output_root, args):
    reset_output_directory(output, output_root)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        args.pix_fmt,
        "-video_size",
        f"{args.width}x{args.height}",
        "-i",
        str(source),
        "-frames:v",
        str(args.frames),
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(output / "%03d.png"),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    actual = png_count(output)
    if actual != args.frames:
        raise RuntimeError(f"FFmpeg produced {actual} frames, expected {args.frames}: {output}")


def inspect_pair(raw_dir, encoded_dir, video_id, qp, bytes_per_frame, frames):
    raw = resolve_raw_yuv(raw_dir, video_id)
    reconstructed = resolve_reconstructed_yuv(encoded_dir, video_id, qp)
    raw_frames = frame_count(raw, bytes_per_frame)
    reconstructed_frames = frame_count(reconstructed, bytes_per_frame)
    available = min(raw_frames, reconstructed_frames)
    if available < frames:
        raise ValueError(
            f"Video {video_id} has fewer than {frames} usable leading frames: "
            f"Raw={raw_frames}, reconstructed={reconstructed_frames}"
        )
    if raw_frames != reconstructed_frames:
        print(
            f"Warning: video {video_id} total frame counts differ "
            f"(Raw={raw_frames}, reconstructed={reconstructed_frames}); "
            f"using the first {frames} aligned frames."
        )
    return raw, reconstructed, raw_frames, reconstructed_frames


def prepare_video(args, ffmpeg, output_root, split, video_id, pair):
    raw, reconstructed, raw_frames, reconstructed_frames = pair
    raw_output = output_root / "Raw" / f"{video_id:03d}"
    encoded_output = output_root / "Encoded" / f"{video_id:03d}" / f"QP-{args.qp}"
    marker = marker_path(output_root, video_id, args.qp)
    expected = {
        "protocol": PROTOCOL_NAME,
        "split": split,
        "video_id": video_id,
        "qp": args.qp,
        "width": args.width,
        "height": args.height,
        "pix_fmt": args.pix_fmt,
        "frames": args.frames,
        "raw_source": str(raw),
        "raw_size": raw.stat().st_size,
        "reconstructed_source": str(reconstructed),
        "reconstructed_size": reconstructed.stat().st_size,
    }
    if not args.overwrite and marker_matches(
        marker, expected, raw_output, encoded_output
    ):
        print(f"Skipping prepared {split} video {video_id:03d}")
        return raw_frames, reconstructed_frames

    print(f"\nPreparing {split} video {video_id:03d}, QP {args.qp}", flush=True)
    extract_yuv(ffmpeg, raw, raw_output, output_root, args)
    extract_yuv(ffmpeg, reconstructed, encoded_output, output_root, args)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(expected, indent=2), encoding="utf8")
    return raw_frames, reconstructed_frames


def main():
    args = parse_args()
    if args.qp < 0 or args.qp > 51:
        raise ValueError("HEVC --qp must be in [0, 51].")
    if args.frames < 3 or args.frames > 300:
        raise ValueError("--frames must be in [3, 300].")
    validate_disjoint_ids(args.train_ids, args.val_ids, args.test_ids)

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    encoded_dir = Path(args.encoded_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not raw_dir.is_dir() or not encoded_dir.is_dir():
        raise FileNotFoundError("Raw or reconstructed YUV directory does not exist.")
    output_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    bytes_per_frame = yuv420p_frame_bytes(args.width, args.height)

    # Reserved test videos stay untouched; only train/validation inputs need PNG data.
    requested = sorted(set(args.train_ids + args.val_ids))
    pairs = {
        video_id: inspect_pair(
            raw_dir,
            encoded_dir,
            video_id,
            args.qp,
            bytes_per_frame,
            args.frames,
        )
        for video_id in requested
    }

    rows = []
    for split, video_ids in (("train", args.train_ids), ("val", args.val_ids)):
        for video_id in sorted(set(video_ids)):
            raw_frames, reconstructed_frames = prepare_video(
                args, ffmpeg, output_root, split, video_id, pairs[video_id]
            )
            rows.append(
                [
                    video_id,
                    split,
                    args.qp,
                    args.frames,
                    min(raw_frames, reconstructed_frames),
                    PROTOCOL_NAME,
                    "unknown_preencoded_encoder",
                    "woLF_oneI",
                    0,
                ]
            )

    split_path = output_root / "split.csv"
    with split_path.open("w", encoding="utf8", newline="") as split_file:
        writer = csv.writer(split_file)
        writer.writerow(
            [
                "Sequence",
                "Split",
                "QP",
                "Frames Used",
                "Frames Available",
                "Encoder",
                "HM Encoder",
                "HM Config",
                "HM Padding",
            ]
        )
        writer.writerows(rows)

    protocol = {
        "protocol": PROTOCOL_NAME,
        "raw_dir": str(raw_dir),
        "encoded_dir": str(encoded_dir),
        "output_root": str(output_root),
        "qp": args.qp,
        "geometry": f"{args.width}x{args.height}",
        "pix_fmt": args.pix_fmt,
        "frames": args.frames,
        "train_ids": sorted(set(args.train_ids)),
        "val_ids": sorted(set(args.val_ids)),
        "reserved_test_ids": sorted(set(args.test_ids)),
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf8"
    )

    print("\nPre-encoded woLF + oneI fine-tuning data is ready")
    print(f"Train / val videos: {len(set(args.train_ids))} / {len(set(args.val_ids))}")
    print(f"Reserved test IDs : {sorted(set(args.test_ids))}")
    print(f"QP / frames       : {args.qp} / {args.frames}")
    print(f"Data root         : {output_root}")
    print(f"Split manifest    : {split_path}")


if __name__ == "__main__":
    main()
