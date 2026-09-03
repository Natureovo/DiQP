import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from evaluate_ldv_batch import read_metrics, read_protocol, write_failures, write_summary


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DiQP on aligned raw and pre-encoded reconstructed YUV files "
            "without running an encoder."
        )
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--encoded-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video-ids", type=int, nargs="+")
    selection.add_argument("--all-matched", action="store_true")
    parser.add_argument("--qp", type=int, default=37)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=536)
    parser.add_argument("--pix-fmt", choices=("yuv420p",), default="yuv420p")
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of leading frames to evaluate; use 0 for all available frames.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-limit", type=int, default=2)
    parser.add_argument(
        "--model-path",
        default=str(BASE_DIR / "pretrained" / "checkpoint_HEVC.pt"),
    )
    parser.add_argument("--ffmpeg", default=os.environ.get("DIQP_FFMPEG", "auto"))
    parser.add_argument(
        "--work-root",
        default=os.path.join(tempfile.gettempdir(), "diqp_preencoded_yuv"),
        help="Temporary PNG workspace. Per-video frames are removed by default.",
    )
    parser.add_argument(
        "--run-root",
        default=str(BASE_DIR / "batchResults" / "preencoded_yuv"),
    )
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def numeric_raw_ids(raw_dir):
    ids = set()
    for path in Path(raw_dir).glob("*.yuv"):
        if path.stem.isdigit():
            ids.add(int(path.stem))
    return ids


def numeric_reconstructed_ids(encoded_dir, qp):
    suffix = f"_{qp}rec.yuv"
    ids = set()
    for path in Path(encoded_dir).glob(f"*{suffix}"):
        prefix = path.name[: -len(suffix)]
        if prefix.isdigit():
            ids.add(int(prefix))
    return ids


def discover_pairs(raw_dir, encoded_dir, qp):
    raw_ids = numeric_raw_ids(raw_dir)
    reconstructed_ids = numeric_reconstructed_ids(encoded_dir, qp)
    return (
        sorted(raw_ids & reconstructed_ids),
        sorted(raw_ids - reconstructed_ids),
        sorted(reconstructed_ids - raw_ids),
    )


def resolve_raw_yuv(raw_dir, video_id):
    candidates = [
        Path(raw_dir) / f"{video_id}.yuv",
        Path(raw_dir) / f"{video_id:03d}.yuv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find Raw YUV for video {video_id}: {candidates}")


def resolve_reconstructed_yuv(encoded_dir, video_id, qp):
    candidates = [
        Path(encoded_dir) / f"{video_id}_{qp}rec.yuv",
        Path(encoded_dir) / f"{video_id:03d}_{qp}rec.yuv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find reconstructed YUV for video {video_id}, QP {qp}: {candidates}"
    )


def yuv420p_frame_bytes(width, height):
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("yuv420p requires positive even width and height values.")
    return width * height * 3 // 2


def aligned_frame_count(raw_path, reconstructed_path, width, height):
    bytes_per_frame = yuv420p_frame_bytes(width, height)
    raw_size = raw_path.stat().st_size
    reconstructed_size = reconstructed_path.stat().st_size
    if raw_size % bytes_per_frame:
        raise ValueError(
            f"Raw YUV size is not divisible by one frame: {raw_path} ({raw_size} bytes)."
        )
    if reconstructed_size % bytes_per_frame:
        raise ValueError(
            "Reconstructed YUV size is not divisible by one frame: "
            f"{reconstructed_path} ({reconstructed_size} bytes)."
        )
    raw_frames = raw_size // bytes_per_frame
    reconstructed_frames = reconstructed_size // bytes_per_frame
    if raw_frames != reconstructed_frames:
        print(
            "Warning: total frame counts differ; using their common leading range: "
            f"raw={raw_frames}, reconstructed={reconstructed_frames}."
        )
    return min(raw_frames, reconstructed_frames)


def resolve_executable(value):
    candidates = (
        ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", shutil.which("ffmpeg")]
        if value == "auto"
        else [value]
    )
    checked = []
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.append(candidate)
        resolved = shutil.which(candidate)
        if resolved is None:
            path = Path(candidate).expanduser()
            resolved = str(path.resolve()) if path.is_file() else None
        if resolved is None:
            continue
        result = subprocess.run(
            [resolved, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return resolved
    raise FileNotFoundError(f"No working FFmpeg found. Checked: {checked}")


def safe_remove(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path == root or root not in path.parents:
        raise RuntimeError(f"Refusing to remove path outside work root: {path}")
    if path.exists():
        shutil.rmtree(path)


def reset_directory(path, root):
    safe_remove(path, root)
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(command):
    print("Running:", shlex.join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=BASE_DIR, check=True)


def extract_yuv(ffmpeg, source, output_dir, work_root, width, height, pix_fmt, frames):
    reset_directory(output_dir, work_root)
    output_pattern = Path(output_dir) / "%03d.png"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-video_size",
        f"{width}x{height}",
        "-i",
        source,
        "-frames:v",
        str(frames),
        "-vsync",
        "0",
        "-start_number",
        "0",
        output_pattern,
    ]
    run_command(command)
    png_count = sum(1 for _ in Path(output_dir).glob("*.png"))
    if png_count != frames:
        raise RuntimeError(
            f"FFmpeg produced {png_count} PNG files, expected {frames}: {output_dir}"
        )


def prepare_pair(args, ffmpeg, video_id):
    raw_source = resolve_raw_yuv(args.raw_dir, video_id)
    reconstructed_source = resolve_reconstructed_yuv(
        args.encoded_dir, video_id, args.qp
    )
    available_frames = aligned_frame_count(
        raw_source, reconstructed_source, args.width, args.height
    )
    frames = available_frames if args.frames == 0 else min(args.frames, available_frames)
    if frames < 3:
        raise ValueError(f"At least 3 aligned frames are required, found {frames}.")
    if frames > 300:
        print("DiQP supports at most 300 frames; using the first 300.")
        frames = 300

    work_root = Path(args.work_root).resolve()
    raw_output = work_root / "Raw" / f"{video_id:03d}"
    encoded_output = work_root / "Encoded" / f"{video_id:03d}" / f"QP-{args.qp}"
    extract_yuv(
        ffmpeg,
        raw_source,
        raw_output,
        work_root,
        args.width,
        args.height,
        args.pix_fmt,
        frames,
    )
    extract_yuv(
        ffmpeg,
        reconstructed_source,
        encoded_output,
        work_root,
        args.width,
        args.height,
        args.pix_fmt,
        frames,
    )
    return frames


def clean_pair(args, video_id):
    work_root = Path(args.work_root).resolve()
    safe_remove(work_root / "Raw" / f"{video_id:03d}", work_root)
    safe_remove(work_root / "Encoded" / f"{video_id:03d}", work_root)


def write_protocol(path, args, matched, raw_only, reconstructed_only):
    with open(path, "w", encoding="utf8") as protocol_file:
        for key, value in sorted(vars(args).items()):
            protocol_file.write(f"{key}={value}\n")
        protocol_file.write(f"matched_count={len(matched)}\n")
        protocol_file.write(f"raw_only_count={len(raw_only)}\n")
        protocol_file.write(f"reconstructed_only_count={len(reconstructed_only)}\n")
        protocol_file.write("input_kind=preencoded_reconstructed_yuv\n")
        protocol_file.write("encoder_was_run=false\n")


def validate_resume_protocol(path, args):
    protocol = read_protocol(path)
    expected = {
        "raw_dir": args.raw_dir,
        "encoded_dir": args.encoded_dir,
        "qp": str(args.qp),
        "width": str(args.width),
        "height": str(args.height),
        "pix_fmt": args.pix_fmt,
        "frames": str(args.frames),
        "model_path": args.model_path,
        "input_kind": "preencoded_reconstructed_yuv",
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume protocol does not match this request: {mismatches}")


def main():
    args = parse_args()
    if args.qp < 0 or args.qp > 255:
        raise ValueError("--qp must be in [0, 255].")
    if args.frames < 0:
        raise ValueError("--frames must be non-negative.")
    if args.batch_size < 1 or args.workers < 0 or args.save_limit < 0:
        raise ValueError("Invalid batch size, workers, or save limit.")

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    encoded_dir = Path(args.encoded_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw YUV directory does not exist: {raw_dir}")
    if not encoded_dir.is_dir():
        raise FileNotFoundError(
            f"Reconstructed YUV directory does not exist: {encoded_dir}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    args.raw_dir = str(raw_dir)
    args.encoded_dir = str(encoded_dir)
    args.model_path = str(model_path)
    args.work_root = str(Path(args.work_root).expanduser().resolve())
    ffmpeg = resolve_executable(args.ffmpeg)

    matched, raw_only, reconstructed_only = discover_pairs(raw_dir, encoded_dir, args.qp)
    if not matched:
        raise RuntimeError(f"No matching QP {args.qp} raw/reconstructed YUV pairs found.")
    if args.all_matched:
        selected = matched
    else:
        selected = sorted(set(args.video_ids))
        missing = sorted(set(selected) - set(matched))
        if missing:
            raise FileNotFoundError(f"Requested video IDs have no aligned pair: {missing}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_run:
        run_dir = Path(args.resume_run).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        validate_resume_protocol(run_dir / "protocol.txt", args)
        protocol_name = f"resume_{timestamp}.txt"
    else:
        run_dir = Path(args.run_root).expanduser().resolve() / f"qp{args.qp}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        protocol_name = "protocol.txt"

    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.csv"
    failures_path = run_dir / "failures.csv"
    write_protocol(
        run_dir / protocol_name, args, matched, raw_only, reconstructed_only
    )

    existing_rows = read_metrics(metrics_path) if metrics_path.is_file() else []
    existing_models = {
        os.path.abspath(row["Model"]) for row in existing_rows if row.get("Model")
    }
    if existing_models and existing_models != {str(model_path)}:
        raise ValueError("Cannot resume metrics produced by a different model checkpoint.")
    completed = {
        int(row["Sequence"])
        for row in existing_rows
        if int(row["QP"]) == args.qp
    }

    print(f"Raw YUV dir     : {raw_dir}")
    print(f"Reconstructed dir: {encoded_dir}")
    print(f"Matched pairs   : {len(matched)}")
    print(f"Raw only        : {len(raw_only)}")
    print(f"Rec only        : {len(reconstructed_only)}")
    print(f"Selected videos : {len(selected)}")
    print(f"QP / geometry   : {args.qp} / {args.width}x{args.height} {args.pix_fmt}")
    print(f"Temporary frames: {args.work_root}")
    print(f"Run directory   : {run_dir}")

    failures = []
    test_script = BASE_DIR / "test.py"
    remaining = [video_id for video_id in selected if video_id not in completed]
    for index, video_id in enumerate(remaining, start=1):
        print(
            f"\n=== Video {index}/{len(remaining)}: {video_id}, QP {args.qp} ===",
            flush=True,
        )
        try:
            frames = prepare_pair(args, ffmpeg, video_id)
            visual_dir = run_dir / "visuals" / f"{video_id:03d}"
            command = [
                sys.executable,
                test_script,
                "--raw-path",
                Path(args.work_root) / "Raw",
                "--encoded-path",
                Path(args.work_root) / "Encoded",
                "--sequence",
                str(video_id),
                "--qp",
                str(args.qp),
                "--frame-limit",
                str(frames),
                "--fraction",
                "1",
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--save-limit",
                str(args.save_limit),
                "--results-dir",
                visual_dir,
                "--report",
                metrics_path,
                "--model-path",
                model_path,
                "--full-frame-metrics",
            ]
            if args.no_amp:
                command.append("--no-amp")
            run_command(command)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            failures.append([video_id, video_id, args.qp, str(error)])
            print(f"Video {video_id} failed: {error}")
            if args.stop_on_error:
                write_failures(failures_path, failures)
                raise
        finally:
            if not args.keep_frames:
                clean_pair(args, video_id)

    write_failures(failures_path, failures)
    if not metrics_path.is_file():
        raise RuntimeError(f"No evaluation completed. See: {failures_path}")
    rows = read_metrics(metrics_path)
    summary_rows = write_summary(summary_path, rows)

    print("\nPre-encoded YUV evaluation finished")
    for row in summary_rows:
        y_gain = f"{row[8]:+.4f} dB" if row[8] != "" else "n/a"
        print(
            f"{row[0]:>5}: runs={row[2]}, RGB PSNR gain={row[5]:+.4f} dB, "
            f"Y PSNR gain={y_gain}, SSIM gain={row[11]:+.6f}, "
            f"positive RGB={row[12]}/{row[2]}"
        )
    print(f"Metrics        : {metrics_path}")
    print(f"Summary        : {summary_path}")
    print(f"Failures       : {failures_path}")


if __name__ == "__main__":
    main()
