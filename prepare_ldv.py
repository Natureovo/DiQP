import argparse
import os
import shlex
import shutil
import subprocess
import sys
from fractions import Fraction


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a small aligned LDV Raw/HEVC frame pair for DiQP testing."
    )
    parser.add_argument("--input", required=True, help="Path to one LDV .mkv video.")
    parser.add_argument("--output-root", default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--sequence", type=int, default=101)
    parser.add_argument("--qp", type=int, default=37)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional resampling FPS. Omit it to preserve the source frame sequence.",
    )
    parser.add_argument(
        "--encoder",
        choices=("hevc_nvenc", "libx265"),
        default="hevc_nvenc",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the generated PNG and MP4 files for this sequence/QP.",
    )
    return parser.parse_args()


def run(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def find_working_ffmpeg():
    candidates = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", shutil.which("ffmpeg")]
    checked = []
    for candidate in candidates:
        if not candidate or candidate in checked or not os.path.isfile(candidate):
            continue
        checked.append(candidate)
        result = subprocess.run(
            [candidate, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return candidate
    raise RuntimeError(
        "No working ffmpeg was found. Checked: " + ", ".join(checked)
    )


def detect_source_frame_rate(ffmpeg, input_video):
    ffprobe_candidates = [
        os.path.join(os.path.dirname(ffmpeg), "ffprobe"),
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        shutil.which("ffprobe"),
    ]
    checked = []
    for ffprobe in ffprobe_candidates:
        if not ffprobe or ffprobe in checked or not os.path.isfile(ffprobe):
            continue
        checked.append(ffprobe)
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_video,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            try:
                if Fraction(value) > 0:
                    return value
            except (ValueError, ZeroDivisionError):
                pass
    print("Could not detect source FPS; using 30 fps for encoded video timing.")
    return "30"


def remove_generated_files(raw_dir, encoded_dir, compressed_video):
    for directory in (raw_dir, encoded_dir):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.lower().endswith(".png"):
                os.remove(os.path.join(directory, name))
    if os.path.isfile(compressed_video):
        os.remove(compressed_video)


def ensure_output_is_empty(raw_dir, encoded_dir, compressed_video, overwrite):
    generated_files = []
    for directory in (raw_dir, encoded_dir):
        if os.path.isdir(directory):
            generated_files.extend(
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if name.lower().endswith(".png")
            )
    if os.path.isfile(compressed_video):
        generated_files.append(compressed_video)

    if generated_files and not overwrite:
        raise FileExistsError(
            "Output already contains generated files. Use another --sequence or pass --overwrite."
        )
    if overwrite:
        remove_generated_files(raw_dir, encoded_dir, compressed_video)


def encoder_options(encoder, qp):
    if encoder == "hevc_nvenc":
        return [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "medium",
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-x265-params",
        f"qp={qp}",
        "-pix_fmt",
        "yuv420p",
    ]


def count_png_files(directory):
    return len([name for name in os.listdir(directory) if name.lower().endswith(".png")])


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Cannot find LDV video: {args.input}")
    if args.frames < 3:
        raise ValueError("--frames must be at least 3.")
    if args.sequence < 0 or args.qp < 0:
        raise ValueError("--sequence and --qp must be non-negative.")

    ffmpeg = find_working_ffmpeg()
    print(f"Using ffmpeg   : {ffmpeg}")

    sequence_name = f"{args.sequence:03d}"
    raw_dir = os.path.join(args.output_root, "Raw", sequence_name)
    encoded_dir = os.path.join(
        args.output_root, "Encoded", sequence_name, f"QP-{args.qp}"
    )
    compressed_video = os.path.join(encoded_dir, f"qp_{args.qp:03d}.mp4")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(encoded_dir, exist_ok=True)
    ensure_output_is_empty(
        raw_dir, encoded_dir, compressed_video, overwrite=args.overwrite
    )

    raw_pattern = os.path.join(raw_dir, "%03d.png")
    encoded_pattern = os.path.join(encoded_dir, "%03d.png")
    fps_value = (
        str(args.fps)
        if args.fps is not None
        else detect_source_frame_rate(ffmpeg, os.path.abspath(args.input))
    )
    print(f"Encoding FPS   : {fps_value}")

    extract_command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        os.path.abspath(args.input),
        "-map",
        "0:v:0",
    ]
    if args.fps is not None:
        extract_command.extend(["-vf", f"fps={fps_value}"])
    extract_command.extend(
        [
            "-frames:v",
            str(args.frames),
            "-start_number",
            "0",
            raw_pattern,
        ]
    )
    run(extract_command)

    encode_command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-framerate",
        fps_value,
        "-start_number",
        "0",
        "-i",
        raw_pattern,
        "-frames:v",
        str(args.frames),
        *encoder_options(args.encoder, args.qp),
        compressed_video,
    ]
    try:
        run(encode_command)
    except subprocess.CalledProcessError:
        if args.encoder != "hevc_nvenc":
            raise
        print("hevc_nvenc failed; retrying with CPU libx265.")
        encode_command = [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-framerate",
            fps_value,
            "-start_number",
            "0",
            "-i",
            raw_pattern,
            "-frames:v",
            str(args.frames),
            *encoder_options("libx265", args.qp),
            compressed_video,
        ]
        run(encode_command)

    run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            compressed_video,
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-frames:v",
            str(args.frames),
            "-start_number",
            "0",
            encoded_pattern,
        ]
    )

    raw_count = count_png_files(raw_dir)
    encoded_count = count_png_files(encoded_dir)
    if raw_count != args.frames or encoded_count != args.frames:
        raise RuntimeError(
            f"Frame preparation is incomplete: Raw={raw_count}, Encoded={encoded_count}, "
            f"expected={args.frames}."
        )

    print("LDV preparation finished")
    print(f"Raw frames     : {raw_dir} ({raw_count})")
    print(f"Encoded frames : {encoded_dir} ({encoded_count})")
    print(f"Compressed HEVC: {compressed_video}")
    print("Next command:")
    print(
        f"{shlex.quote(sys.executable)} {shlex.quote(os.path.join(BASE_DIR, 'test.py'))} "
        f"--sequence {args.sequence} --qp {args.qp} --fraction 1 "
        "--batch-size 1 --save-limit 3 --full-frame-metrics"
    )


if __name__ == "__main__":
    main()
