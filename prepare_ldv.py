import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

from videoRelated.hm16_3_ldp import verify_package_encoder, write_ldp_config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HM16_3_ENCODER = os.environ.get(
    "DIQP_HM16_3_ENCODER",
    "/home/cp/tools/hm16_3/HM16.3-standard.exe",
)
DEFAULT_HM_ENCODER = os.environ.get(
    "DIQP_HM_ENCODER",
    "/home/cp/\u684c\u9762/yx/HM/bin/umake/gcc-9.4/x86_64/release/TAppEncoder",
)
DEFAULT_HM_CONFIG = os.environ.get(
    "DIQP_HM_CONFIG",
    "/home/cp/\u684c\u9762/yx/HM/cfg/encoder_randomaccess_main.cfg",
)


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
        choices=("hm16_3_ldp", "hm", "hevc_nvenc", "libx265"),
        default="hm16_3_ldp",
        help="HEVC protocol. Defaults to the supplied HM 16.3 Low-Delay P package.",
    )
    parser.add_argument("--hm16-3-encoder", default=DEFAULT_HM16_3_ENCODER)
    parser.add_argument(
        "--hm16-3-runner",
        choices=("auto", "direct", "wine"),
        default="auto",
        help="Run the package EXE directly on Windows or through Wine on Linux.",
    )
    parser.add_argument("--wine", default=os.environ.get("DIQP_WINE", "wine"))
    parser.add_argument(
        "--hm16-3-padding",
        type=int,
        default=8,
        help="Pad input dimensions to this multiple, matching the package workflow.",
    )
    parser.add_argument(
        "--hm16-3-work-root",
        default=os.environ.get(
            "DIQP_HM16_3_WORK_ROOT",
            os.path.join(tempfile.gettempdir(), "diqp_hm16_3"),
        ),
        help="ASCII-only temporary directory used by the legacy Windows encoder.",
    )
    parser.add_argument("--hm-encoder", default=DEFAULT_HM_ENCODER)
    parser.add_argument("--hm-config", default=DEFAULT_HM_CONFIG)
    parser.add_argument(
        "--hm-padding",
        type=int,
        default=32,
        help="Pad HM input dimensions to this multiple, then crop reconstructions back.",
    )
    parser.add_argument(
        "--keep-hm-yuv",
        action="store_true",
        help="Keep the large temporary HM input and reconstructed YUV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated frames and codec artifacts for this sequence/QP.",
    )
    return parser.parse_args()


def run(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def run_logged(command, log_path, cwd=None):
    print("Running:", " ".join(command))
    print(f"HM log         : {log_path}")
    with open(log_path, "w", encoding="utf8") as log_file:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


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


def detect_source_size(ffmpeg, input_video):
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
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                input_video,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if result.returncode == 0 and "x" in value:
            try:
                width, height = (int(part) for part in value.split("x", 1))
                if width > 0 and height > 0:
                    return width, height
            except ValueError:
                pass
    raise RuntimeError(f"Could not detect source dimensions with ffprobe: {input_video}")


def remove_generated_files(raw_dir, encoded_dir):
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            if name.lower().endswith(".png"):
                os.remove(os.path.join(raw_dir, name))
    if os.path.isdir(encoded_dir):
        for name in os.listdir(encoded_dir):
            path = os.path.join(encoded_dir, name)
            if os.path.isdir(path) and name == ".hm_work":
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)


def ensure_output_is_empty(raw_dir, encoded_dir, overwrite):
    generated_files = []
    if os.path.isdir(raw_dir):
        generated_files.extend(
            os.path.join(raw_dir, name)
            for name in os.listdir(raw_dir)
            if name.lower().endswith(".png")
        )
    if os.path.isdir(encoded_dir):
        generated_files.extend(
            os.path.join(encoded_dir, name) for name in os.listdir(encoded_dir)
        )

    if generated_files and not overwrite:
        raise FileExistsError(
            "Output already contains generated files. Use another --sequence or pass --overwrite."
        )
    if overwrite:
        remove_generated_files(raw_dir, encoded_dir)


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


def padded_size(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def hm_frame_rate(fps_value):
    try:
        return max(1, round(float(Fraction(str(fps_value)))))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"HM requires a valid frame rate, got: {fps_value}") from None


def resolve_hm16_3_runner(args):
    runner = args.hm16_3_runner
    if runner == "auto":
        runner = "direct" if os.name == "nt" else "wine"
    if runner == "direct":
        if os.name != "nt":
            raise RuntimeError(
                "HM16.3-standard.exe cannot run directly on Linux. Install Wine and "
                "use --hm16-3-runner wine."
            )
        return runner, None

    wine = shutil.which(args.wine)
    if wine is None:
        raise FileNotFoundError(
            f"Cannot find Wine executable '{args.wine}'. The supplied encoder is a "
            "32-bit Windows EXE, so the server needs Wine with 32-bit support."
        )
    return runner, wine


def hm16_3_path(path, runner):
    absolute = os.path.abspath(path)
    if runner == "wine":
        return "Z:" + absolute.replace("/", "\\")
    return absolute


def prepare_with_hm16_3_ldp(
    args,
    ffmpeg,
    raw_pattern,
    encoded_pattern,
    encoded_dir,
    frame_count,
    fps_value,
    source_size,
):
    encoder = os.path.abspath(args.hm16_3_encoder)
    if not os.path.isfile(encoder):
        raise FileNotFoundError(
            "Cannot find the supplied HM16.3 encoder: " + encoder
        )
    encoder_sha256 = verify_package_encoder(encoder)
    runner, wine = resolve_hm16_3_runner(args)
    if args.hm16_3_padding < 2:
        raise ValueError("--hm16-3-padding must be at least 2.")

    source_width, source_height = source_size
    coded_width = padded_size(source_width, args.hm16_3_padding)
    coded_height = padded_size(source_height, args.hm16_3_padding)
    frame_rate = hm_frame_rate(fps_value)
    work_root = os.path.abspath(args.hm16_3_work_root)
    os.makedirs(work_root, exist_ok=True)
    work_dir = tempfile.mkdtemp(
        prefix=f"seq{args.sequence:03d}_qp{args.qp}_",
        dir=work_root,
    )
    print(f"HM16.3 work dir : {work_dir}")
    input_yuv = os.path.join(work_dir, "input.yuv")
    recon_yuv = os.path.join(work_dir, "recon.yuv")
    temp_bitstream = os.path.join(work_dir, "stream.bin")
    temp_config = os.path.join(work_dir, "hm16_3_ldp.cfg")
    bitstream = os.path.join(encoded_dir, f"qp_{args.qp:03d}_hm16_3_ldp.bin")
    archived_config = os.path.join(encoded_dir, "hm16_3_ldp.cfg")
    log_path = os.path.join(encoded_dir, "hm16_3_ldp_encode.log")
    protocol_path = os.path.join(encoded_dir, "encoding_protocol.txt")

    print(f"Source size     : {source_width}x{source_height}")
    print(f"HM16.3 size     : {coded_width}x{coded_height}")
    print(f"HM16.3 FPS      : {frame_rate}")
    print(f"HM16.3 runner   : {runner}")
    print(f"HM16.3 SHA256   : {encoder_sha256}")
    run(
        [
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
            str(frame_count),
            "-vf",
            f"pad={coded_width}:{coded_height}:0:0:black",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            input_yuv,
        ]
    )

    write_ldp_config(
        temp_config,
        hm16_3_path(input_yuv, runner),
        hm16_3_path(temp_bitstream, runner),
        hm16_3_path(recon_yuv, runner),
        frame_rate,
        coded_width,
        coded_height,
        frame_count,
        args.qp,
    )
    if runner == "wine":
        hm_command = [wine, encoder, "-c", hm16_3_path(temp_config, runner)]
    else:
        hm_command = [encoder, "-c", temp_config]
    run_logged(hm_command, log_path, cwd=work_dir)

    if not os.path.isfile(temp_bitstream) or os.path.getsize(temp_bitstream) == 0:
        raise RuntimeError(f"HM16.3 did not produce a bitstream. See: {log_path}")
    expected_yuv_size = coded_width * coded_height * 3 // 2 * frame_count
    if not os.path.isfile(recon_yuv) or os.path.getsize(recon_yuv) != expected_yuv_size:
        actual_size = os.path.getsize(recon_yuv) if os.path.isfile(recon_yuv) else 0
        raise RuntimeError(
            "HM16.3 reconstructed YUV has an unexpected size: "
            f"expected={expected_yuv_size}, actual={actual_size}. See: {log_path}"
        )

    shutil.copy2(temp_bitstream, bitstream)
    shutil.copy2(temp_config, archived_config)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-video_size",
            f"{coded_width}x{coded_height}",
            "-framerate",
            str(frame_rate),
            "-i",
            recon_yuv,
            "-frames:v",
            str(frame_count),
            "-vf",
            f"crop={source_width}:{source_height}:0:0",
            "-vsync",
            "0",
            "-start_number",
            "0",
            encoded_pattern,
        ]
    )

    with open(protocol_path, "w", encoding="utf8") as protocol_file:
        protocol_file.write("encoder=HM16.3-standard.exe\n")
        protocol_file.write("configuration=Low-Delay P supplied package\n")
        protocol_file.write(f"encoder_path={encoder}\n")
        protocol_file.write(f"encoder_sha256={encoder_sha256}\n")
        protocol_file.write(f"runner={runner}\n")
        protocol_file.write(f"config={os.path.abspath(archived_config)}\n")
        protocol_file.write(f"qp={args.qp}\n")
        protocol_file.write(f"frames={frame_count}\n")
        protocol_file.write(f"source_size={source_width}x{source_height}\n")
        protocol_file.write(f"coded_size={coded_width}x{coded_height}\n")
        protocol_file.write(f"frame_rate={frame_rate}\n")
        protocol_file.write(f"command={shlex.join(hm_command)}\n")

    if not args.keep_hm_yuv:
        shutil.rmtree(work_dir)
    return bitstream


def prepare_with_hm(
    args,
    ffmpeg,
    raw_pattern,
    encoded_pattern,
    encoded_dir,
    frame_count,
    fps_value,
    source_size,
):
    if not os.path.isfile(args.hm_encoder):
        raise FileNotFoundError(f"Cannot find HM encoder: {args.hm_encoder}")
    if not os.path.isfile(args.hm_config):
        raise FileNotFoundError(f"Cannot find HM config: {args.hm_config}")
    if args.hm_padding < 2:
        raise ValueError("--hm-padding must be at least 2.")

    source_width, source_height = source_size
    coded_width = padded_size(source_width, args.hm_padding)
    coded_height = padded_size(source_height, args.hm_padding)
    frame_rate = hm_frame_rate(fps_value)
    work_dir = os.path.join(encoded_dir, ".hm_work")
    os.makedirs(work_dir, exist_ok=True)
    input_yuv = os.path.join(work_dir, f"input_{coded_width}x{coded_height}.yuv")
    recon_yuv = os.path.join(work_dir, f"recon_{coded_width}x{coded_height}.yuv")
    bitstream = os.path.join(encoded_dir, f"qp_{args.qp:03d}.bin")
    log_path = os.path.join(encoded_dir, "hm_encode.log")
    protocol_path = os.path.join(encoded_dir, "encoding_protocol.txt")

    print(f"Source size     : {source_width}x{source_height}")
    print(f"HM coded size   : {coded_width}x{coded_height}")
    print(f"HM frame rate   : {frame_rate}")
    run(
        [
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
            str(frame_count),
            "-vf",
            f"pad={coded_width}:{coded_height}:0:0:black",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            input_yuv,
        ]
    )
    hm_command = [
        args.hm_encoder,
        "-c",
        args.hm_config,
        "-i",
        input_yuv,
        "-b",
        bitstream,
        "-o",
        recon_yuv,
        "-wdt",
        str(coded_width),
        "-hgt",
        str(coded_height),
        "-fr",
        str(frame_rate),
        "-f",
        str(frame_count),
        "-q",
        str(args.qp),
    ]
    run_logged(hm_command, log_path)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-video_size",
            f"{coded_width}x{coded_height}",
            "-framerate",
            str(frame_rate),
            "-i",
            recon_yuv,
            "-frames:v",
            str(frame_count),
            "-vf",
            f"crop={source_width}:{source_height}:0:0",
            "-vsync",
            "0",
            "-start_number",
            "0",
            encoded_pattern,
        ]
    )

    with open(protocol_path, "w", encoding="utf8") as protocol_file:
        protocol_file.write("encoder=HM TAppEncoder\n")
        protocol_file.write(f"hm_encoder={os.path.abspath(args.hm_encoder)}\n")
        protocol_file.write(f"hm_config={os.path.abspath(args.hm_config)}\n")
        protocol_file.write(f"qp={args.qp}\n")
        protocol_file.write(f"frames={frame_count}\n")
        protocol_file.write(f"source_size={source_width}x{source_height}\n")
        protocol_file.write(f"coded_size={coded_width}x{coded_height}\n")
        protocol_file.write(f"frame_rate={frame_rate}\n")
        protocol_file.write(f"command={shlex.join(hm_command)}\n")

    if not args.keep_hm_yuv:
        shutil.rmtree(work_dir)
    return bitstream


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Cannot find LDV video: {args.input}")
    if args.frames < 3:
        raise ValueError("--frames must be at least 3.")
    if args.sequence < 0 or args.qp < 0:
        raise ValueError("--sequence and --qp must be non-negative.")
    if args.encoder in ("hm16_3_ldp", "hm") and args.qp > 51:
        raise ValueError("HM encoders require --qp in [0, 51].")

    ffmpeg = find_working_ffmpeg()
    print(f"Using ffmpeg   : {ffmpeg}")

    sequence_name = f"{args.sequence:03d}"
    raw_dir = os.path.join(args.output_root, "Raw", sequence_name)
    encoded_dir = os.path.join(
        args.output_root, "Encoded", sequence_name, f"QP-{args.qp}"
    )
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(encoded_dir, exist_ok=True)
    ensure_output_is_empty(raw_dir, encoded_dir, overwrite=args.overwrite)

    raw_pattern = os.path.join(raw_dir, "%03d.png")
    encoded_pattern = os.path.join(encoded_dir, "%03d.png")
    fps_value = (
        str(args.fps)
        if args.fps is not None
        else detect_source_frame_rate(ffmpeg, os.path.abspath(args.input))
    )
    source_size = detect_source_size(ffmpeg, os.path.abspath(args.input))
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

    frame_count = count_png_files(raw_dir)
    if frame_count < 3:
        raise RuntimeError(
            f"The source produced only {frame_count} frames; at least 3 are required."
        )
    if frame_count < args.frames:
        print(
            f"Requested {args.frames} frames, but the source contains only "
            f"{frame_count}; using all available frames."
        )

    if args.encoder == "hm16_3_ldp":
        compressed_output = prepare_with_hm16_3_ldp(
            args,
            ffmpeg,
            raw_pattern,
            encoded_pattern,
            encoded_dir,
            frame_count,
            fps_value,
            source_size,
        )
    elif args.encoder == "hm":
        compressed_output = prepare_with_hm(
            args,
            ffmpeg,
            raw_pattern,
            encoded_pattern,
            encoded_dir,
            frame_count,
            fps_value,
            source_size,
        )
    else:
        compressed_output = os.path.join(encoded_dir, f"qp_{args.qp:03d}.mp4")
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
            str(frame_count),
            *encoder_options(args.encoder, args.qp),
            compressed_output,
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
                str(frame_count),
                *encoder_options("libx265", args.qp),
                compressed_output,
            ]
            run(encode_command)

        run(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-i",
                compressed_output,
                "-map",
                "0:v:0",
                "-vsync",
                "0",
                "-frames:v",
                str(frame_count),
                "-start_number",
                "0",
                encoded_pattern,
            ]
        )

    raw_count = count_png_files(raw_dir)
    encoded_count = count_png_files(encoded_dir)
    if raw_count != frame_count or encoded_count != frame_count:
        raise RuntimeError(
            f"Frame preparation is incomplete: Raw={raw_count}, Encoded={encoded_count}, "
            f"expected={frame_count}."
        )

    print("LDV preparation finished")
    print(f"Encoder        : {args.encoder}")
    print(f"Raw frames     : {raw_dir} ({raw_count})")
    print(f"Encoded frames : {encoded_dir} ({encoded_count})")
    print(f"Compressed HEVC: {compressed_output}")
    print("Next command:")
    print(
        f"{shlex.quote(sys.executable)} {shlex.quote(os.path.join(BASE_DIR, 'test.py'))} "
        f"--sequence {args.sequence} --qp {args.qp} --fraction 1 "
        "--batch-size 1 --save-limit 3 --full-frame-metrics"
    )


if __name__ == "__main__":
    main()
