import argparse
import csv
import json
import os
import subprocess
import sys


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
        description="Prepare a small fixed LDV split for DiQP fine-tuning."
    )
    parser.add_argument(
        "--ldv-dir",
        default="/home/cp/datasets/LDV1/training_raw",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(BASE_DIR, "data", "LDV_finetune"),
    )
    parser.add_argument(
        "--train-ids",
        type=int,
        nargs="+",
        default=list(range(21, 31)),
    )
    parser.add_argument("--val-ids", type=int, nargs="+", default=[31, 32, 33])
    parser.add_argument(
        "--qps",
        type=int,
        nargs="+",
        default=[22, 27, 32, 37, 42, 51],
        help="HM QPs to prepare. Defaults to the common benchmark set plus QP 51.",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=None,
        help="Backward-compatible single-QP override.",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--encoder",
        choices=("hm16_3_ldp", "hm", "hevc_nvenc", "libx265"),
        default="hm16_3_ldp",
    )
    parser.add_argument("--hm16-3-encoder", default=DEFAULT_HM16_3_ENCODER)
    parser.add_argument(
        "--hm16-3-runner",
        choices=("auto", "direct", "wine"),
        default="auto",
    )
    parser.add_argument("--wine", default=os.environ.get("DIQP_WINE", "wine"))
    parser.add_argument("--hm16-3-padding", type=int, default=8)
    parser.add_argument(
        "--hm16-3-work-root",
        default=os.environ.get("DIQP_HM16_3_WORK_ROOT", "/tmp/diqp_hm16_3"),
    )
    parser.add_argument("--hm-encoder", default=DEFAULT_HM_ENCODER)
    parser.add_argument("--hm-config", default=DEFAULT_HM_CONFIG)
    parser.add_argument("--hm-padding", type=int, default=32)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate pairs even when a matching completion marker exists.",
    )
    args = parser.parse_args()
    if args.qp is not None:
        args.qps = [args.qp]
    args.qps = sorted(set(args.qps))
    return args


def count_png_files(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(name.lower().endswith(".png") for name in os.listdir(directory))


def marker_path(output_root, sequence, qp):
    return os.path.join(output_root, ".prepared", f"{sequence:03d}_QP-{qp}.json")


def load_marker(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf8") as marker_file:
        return json.load(marker_file)


def hm_protocol_values(args):
    if args.encoder == "hm16_3_ldp":
        return (
            os.path.abspath(args.hm16_3_encoder),
            "python-generated supplied-package Low-Delay P",
            args.hm16_3_padding,
        )
    if args.encoder == "hm":
        return (
            os.path.abspath(args.hm_encoder),
            os.path.abspath(args.hm_config),
            args.hm_padding,
        )
    return "", "", 0


def prepared_pair(
    output_root,
    source,
    sequence,
    qp,
    requested_frames,
    encoder,
    hm_encoder,
    hm_config,
    hm_padding,
    hm16_3_runner,
    wine,
):
    marker = load_marker(marker_path(output_root, sequence, qp))
    if marker is None:
        return None
    if (
        marker.get("source") != os.path.abspath(source)
        or int(marker.get("qp", -1)) != qp
        or int(marker.get("requested_frames", -1)) != requested_frames
        or marker.get("encoder") != encoder
    ):
        return None
    if encoder.startswith("hm") and (
        marker.get("hm_encoder") != hm_encoder
        or marker.get("hm_config") != hm_config
        or int(marker.get("hm_padding", -1)) != hm_padding
    ):
        return None
    if encoder == "hm16_3_ldp" and (
        marker.get("hm16_3_runner") != hm16_3_runner
        or marker.get("wine") != wine
    ):
        return None

    raw_dir = os.path.join(output_root, "Raw", f"{sequence:03d}")
    encoded_dir = os.path.join(
        output_root, "Encoded", f"{sequence:03d}", f"QP-{qp}"
    )
    actual_frames = int(marker.get("actual_frames", 0))
    if (
        actual_frames >= 3
        and count_png_files(raw_dir) == actual_frames
        and count_png_files(encoded_dir) == actual_frames
    ):
        return actual_frames
    return None


def prepare_pair(args, split, video_id, qp):
    source = os.path.join(args.ldv_dir, f"{video_id:03d}.mkv")
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Cannot find LDV source video: {source}")

    hm_encoder, hm_config, hm_padding = hm_protocol_values(args)
    if not args.overwrite:
        actual_frames = prepared_pair(
            args.output_root,
            source,
            video_id,
            qp,
            args.frames,
            args.encoder,
            hm_encoder,
            hm_config,
            hm_padding,
            args.hm16_3_runner,
            args.wine,
        )
        if actual_frames is not None:
            print(
                f"Skipping prepared {split} video {video_id:03d}: "
                f"QP {qp}, {actual_frames} frames"
            )
            return actual_frames

    command = [
        sys.executable,
        os.path.join(BASE_DIR, "prepare_ldv.py"),
        "--input",
        source,
        "--output-root",
        args.output_root,
        "--sequence",
        str(video_id),
        "--qp",
        str(qp),
        "--frames",
        str(args.frames),
        "--encoder",
        args.encoder,
        "--hm16-3-encoder",
        args.hm16_3_encoder,
        "--hm16-3-runner",
        args.hm16_3_runner,
        "--wine",
        args.wine,
        "--hm16-3-padding",
        str(args.hm16_3_padding),
        "--hm16-3-work-root",
        args.hm16_3_work_root,
        "--hm-encoder",
        args.hm_encoder,
        "--hm-config",
        args.hm_config,
        "--hm-padding",
        str(args.hm_padding),
        "--overwrite",
    ]
    if args.fps is not None:
        command.extend(["--fps", str(args.fps)])

    print(f"\nPreparing {split} video {video_id:03d}, QP {qp}", flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)

    raw_dir = os.path.join(args.output_root, "Raw", f"{video_id:03d}")
    encoded_dir = os.path.join(
        args.output_root, "Encoded", f"{video_id:03d}", f"QP-{qp}"
    )
    raw_count = count_png_files(raw_dir)
    encoded_count = count_png_files(encoded_dir)
    if raw_count < 3 or raw_count != encoded_count:
        raise RuntimeError(
            f"Prepared pair is not aligned for {video_id:03d}: "
            f"Raw={raw_count}, Encoded={encoded_count}."
        )

    marker = {
        "source": os.path.abspath(source),
        "split": split,
        "sequence": video_id,
        "qp": qp,
        "requested_frames": args.frames,
        "actual_frames": raw_count,
        "encoder": args.encoder,
        "hm_encoder": hm_encoder,
        "hm_config": hm_config,
        "hm_padding": hm_padding,
        "hm16_3_runner": args.hm16_3_runner if args.encoder == "hm16_3_ldp" else "",
        "wine": args.wine if args.encoder == "hm16_3_ldp" else "",
    }
    path = marker_path(args.output_root, video_id, qp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as marker_file:
        json.dump(marker, marker_file, indent=2)
    return raw_count


def main():
    args = parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3.")
    max_qp = 51 if args.encoder.startswith("hm") else 255
    if not args.qps or any(qp < 0 or qp > max_qp for qp in args.qps):
        raise ValueError(f"--qps must contain values in [0, {max_qp}].")
    overlap = sorted(set(args.train_ids) & set(args.val_ids))
    if overlap:
        raise ValueError(f"Train and validation IDs overlap: {overlap}")

    os.makedirs(args.output_root, exist_ok=True)
    hm_encoder, hm_config, hm_padding = hm_protocol_values(args)
    rows = []
    for split, video_ids in (("train", args.train_ids), ("val", args.val_ids)):
        for video_id in video_ids:
            for qp in args.qps:
                actual_frames = prepare_pair(args, split, video_id, qp)
                if actual_frames < args.frames:
                    raise RuntimeError(
                        f"Video {video_id:03d} has only {actual_frames} frames, fewer than "
                        f"the fixed training length {args.frames}. Choose a smaller --frames value."
                    )
                rows.append(
                    [
                        video_id,
                        split,
                        qp,
                        args.frames,
                        actual_frames,
                        args.encoder,
                        hm_encoder,
                        hm_config,
                        hm_padding,
                    ]
                )

    split_path = os.path.join(args.output_root, "split.csv")
    with open(split_path, "w", encoding="utf8", newline="") as split_file:
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

    print("\nLDV fine-tuning data is ready")
    print(f"Train videos   : {len(args.train_ids)}")
    print(f"Val videos     : {len(args.val_ids)}")
    print(f"QPs / frames   : {args.qps} / {args.frames}")
    print(f"Encoder        : {args.encoder}")
    print(f"Data root      : {args.output_root}")
    print(f"Split manifest : {split_path}")


if __name__ == "__main__":
    main()
