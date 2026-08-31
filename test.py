import argparse
import csv
import os
import random
import re
import warnings
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import myDataset
from model import DiQP
from utils.frame_utils import batch_ssim, calcPSNR, reorder_image

warnings.filterwarnings("ignore")

SEED = 1234
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@contextmanager
def null_context():
    yield


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DiQP on SEPE-8K, custom video, or prepared LDV frames."
    )
    parser.add_argument("--raw-path", default=os.path.join(BASE_DIR, "data", "Raw"))
    parser.add_argument("--encoded-path", default=os.path.join(BASE_DIR, "data", "Encoded"))
    parser.add_argument(
        "--model-path",
        default=os.path.join(BASE_DIR, "pretrained", "checkpoint_HEVC.pt"),
    )
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--qp", type=int, default=None, help="Auto-detect when omitted.")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument(
        "--save-limit",
        type=int,
        default=6,
        help="Number of middle-frame visual comparisons to save; use 0 to disable.",
    )
    parser.add_argument("--results-dir", default=os.path.join(BASE_DIR, "testResults"))
    parser.add_argument("--report", default=os.path.join(BASE_DIR, "report_v2.csv"))
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_reproducible_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def strip_module_prefix(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def resolve_seq_dir(root, seq_num):
    candidates = [
        os.path.join(root, f"{seq_num:03d}"),
        os.path.join(root, str(seq_num)),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Cannot find sequence directory. Tried: {candidates}")


def resolve_qp_dir(encoded_root, seq_num, preferred=None):
    seq_dir = resolve_seq_dir(encoded_root, seq_num)
    qp_dirs = sorted(
        directory
        for directory in os.listdir(seq_dir)
        if os.path.isdir(os.path.join(seq_dir, directory)) and directory.startswith("QP-")
    )
    if not qp_dirs:
        raise FileNotFoundError(f"No QP-* directory found under: {seq_dir}")

    if preferred is not None:
        qp_dir_name = f"QP-{preferred}"
        if qp_dir_name not in qp_dirs:
            raise FileNotFoundError(
                f"{qp_dir_name} not found under {seq_dir}. Available: {qp_dirs}"
            )
    else:
        qp_dir_name = qp_dirs[0]
        if len(qp_dirs) > 1:
            print(f"Found multiple QP folders: {qp_dirs}. Using {qp_dir_name}.")

    match = re.fullmatch(r"QP-(\d+)", qp_dir_name)
    if match is None:
        raise RuntimeError(f"QP folder name must look like QP-37, got: {qp_dir_name}")
    return int(match.group(1)), os.path.join(seq_dir, qp_dir_name)


def count_contiguous_frames(raw_seq_dir, encoded_seq_dir, frame_limit=None):
    suffix_counts = {}
    for raw_suffix in ("_8K", ""):
        count = 0
        while True:
            raw_frame = os.path.join(raw_seq_dir, f"{count:03d}{raw_suffix}.png")
            encoded_frame = os.path.join(encoded_seq_dir, f"{count:03d}.png")
            if not os.path.isfile(raw_frame) or not os.path.isfile(encoded_frame):
                break
            count += 1
        suffix_counts[raw_suffix] = count

    raw_suffix = max(suffix_counts, key=suffix_counts.get)
    frame_count = suffix_counts[raw_suffix]
    if frame_limit is not None:
        if frame_limit < 3:
            raise ValueError("--frame-limit must be at least 3.")
        frame_count = min(frame_count, frame_limit)
    if frame_count < 3:
        raise RuntimeError(
            "At least 3 aligned frames numbered from 000 are required. "
            f"Found counts by Raw suffix: {suffix_counts}"
        )
    return frame_count, raw_suffix


def detect_frame_geometry(raw_seq_dir, encoded_seq_dir, raw_suffix, crop_size):
    raw_frame = cv2.imread(os.path.join(raw_seq_dir, f"000{raw_suffix}.png"))
    encoded_frame = cv2.imread(os.path.join(encoded_seq_dir, "000.png"))
    if raw_frame is None or encoded_frame is None:
        raise RuntimeError("Could not read frame 000 from Raw and Encoded directories.")
    if raw_frame.shape != encoded_frame.shape:
        raise RuntimeError(
            f"Raw and Encoded dimensions differ: {raw_frame.shape} vs {encoded_frame.shape}"
        )

    height, width = raw_frame.shape[:2]
    visible_height = (height // crop_size) * crop_size
    visible_width = (width // crop_size) * crop_size
    if visible_height < crop_size or visible_width < crop_size:
        raise RuntimeError(
            f"Frame size {width}x{height} is smaller than crop size {crop_size}."
        )
    return width, height, visible_width, visible_height


def load_checkpoint(path, target_device):
    checkpoint = torch.load(path, map_location=target_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return strip_module_prefix(checkpoint)


def save_visual_comparisons(output, encoded, raw, result_dir, saved, limit):
    if saved >= limit:
        return saved

    output_vis = reorder_image(output)
    encoded_vis = reorder_image(encoded)
    raw_vis = reorder_image(raw)
    middle_frame = output_vis.shape[1] // 2

    for sample_index in range(output_vis.shape[0]):
        if saved >= limit:
            break
        prediction = output_vis[sample_index, middle_frame]
        compressed = encoded_vis[sample_index, middle_frame]
        reference = raw_vis[sample_index, middle_frame]

        save_image(compressed, os.path.join(result_dir, f"x_{saved:03d}.png"))
        save_image(prediction, os.path.join(result_dir, f"o_{saved:03d}.png"))
        save_image(reference, os.path.join(result_dir, f"y_{saved:03d}.png"))
        save_image(
            torch.cat([compressed, prediction, reference], dim=2),
            os.path.join(result_dir, f"comparison_{saved:03d}.png"),
        )

        encoded_error = torch.clamp(torch.abs(compressed - reference) * 5.0, 0, 1)
        predicted_error = torch.clamp(torch.abs(prediction - reference) * 5.0, 0, 1)
        save_image(
            torch.cat([encoded_error, predicted_error], dim=2),
            os.path.join(result_dir, f"difference_x5_{saved:03d}.png"),
        )
        saved += 1
    return saved


def append_report(path, row):
    field_names = [
        "Dataset",
        "Sequence",
        "Resolution",
        "QP",
        "Model",
        "Samples",
        "PSNR Predicted",
        "PSNR Base",
        "PSNR Gain",
        "SSIM Predicted",
        "SSIM Base",
        "SSIM Gain",
    ]
    file_exists = os.path.isfile(path)
    with open(path, "a", encoding="utf8", newline="") as report_file:
        writer = csv.writer(report_file)
        if not file_exists:
            writer.writerow(field_names)
        writer.writerow(row)


def main():
    args = parse_args()
    set_reproducible_seed()

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and no greater than 1.")
    if args.crop_size != 512:
        raise ValueError("The released pretrained checkpoints require --crop-size 512.")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Cannot find model checkpoint: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_seq_dir = resolve_seq_dir(args.raw_path, args.sequence)
    qp_value, encoded_seq_dir = resolve_qp_dir(args.encoded_path, args.sequence, args.qp)
    frame_count, raw_suffix = count_contiguous_frames(
        raw_seq_dir, encoded_seq_dir, args.frame_limit
    )
    if frame_count > 300:
        print("The pretrained frame embedding supports 300 frames; using the first 300.")
        frame_count = 300
    if qp_value > 255:
        raise ValueError("The released pretrained model only supports QP values up to 255.")
    width, height, visible_width, visible_height = detect_frame_geometry(
        raw_seq_dir, encoded_seq_dir, raw_suffix, args.crop_size
    )

    print(f"Raw dir        : {raw_seq_dir}")
    print(f"Encoded dir    : {encoded_seq_dir}")
    print(f"QP value       : {qp_value}")
    print(f"Frames used    : {frame_count}")
    print(f"Frame size     : {width}x{height}")
    print(f"Test window    : {visible_width}x{visible_height}")
    print(f"Raw suffix     : {raw_suffix or '(none)'}")
    print(f"Device         : {device}")

    testset = myDataset(
        seqNumbers=[args.sequence],
        numOfFramesPerSeq=frame_count,
        rawPath=args.raw_path,
        qpPath=args.encoded_path,
        extractingMethod="even",
        totalQualities=qp_value,
        cropSize=args.crop_size,
        frac=args.fraction,
        random_state=SEED,
        augmentation=False,
        train=False,
        height=height,
        width=width,
        visibleHeight=visible_height,
        visibleWidth=visible_width,
        rawFrameSuffix=raw_suffix,
    )
    if len(testset) == 0:
        raise RuntimeError("The test set is empty. Increase --fraction or frame count.")
    print(f"Test samples   : {len(testset)}")

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    model = DiQP(
        crop_size=args.crop_size,
        embed_dim=15,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        win_size=8,
        mlp_ratio=3.0,
        token_projection="linear",
        token_mlp="steff",
        shift_flag=True,
    ).to(device)
    model.load_state_dict(load_checkpoint(args.model_path, device), strict=True)
    model.eval()

    if args.save_limit > 0:
        os.makedirs(args.results_dir, exist_ok=True)
        print("Comparison      : left=Encoded, middle=Model, right=Raw")
        print("Difference x5   : left=Encoded error, right=Model error")

    metric_sums = np.zeros(10, dtype=np.float64)
    evaluated_samples = 0
    visualizations_saved = 0

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            x_cropped, y_cropped, around, ahead_cropped, ahead_scaled, loc, _, decay = batch
            current_batch_size = x_cropped.shape[0]

            x_cropped = x_cropped.to(device, non_blocking=True)
            y_cropped = y_cropped.to(device, non_blocking=True)
            around = around.to(device, non_blocking=True)
            ahead_cropped = ahead_cropped.to(device, non_blocking=True)
            ahead_scaled = ahead_scaled.to(device, non_blocking=True)
            loc = loc.to(device, non_blocking=True).permute(1, 0, 2)
            decay = decay.to(device, non_blocking=True).view(-1, 1, 1, 1, 1)

            amp_context = (
                torch.cuda.amp.autocast(enabled=not args.no_amp)
                if device.type == "cuda"
                else null_context()
            )
            with amp_context:
                output = model(x_cropped, around, ahead_cropped, ahead_scaled, loc, decay)
            output = torch.clamp(output, 0, 1)

            output_cpu = output.detach().cpu()
            raw_cpu = y_cropped.detach().cpu()
            encoded_cpu = x_cropped.detach().cpu()

            predicted_psnr = calcPSNR(output_cpu.numpy(), raw_cpu.numpy(), data_range=1.0)
            base_psnr = calcPSNR(encoded_cpu.numpy(), raw_cpu.numpy(), data_range=1.0)
            predicted_ssim = batch_ssim(
                output_cpu.numpy(), raw_cpu.numpy(), data_range=1.0
            )
            base_ssim = batch_ssim(
                encoded_cpu.numpy(), raw_cpu.numpy(), data_range=1.0
            )
            batch_metrics = np.array(
                [*predicted_psnr, *base_psnr, predicted_ssim, base_ssim],
                dtype=np.float64,
            )
            metric_sums += batch_metrics * current_batch_size
            evaluated_samples += current_batch_size

            if args.save_limit > 0:
                visualizations_saved = save_visual_comparisons(
                    output_cpu,
                    encoded_cpu,
                    raw_cpu,
                    args.results_dir,
                    visualizations_saved,
                    args.save_limit,
                )

    if evaluated_samples == 0:
        raise RuntimeError("No samples were evaluated.")

    metrics = metric_sums / evaluated_samples
    psnr_predicted = metrics[3]
    psnr_base = metrics[7]
    ssim_predicted = metrics[8]
    ssim_base = metrics[9]
    psnr_gain = psnr_predicted - psnr_base
    ssim_gain = ssim_predicted - ssim_base

    append_report(
        args.report,
        [
            encoded_seq_dir,
            args.sequence,
            f"{width}x{height}",
            qp_value,
            args.model_path,
            evaluated_samples,
            psnr_predicted,
            psnr_base,
            psnr_gain,
            ssim_predicted,
            ssim_base,
            ssim_gain,
        ],
    )

    print("Testing finished")
    print(f"Evaluated samples: {evaluated_samples}")
    print(f"PSNR Predicted : {psnr_predicted:.4f} dB")
    print(f"PSNR Base      : {psnr_base:.4f} dB")
    print(f"PSNR Gain      : {psnr_gain:+.4f} dB")
    print(f"SSIM Predicted : {ssim_predicted:.6f}")
    print(f"SSIM Base      : {ssim_base:.6f}")
    print(f"SSIM Gain      : {ssim_gain:+.6f}")
    print(f"Report         : {args.report}")
    if visualizations_saved:
        print(f"Visualizations : {visualizations_saved} saved to {args.results_dir}")


if __name__ == "__main__":
    main()
