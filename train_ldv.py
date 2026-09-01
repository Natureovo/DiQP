import argparse
import csv
import json
import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import myDataset
from losses import CharbonnierLoss
from model import DiQP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 1234


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU LDV fine-tuning for the released DiQP HEVC model."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(BASE_DIR, "data", "LDV_finetune"),
    )
    parser.add_argument("--split-file", default=None)
    parser.add_argument(
        "--pretrained",
        default=os.path.join(BASE_DIR, "pretrained", "checkpoint_HEVC.pt"),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--train-scope",
        choices=("output", "decoder", "all"),
        default="output",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps; 0 runs all epochs.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return strip_module_prefix(checkpoint)


def read_split(path):
    with open(path, "r", encoding="utf8", newline="") as split_file:
        rows = list(csv.DictReader(split_file))
    if not rows:
        raise RuntimeError(f"Split manifest is empty: {path}")

    qps = {int(row["QP"]) for row in rows}
    frame_counts = {int(row["Frames Used"]) for row in rows}
    encoders = {row.get("Encoder", "unknown") or "unknown" for row in rows}
    hm_encoders = {row.get("HM Encoder", "") for row in rows}
    hm_configs = {row.get("HM Config", "") for row in rows}
    hm_paddings = {int(row.get("HM Padding", 0) or 0) for row in rows}
    if not qps or len(frame_counts) != 1:
        raise ValueError("Fine-tuning requires at least one QP and one fixed frame count.")
    if len(encoders) != 1:
        raise ValueError(f"Split manifest mixes encoder protocols: {sorted(encoders)}")
    if len(hm_encoders) != 1 or len(hm_configs) != 1 or len(hm_paddings) != 1:
        raise ValueError("Split manifest mixes HM executable/config/padding settings.")

    train_sequences = sorted(
        {int(row["Sequence"]) for row in rows if row["Split"] == "train"}
    )
    val_sequences = sorted(
        {int(row["Sequence"]) for row in rows if row["Split"] == "val"}
    )
    if not train_sequences or not val_sequences:
        raise ValueError("Split manifest must contain train and val sequences.")
    expected_pairs = {
        (sequence, qp)
        for sequence in train_sequences + val_sequences
        for qp in qps
    }
    actual_pairs = {
        (int(row["Sequence"]), int(row["QP"]))
        for row in rows
        if row["Split"] in ("train", "val")
    }
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if missing_pairs:
        raise ValueError(f"Split manifest is missing sequence/QP pairs: {missing_pairs}")
    return (
        train_sequences,
        val_sequences,
        sorted(qps),
        frame_counts.pop(),
        encoders.pop(),
        hm_encoders.pop(),
        hm_configs.pop(),
        hm_paddings.pop(),
    )


def detect_geometry(data_root, sequence):
    raw_dir = os.path.join(data_root, "Raw", f"{sequence:03d}")
    candidates = [
        (os.path.join(raw_dir, "000.png"), ""),
        (os.path.join(raw_dir, "000_8K.png"), "_8K"),
    ]
    for path, suffix in candidates:
        image = cv2.imread(path)
        if image is not None:
            height, width = image.shape[:2]
            return width, height, suffix
    raise FileNotFoundError(f"Could not read frame 000 under: {raw_dir}")


def validate_prepared_data(data_root, sequences, qps, frames, raw_suffix):
    for sequence in sequences:
        raw_dir = os.path.join(data_root, "Raw", f"{sequence:03d}")
        for qp in qps:
            encoded_dir = os.path.join(
                data_root, "Encoded", f"{sequence:03d}", f"QP-{qp}"
            )
            for frame in range(frames):
                raw_frame = os.path.join(raw_dir, f"{frame:03d}{raw_suffix}.png")
                encoded_frame = os.path.join(encoded_dir, f"{frame:03d}.png")
                if not os.path.isfile(raw_frame) or not os.path.isfile(encoded_frame):
                    raise FileNotFoundError(
                        f"Missing aligned frame for sequence {sequence:03d}, "
                        f"QP {qp}, frame {frame:03d}."
                    )


def build_model():
    return DiQP(
        crop_size=512,
        embed_dim=15,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        win_size=8,
        mlp_ratio=3.0,
        token_projection="linear",
        token_mlp="steff",
        shift_flag=True,
    )


def configure_train_scope(model, scope):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if scope == "all":
        prefixes = ("",)
    elif scope == "decoder":
        prefixes = ("decoderlayer_", "upsample_", "output_proj")
    else:
        prefixes = ("output_proj",)

    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"No parameters selected by train scope: {scope}")
    return trainable


def set_training_mode(model, scope):
    model.train()
    if scope == "all":
        return
    model.eval()
    for module in model.modules():
        if any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
            module.train()


def move_batch(batch, device):
    x, y, around, ahead_cropped, ahead_scaled, loc, log, decay = batch
    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        around.to(device, non_blocking=True),
        ahead_cropped.to(device, non_blocking=True),
        ahead_scaled.to(device, non_blocking=True),
        loc.to(device, non_blocking=True).permute(1, 0, 2),
        decay.to(device, non_blocking=True).view(-1, 1, 1, 1, 1),
        log,
    )


def sample_psnr(prediction, reference):
    mse = torch.mean((prediction - reference) ** 2, dim=(1, 2, 3, 4))
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-12))


def batch_qps(log):
    if not isinstance(log, (list, tuple)) or len(log) < 3:
        raise RuntimeError("Dataset log does not contain QP values.")
    values = log[2]
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
    if isinstance(values, np.ndarray):
        return [int(value) for value in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [int(values)]


def validate(model, loader, loss_fn, device, amp_enabled):
    model.eval()
    loss_sum = 0.0
    predicted_psnr_sum = 0.0
    base_psnr_sum = 0.0
    sample_count = 0
    per_qp = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            x, y, around, ahead_cropped, ahead_scaled, loc, decay, log = move_batch(
                batch, device
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(x, around, ahead_cropped, ahead_scaled, loc, decay)
                loss = loss_fn(output, y)
            output = torch.clamp(output, 0, 1)
            current_batch = x.shape[0]
            predicted_values = sample_psnr(output, y).detach().cpu().tolist()
            base_values = sample_psnr(x, y).detach().cpu().tolist()
            qps = batch_qps(log)
            if len(qps) != current_batch:
                raise RuntimeError(
                    f"Validation QP count mismatch: qps={len(qps)}, batch={current_batch}"
                )
            loss_sum += loss.item() * current_batch
            predicted_psnr_sum += sum(predicted_values)
            base_psnr_sum += sum(base_values)
            sample_count += current_batch
            for qp, predicted_value, base_value in zip(
                qps, predicted_values, base_values
            ):
                bucket = per_qp.setdefault(
                    qp, {"predicted_sum": 0.0, "base_sum": 0.0, "samples": 0}
                )
                bucket["predicted_sum"] += predicted_value
                bucket["base_sum"] += base_value
                bucket["samples"] += 1
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty.")
    predicted_psnr = predicted_psnr_sum / sample_count
    base_psnr = base_psnr_sum / sample_count
    per_qp_metrics = {}
    for qp, bucket in sorted(per_qp.items()):
        qp_predicted = bucket["predicted_sum"] / bucket["samples"]
        qp_base = bucket["base_sum"] / bucket["samples"]
        per_qp_metrics[qp] = {
            "psnr_predicted": qp_predicted,
            "psnr_base": qp_base,
            "psnr_gain": qp_predicted - qp_base,
            "samples": bucket["samples"],
        }
    return {
        "loss": loss_sum / sample_count,
        "psnr_predicted": predicted_psnr,
        "psnr_base": base_psnr,
        "psnr_gain": predicted_psnr - base_psnr,
        "samples": sample_count,
        "per_qp": per_qp_metrics,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, state):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        **state,
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LDV fine-tuning requires a CUDA GPU.")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid epochs, batch size, or workers value.")
    if not 0 < args.val_fraction <= 1:
        raise ValueError("--val-fraction must be in (0, 1].")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative.")

    set_seed(SEED)
    device = torch.device("cuda")
    split_path = args.split_file or os.path.join(args.data_root, "split.csv")
    (
        train_sequences,
        val_sequences,
        qps,
        frames,
        encoder,
        hm_encoder,
        hm_config,
        hm_padding,
    ) = read_split(split_path)
    qp_conditions = {qp: qp // 3 for qp in qps}
    if len(set(qp_conditions.values())) != len(qp_conditions):
        raise ValueError(
            "QP values collide after DiQP qp//3 conditioning: "
            f"{qp_conditions}"
        )
    width, height, raw_suffix = detect_geometry(args.data_root, train_sequences[0])
    if width < 512 or height < 512:
        raise RuntimeError(f"Frames are too small for DiQP: {width}x{height}")
    validate_prepared_data(
        args.data_root,
        train_sequences + val_sequences,
        qps,
        frames,
        raw_suffix,
    )

    trainset = myDataset(
        seqNumbers=train_sequences,
        numOfFramesPerSeq=frames,
        rawPath=os.path.join(args.data_root, "Raw"),
        qpPath=os.path.join(args.data_root, "Encoded"),
        extractingMethod="even",
        totalQualities=qps,
        cropSize=512,
        frac=1,
        random_state=SEED,
        augmentation=False,
        train=True,
        height=height,
        width=width,
        visibleHeight=height,
        visibleWidth=width,
        rawFrameSuffix=raw_suffix,
    )
    valset = myDataset(
        seqNumbers=val_sequences,
        numOfFramesPerSeq=frames,
        rawPath=os.path.join(args.data_root, "Raw"),
        qpPath=os.path.join(args.data_root, "Encoded"),
        extractingMethod="even",
        totalQualities=qps,
        cropSize=512,
        frac=args.val_fraction,
        random_state=SEED,
        augmentation=False,
        train=False,
        height=height,
        width=width,
        visibleHeight=height,
        visibleWidth=width,
        rawFrameSuffix=raw_suffix,
    )
    generator = torch.Generator().manual_seed(SEED)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
    )
    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_model().to(device)
    if not os.path.isfile(args.pretrained):
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {args.pretrained}")
    pretrained = torch.load(args.pretrained, map_location="cpu")
    model.load_state_dict(checkpoint_state_dict(pretrained), strict=True)
    trainable = configure_train_scope(model, args.train_scope)
    optimizer = optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    amp_enabled = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_fn = CharbonnierLoss().to(device)

    start_epoch = 0
    global_step = 0
    best_gain = float("-inf")
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu")
        if resume.get("train_scope") != args.train_scope:
            raise ValueError(
                "Resume checkpoint train scope differs from --train-scope: "
                f"{resume.get('train_scope')} vs {args.train_scope}."
            )
        resume_qps = resume.get("qps")
        if resume_qps is None and resume.get("qp") is not None:
            resume_qps = [int(resume["qp"])]
        if sorted(resume_qps or []) != qps or int(resume.get("frames", -1)) != frames:
            raise ValueError("Resume checkpoint QP/frame settings differ from the split.")
        resume_encoder = resume.get("encoder", "unknown")
        if resume_encoder != encoder:
            raise ValueError(
                "Resume checkpoint encoder differs from the split: "
                f"{resume_encoder} vs {encoder}."
            )
        if encoder == "hm" and (
            resume.get("hm_encoder", "") != hm_encoder
            or resume.get("hm_config", "") != hm_config
            or int(resume.get("hm_padding", -1)) != hm_padding
        ):
            raise ValueError("Resume checkpoint HM executable/config/padding differs from the split.")
        model.load_state_dict(checkpoint_state_dict(resume), strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume.get("epoch", -1)) + 1
        global_step = int(resume.get("global_step", 0))
        best_gain = float(resume.get("best_val_psnr_gain", best_gain))

    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    elif args.resume is not None:
        output_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        run_name = args.run_name or datetime.now().strftime("ldv_%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_DIR, "runs", "ldv_finetune", run_name)
    if os.path.isdir(output_dir) and args.resume is None:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        **vars(args),
        "split_file": os.path.abspath(split_path),
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "qps": qps,
        "qp_conditions": qp_conditions,
        "frames": frames,
        "encoder": encoder,
        "hm_encoder": hm_encoder,
        "hm_config": hm_config,
        "hm_padding": hm_padding,
        "resolution": f"{width}x{height}",
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)

    history_path = os.path.join(output_dir, "history.csv")
    history_exists = os.path.isfile(history_path)
    history_file = open(history_path, "a", encoding="utf8", newline="")
    history = csv.writer(history_file)
    if not history_exists:
        history.writerow(
            [
                "Epoch",
                "Global Step",
                "Train Loss",
                "Val Loss",
                "Val PSNR Predicted",
                "Val PSNR Base",
                "Val PSNR Gain",
                "Learning Rate",
            ]
        )
    qp_history_path = os.path.join(output_dir, "validation_by_qp.csv")
    qp_history_exists = os.path.isfile(qp_history_path)
    qp_history_file = open(qp_history_path, "a", encoding="utf8", newline="")
    qp_history = csv.writer(qp_history_file)
    if not qp_history_exists:
        qp_history.writerow(
            [
                "Epoch",
                "Global Step",
                "QP",
                "Val PSNR Predicted",
                "Val PSNR Base",
                "Val PSNR Gain",
                "Samples",
            ]
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    print(f"Device            : {torch.cuda.get_device_name(0)}")
    print(f"Data / resolution : {args.data_root} / {width}x{height}")
    print(f"Train / val videos: {len(train_sequences)} / {len(val_sequences)}")
    print(f"Train / val samples: {len(trainset)} / {len(valset)}")
    print(f"QPs / frames      : {qps} / {frames}")
    print(f"QP conditions     : {qp_conditions}")
    print(f"Encoder           : {encoder}")
    if encoder == "hm":
        print(f"HM config         : {hm_config}")
        print(f"HM padding        : {hm_padding}")
    print(f"Train scope       : {args.train_scope}")
    print(f"Parameters        : {trainable_parameters:,} trainable / {total_parameters:,}")
    print(f"Output directory  : {output_dir}")

    if start_epoch == 0 and not history_exists:
        baseline = validate(model, valloader, loss_fn, device, amp_enabled)
        best_gain = baseline["psnr_gain"]
        baseline_state = {
            "epoch": -1,
            "global_step": 0,
            "best_val_psnr_gain": best_gain,
            "train_scope": args.train_scope,
            "qps": qps,
            "frames": frames,
            "encoder": encoder,
            "hm_encoder": hm_encoder,
            "hm_config": hm_config,
            "hm_padding": hm_padding,
        }
        save_checkpoint(
            os.path.join(output_dir, "best.pt"),
            model,
            optimizer,
            scheduler,
            scaler,
            baseline_state,
        )
        history.writerow(
            [
                0,
                0,
                "",
                baseline["loss"],
                baseline["psnr_predicted"],
                baseline["psnr_base"],
                baseline["psnr_gain"],
                optimizer.param_groups[0]["lr"],
            ]
        )
        for qp, qp_metrics in baseline["per_qp"].items():
            qp_history.writerow(
                [
                    0,
                    0,
                    qp,
                    qp_metrics["psnr_predicted"],
                    qp_metrics["psnr_base"],
                    qp_metrics["psnr_gain"],
                    qp_metrics["samples"],
                ]
            )
        history_file.flush()
        qp_history_file.flush()
        print(f"Validation baseline: {best_gain:+.4f} dB PSNR gain")
        print(
            "Per-QP baseline : "
            + ", ".join(
                f"QP {qp}={values['psnr_gain']:+.4f} dB"
                for qp, values in baseline["per_qp"].items()
            )
        )

    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        set_training_mode(model, args.train_scope)
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            x, y, around, ahead_cropped, ahead_scaled, loc, decay, _ = move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(x, around, ahead_cropped, ahead_scaled, loc, decay)
                loss = loss_fn(output, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.gradient_clip > 0:
                nn_utils.clip_grad_norm_(trainable, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            current_batch = x.shape[0]
            loss_sum += loss.item() * current_batch
            sample_count += current_batch
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.5f}")
            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break

        train_loss = loss_sum / max(sample_count, 1)
        metrics = validate(model, valloader, loss_fn, device, amp_enabled)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_psnr_gain": max(best_gain, metrics["psnr_gain"]),
            "train_scope": args.train_scope,
            "qps": qps,
            "frames": frames,
            "encoder": encoder,
            "hm_encoder": hm_encoder,
            "hm_config": hm_config,
            "hm_padding": hm_padding,
        }
        save_checkpoint(
            os.path.join(output_dir, "last.pt"),
            model,
            optimizer,
            scheduler,
            scaler,
            state,
        )
        if metrics["psnr_gain"] > best_gain:
            best_gain = metrics["psnr_gain"]
            state["best_val_psnr_gain"] = best_gain
            save_checkpoint(
                os.path.join(output_dir, "best.pt"),
                model,
                optimizer,
                scheduler,
                scaler,
                state,
            )

        history.writerow(
            [
                epoch + 1,
                global_step,
                train_loss,
                metrics["loss"],
                metrics["psnr_predicted"],
                metrics["psnr_base"],
                metrics["psnr_gain"],
                learning_rate,
            ]
        )
        for qp, qp_metrics in metrics["per_qp"].items():
            qp_history.writerow(
                [
                    epoch + 1,
                    global_step,
                    qp,
                    qp_metrics["psnr_predicted"],
                    qp_metrics["psnr_base"],
                    qp_metrics["psnr_gain"],
                    qp_metrics["samples"],
                ]
            )
        history_file.flush()
        qp_history_file.flush()
        print(
            f"Epoch {epoch + 1}: train loss={train_loss:.6f}, "
            f"val gain={metrics['psnr_gain']:+.4f} dB, "
            f"best={best_gain:+.4f} dB"
        )
        print(
            "Per-QP gains     : "
            + ", ".join(
                f"QP {qp}={values['psnr_gain']:+.4f} dB"
                for qp, values in metrics["per_qp"].items()
            )
        )
        if stop_training:
            break

    history_file.close()
    qp_history_file.close()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("Fine-tuning finished")
    print(f"Best checkpoint : {os.path.join(output_dir, 'best.pt')}")
    print(f"Peak GPU memory : {peak_memory:.2f} GiB")


if __name__ == "__main__":
    main()
