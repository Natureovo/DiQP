import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from statistics import mean


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a multi-video, multi-QP LDV zero-shot evaluation."
    )
    parser.add_argument(
        "--ldv-dir",
        default="/home/cp/datasets/LDV1/training_raw",
        help="Directory containing numbered LDV MKV files.",
    )
    parser.add_argument("--video-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--qps", type=int, nargs="+", default=[32, 37, 42])
    parser.add_argument("--start-sequence", type=int, default=101)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-limit", type=int, default=2)
    parser.add_argument(
        "--metric-mode",
        choices=("full-frame", "patch"),
        default="full-frame",
    )
    parser.add_argument(
        "--encoder",
        choices=("hevc_nvenc", "libx265"),
        default="hevc_nvenc",
    )
    parser.add_argument("--output-root", default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--run-root", default=os.path.join(BASE_DIR, "batchResults"))
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Existing run directory; completed Sequence/QP pairs are skipped.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately instead of recording a failed video/QP and continuing.",
    )
    return parser.parse_args()


def run_command(command):
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def read_metrics(path):
    with open(path, "r", encoding="utf8", newline="") as metrics_file:
        return list(csv.DictReader(metrics_file))


def metric_mean(rows, field):
    return mean(float(row[field]) for row in rows)


def optional_metric_mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return mean(values) if values else ""


def positive_count(rows, field):
    return sum(float(row[field]) > 0 for row in rows if row.get(field, "") != "")


def write_summary(path, rows):
    field_names = [
        "Scope",
        "QP",
        "Runs",
        "Mean PSNR Predicted",
        "Mean PSNR Base",
        "Mean PSNR Gain",
        "Mean Y PSNR Predicted",
        "Mean Y PSNR Base",
        "Mean Y PSNR Gain",
        "Mean SSIM Predicted",
        "Mean SSIM Base",
        "Mean SSIM Gain",
        "Positive PSNR Runs",
        "Positive Y PSNR Runs",
        "Positive SSIM Runs",
    ]

    groups = [("ALL", "ALL", rows)]
    for qp in sorted({int(row["QP"]) for row in rows}):
        groups.append((f"QP-{qp}", qp, [row for row in rows if int(row["QP"]) == qp]))

    summary_rows = []
    for scope, qp, group_rows in groups:
        summary_rows.append(
            [
                scope,
                qp,
                len(group_rows),
                metric_mean(group_rows, "PSNR Predicted"),
                metric_mean(group_rows, "PSNR Base"),
                metric_mean(group_rows, "PSNR Gain"),
                optional_metric_mean(group_rows, "Y PSNR Predicted"),
                optional_metric_mean(group_rows, "Y PSNR Base"),
                optional_metric_mean(group_rows, "Y PSNR Gain"),
                metric_mean(group_rows, "SSIM Predicted"),
                metric_mean(group_rows, "SSIM Base"),
                metric_mean(group_rows, "SSIM Gain"),
                positive_count(group_rows, "PSNR Gain"),
                positive_count(group_rows, "Y PSNR Gain"),
                positive_count(group_rows, "SSIM Gain"),
            ]
        )

    with open(path, "w", encoding="utf8", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(field_names)
        writer.writerows(summary_rows)
    return summary_rows


def write_failures(path, failures):
    with open(path, "w", encoding="utf8", newline="") as failure_file:
        writer = csv.writer(failure_file)
        writer.writerow(["Video", "Sequence", "QP", "Error"])
        writer.writerows(failures)


def main():
    args = parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3.")
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and no greater than 1.")
    if args.metric_mode == "full-frame" and args.fraction != 1:
        raise ValueError("Full-frame metrics require --fraction 1 for complete coverage.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_run is not None:
        run_dir = os.path.abspath(args.resume_run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        protocol_name = f"resume_{timestamp}.txt"
    else:
        run_dir = os.path.join(args.run_root, f"ldv_{timestamp}")
        os.makedirs(run_dir, exist_ok=False)
        protocol_name = "protocol.txt"

    metrics_path = os.path.join(run_dir, "metrics.csv")
    summary_path = os.path.join(run_dir, "summary.csv")
    failures_path = os.path.join(run_dir, "failures.csv")
    with open(os.path.join(run_dir, protocol_name), "w", encoding="utf8") as protocol_file:
        for key, value in sorted(vars(args).items()):
            protocol_file.write(f"{key}={value}\n")

    existing_rows = read_metrics(metrics_path) if os.path.isfile(metrics_path) else []
    completed_pairs = {
        (int(row["Sequence"]), int(row["QP"])) for row in existing_rows
    }

    prepare_script = os.path.join(BASE_DIR, "prepare_ldv.py")
    test_script = os.path.join(BASE_DIR, "test.py")
    failures = []
    planned_pairs = [
        (video_id, args.start_sequence + video_offset, qp)
        for video_offset, video_id in enumerate(args.video_ids)
        for qp in args.qps
    ]
    remaining_runs = sum(
        (sequence, qp) not in completed_pairs
        for _, sequence, qp in planned_pairs
    )

    print(f"Run directory  : {run_dir}")
    print(f"Videos         : {args.video_ids}")
    print(f"QP values      : {args.qps}")
    print(f"Completed runs : {len(completed_pairs)}")
    print(f"Remaining runs : {remaining_runs}")

    run_number = 0
    for video_offset, video_id in enumerate(args.video_ids):
        sequence = args.start_sequence + video_offset
        input_video = os.path.join(args.ldv_dir, f"{video_id:03d}.mkv")
        if not os.path.isfile(input_video):
            error = f"Input video does not exist: {input_video}"
            for qp in args.qps:
                if (sequence, qp) not in completed_pairs:
                    failures.append([video_id, sequence, qp, error])
            if args.stop_on_error:
                raise FileNotFoundError(error)
            print(error)
            continue

        for qp in args.qps:
            if (sequence, qp) in completed_pairs:
                print(
                    f"Skipping completed run: video {video_id:03d}, "
                    f"sequence {sequence:03d}, QP {qp}"
                )
                continue

            run_number += 1
            print(
                f"\n=== Remaining run {run_number}/{remaining_runs}: video {video_id:03d}, "
                f"sequence {sequence:03d}, QP {qp} ==="
            )
            result_dir = os.path.join(
                run_dir, "visuals", f"{sequence:03d}", f"QP-{qp}"
            )
            prepare_command = [
                sys.executable,
                prepare_script,
                "--input",
                input_video,
                "--output-root",
                args.output_root,
                "--sequence",
                str(sequence),
                "--qp",
                str(qp),
                "--frames",
                str(args.frames),
                "--encoder",
                args.encoder,
                "--overwrite",
            ]
            test_command = [
                sys.executable,
                test_script,
                "--raw-path",
                os.path.join(args.output_root, "Raw"),
                "--encoded-path",
                os.path.join(args.output_root, "Encoded"),
                "--sequence",
                str(sequence),
                "--qp",
                str(qp),
                "--fraction",
                str(args.fraction),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--save-limit",
                str(args.save_limit),
                "--results-dir",
                result_dir,
                "--report",
                metrics_path,
            ]
            if args.metric_mode == "full-frame":
                test_command.append("--full-frame-metrics")

            try:
                run_command(prepare_command)
                run_command(test_command)
            except subprocess.CalledProcessError as error:
                failures.append([video_id, sequence, qp, str(error)])
                print(f"Run failed: {error}")
                if args.stop_on_error:
                    write_failures(failures_path, failures)
                    raise

    write_failures(failures_path, failures)
    if not os.path.isfile(metrics_path):
        raise RuntimeError(
            f"No evaluation completed successfully. Failures: {failures_path}"
        )

    rows = read_metrics(metrics_path)
    summary_rows = write_summary(summary_path, rows)

    print("\nBatch evaluation finished")
    for row in summary_rows:
        y_gain = f"{row[8]:+.4f} dB" if row[8] != "" else "n/a"
        print(
            f"{row[0]:>5}: runs={row[2]}, RGB PSNR gain={row[5]:+.4f} dB, "
            f"Y PSNR gain={y_gain}, SSIM gain={row[11]:+.6f}, "
            f"positive RGB={row[12]}/{row[2]}"
        )
    print(f"Metrics        : {metrics_path}")
    print(f"Summary        : {summary_path}")
    if failures:
        print(f"Failures       : {failures_path} ({len(failures)})")


if __name__ == "__main__":
    main()
