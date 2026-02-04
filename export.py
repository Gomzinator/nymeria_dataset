#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Nymeria dataset IMU sensor poses and biases to CSV files.

For each device (head, observer, lwrist, rwrist) and each IMU sensor
(imu-left, imu-right), exports:
  - T_world_sensor at every native IMU timestamp
  - Gyroscope and accelerometer biases (online or factory calibration)

All timestamps in the output CSV are in TIME_CODE domain (shared across
all devices) so that rows from different recordings are directly comparable.

Output CSV columns:
  timestamp_ns, x, y, z, qx, qy, qz, qw,
  bias_gyro_x, bias_gyro_y, bias_gyro_z,
  bias_accel_x, bias_accel_y, bias_accel_z
"""

import csv
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger
from tqdm import tqdm


DEVICE_TAGS = ["head", "observer", "lwrist", "rwrist"]
SENSOR_LABELS = ["imu-left", "imu-right"]

CSV_HEADER = [
    "timestamp_ns",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "bias_gyro_x",
    "bias_gyro_y",
    "bias_gyro_z",
    "bias_accel_x",
    "bias_accel_y",
    "bias_accel_z",
]


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def load_report(report_path: Path) -> dict:
    """Load existing JSON report or create empty one."""
    if report_path.is_file():
        with open(report_path) as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(), "sequences": {}}


def save_report(report_path: Path, report: dict) -> None:
    """Save JSON report (atomic-ish write via tmp rename)."""
    report["last_updated"] = datetime.now().isoformat()
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    tmp.rename(report_path)


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


def discover_sequences(batch_path: Path) -> list[Path]:
    """
    Discover sequence directories from a batch path.

    If batch_path is a .txt file: one sequence directory path per line.
    If batch_path is a directory: each subdirectory is a sequence.
    """
    if batch_path.is_file() and batch_path.suffix == ".txt":
        sequences = []
        with open(batch_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line)
                if p.is_dir():
                    sequences.append(p)
                else:
                    logger.warning(f"Skipping non-existent path: {line}")
        return sequences

    if batch_path.is_dir():
        sequences = sorted(
            [d for d in batch_path.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        return sequences

    raise click.BadParameter(
        f"{batch_path} is neither a directory nor a .txt file"
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_sequence(
    sequence_dir: Path,
    output_dir: Path,
    use_online_calib: bool,
) -> dict:
    """
    Process a single Nymeria sequence.

    Returns a report dict:
        {
            "status": "ok" | "error",
            "files": [...],
            "warnings": [...],
            "error": str | None,
        }
    """
    from nymeria.data_provider import NymeriaDataProvider
    from projectaria_tools.core.sensor_data import TimeDomain

    seq_name = sequence_dir.name
    seq_output = output_dir / seq_name
    seq_output.mkdir(parents=True, exist_ok=True)

    entry: dict = {"status": "ok", "files": [], "warnings": []}

    try:
        ndp = NymeriaDataProvider(
            sequence_rootdir=sequence_dir,
            load_head=True,
            load_observer=True,
            load_wrist=True,
            load_body=False,
            load_online_calib=use_online_calib,
        )
    except Exception as e:
        entry["status"] = "error"
        entry["error"] = f"Failed to create NymeriaDataProvider: {e}"
        return entry

    for device_tag in DEVICE_TAGS:
        try:
            rec = ndp.get_recording(device_tag)
        except ValueError:
            msg = f"{device_tag}: recording not available, skipping"
            logger.warning(msg)
            entry["warnings"].append(msg)
            continue

        if not rec.has_vrs:
            msg = f"{device_tag}: no VRS file, skipping"
            logger.warning(msg)
            entry["warnings"].append(msg)
            continue

        if not rec.has_pose:
            msg = f"{device_tag}: no closed-loop trajectory, skipping"
            logger.warning(msg)
            entry["warnings"].append(msg)
            continue

        for sensor_label in SENSOR_LABELS:
            csv_name = f"{device_tag}_{sensor_label}.csv"
            csv_path = seq_output / csv_name

            try:
                n = _export_sensor_csv(
                    ndp=ndp,
                    rec=rec,
                    device_tag=device_tag,
                    sensor_label=sensor_label,
                    csv_path=csv_path,
                    use_online_calib=use_online_calib,
                )
                entry["files"].append({"name": csv_name, "samples": n})
            except Exception as e:
                msg = f"{device_tag}/{sensor_label}: {e}"
                logger.error(msg)
                entry["warnings"].append(msg)

    return entry


def _export_sensor_csv(
    ndp,
    rec,
    device_tag: str,
    sensor_label: str,
    csv_path: Path,
    use_online_calib: bool,
) -> int:
    """
    Export one CSV for one device/sensor pair.

    Iteration strategy:
      - Iterate over native IMU timestamps from VRS (DEVICE_TIME domain)
      - Query pose and calibration in DEVICE_TIME (consistent within recording)
      - Convert timestamp to TIME_CODE for the CSV (comparable across recordings)

    Returns the number of samples written.
    """
    from projectaria_tools.core.sensor_data import TimeDomain

    # Native IMU timestamps in DEVICE_TIME
    timestamps_device_ns = rec.get_sensor_timestamps_device_time_ns(sensor_label)

    logger.info(
        f"  {device_tag}/{sensor_label}: "
        f"{len(timestamps_device_ns)} samples -> {csv_path.name}"
    )

    # For factory calibration: fetch once (static, not time-varying)
    static_calib = None
    if not use_online_calib:
        static_calib = ndp.get_sensor_calibration(
            device_tag, sensor_label, t_ns=None
        )

    vrs_dp = rec.vrs_dp
    n_written = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for t_device_ns in timestamps_device_ns:
            t_device_ns = int(t_device_ns)

            # -- Pose: T_world_sensor via DEVICE_TIME --
            T_world_sensor, _ = ndp.get_sensor_pose(
                t_device_ns,
                device_tag,
                sensor_label,
                time_domain=TimeDomain.DEVICE_TIME,
            )

            # -- Calibration (biases) via DEVICE_TIME --
            if use_online_calib:
                calib = ndp.get_sensor_calibration(
                    device_tag,
                    sensor_label,
                    t_ns=t_device_ns,
                    time_domain=TimeDomain.DEVICE_TIME,
                )
            else:
                calib = static_calib

            gyro_bias = np.array(calib.get_gyro_model().get_bias())
            accel_bias = np.array(calib.get_accel_model().get_bias())

            # -- Convert timestamp to TIME_CODE for CSV --
            t_timecode_ns = vrs_dp.convert_from_device_time_to_timecode_ns(
                t_device_ns
            )

            # -- Extract pose components --
            xyz = T_world_sensor.translation()
            # to_quat_and_translation() returns (quat_xyzw, translation)
            quat_xyzw, _ = T_world_sensor.to_quat_and_translation()

            writer.writerow([
                t_timecode_ns,
                f"{xyz[0]:.8f}",
                f"{xyz[1]:.8f}",
                f"{xyz[2]:.8f}",
                f"{quat_xyzw[0]:.8f}",
                f"{quat_xyzw[1]:.8f}",
                f"{quat_xyzw[2]:.8f}",
                f"{quat_xyzw[3]:.8f}",
                f"{gyro_bias[0]:.10f}",
                f"{gyro_bias[1]:.10f}",
                f"{gyro_bias[2]:.10f}",
                f"{accel_bias[0]:.10f}",
                f"{accel_bias[1]:.10f}",
                f"{accel_bias[2]:.10f}",
            ])
            n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "-i",
    "sequence_dir",
    type=Path,
    default=None,
    help="Single sequence directory to process.",
)
@click.option(
    "-b",
    "batch_path",
    type=Path,
    default=None,
    help="Batch mode: directory of sequences OR .txt file with one path per line.",
)
@click.option(
    "-o",
    "output_dir",
    type=Path,
    required=True,
    help="Output directory (created if needed).",
)
@click.option(
    "-j",
    "num_workers",
    type=int,
    default=0,
    help="Number of parallel workers for batch mode (0 or 1 = sequential).",
)
@click.option(
    "--use-online-calib",
    is_flag=True,
    default=False,
    help="Use online (time-varying) calibration instead of factory calibration.",
)
@click.option(
    "--stop-on-error",
    is_flag=True,
    default=False,
    help="Stop processing on first error (in sequential mode stops immediately; "
    "in parallel mode waits for running tasks then stops).",
)
def main(
    sequence_dir: Path | None,
    batch_path: Path | None,
    output_dir: Path,
    num_workers: int,
    use_online_calib: bool,
    stop_on_error: bool,
) -> None:
    """Export Nymeria IMU sensor poses and biases to CSV."""

    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<level>{level: <7}</level> "
            "<blue>{name}.py:</blue>"
            "<green>{function}</green>"
            "<yellow>:{line}</yellow> {message}"
        ),
        level="INFO",
    )

    # Validate mutual exclusivity
    if sequence_dir is not None and batch_path is not None:
        raise click.UsageError("Options -i and -b are mutually exclusive.")
    if sequence_dir is None and batch_path is None:
        raise click.UsageError("One of -i or -b is required.")

    # Build sequence list
    if sequence_dir is not None:
        if not sequence_dir.is_dir():
            raise click.BadParameter(
                f"{sequence_dir} is not a directory", param_hint="-i"
            )
        sequences = [sequence_dir]
    else:
        sequences = discover_sequences(batch_path)
        if not sequences:
            logger.error("No sequences found.")
            sys.exit(1)
        logger.info(f"Discovered {len(sequences)} sequences")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report = load_report(report_path)

    use_parallel = len(sequences) > 1 and num_workers > 1

    if use_parallel:
        _run_parallel(
            sequences,
            output_dir,
            use_online_calib,
            stop_on_error,
            num_workers,
            report,
            report_path,
        )
    else:
        _run_sequential(
            sequences,
            output_dir,
            use_online_calib,
            stop_on_error,
            report,
            report_path,
        )

    # Final save
    save_report(report_path, report)
    logger.info(f"Report saved to {report_path}")


def _run_sequential(
    sequences: list[Path],
    output_dir: Path,
    use_online_calib: bool,
    stop_on_error: bool,
    report: dict,
    report_path: Path,
) -> None:
    """Process sequences one by one with a tqdm progress bar."""
    for seq_dir in tqdm(sequences, desc="Sequences", unit="seq"):
        seq_name = seq_dir.name
        logger.info(f"Processing: {seq_name}")

        try:
            entry = process_sequence(seq_dir, output_dir, use_online_calib)
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"Unhandled error in {seq_name}:\n{tb}")
            entry = {
                "status": "error",
                "error": tb,
                "files": [],
                "warnings": [],
            }

        report["sequences"][seq_name] = entry
        save_report(report_path, report)

        if entry["status"] == "error":
            logger.error(
                f"FAIL {seq_name}: {entry.get('error', 'unknown error')}"
            )
            if stop_on_error:
                logger.error("--stop-on-error: aborting.")
                sys.exit(1)
        else:
            n_files = len(entry["files"])
            n_warn = len(entry["warnings"])
            logger.info(f"OK   {seq_name}: {n_files} files, {n_warn} warnings")


def _run_parallel(
    sequences: list[Path],
    output_dir: Path,
    use_online_calib: bool,
    stop_on_error: bool,
    num_workers: int,
    report: dict,
    report_path: Path,
) -> None:
    """Process sequences in parallel with a tqdm progress bar."""
    should_stop = False

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_seq = {
            executor.submit(
                process_sequence, seq_dir, output_dir, use_online_calib
            ): seq_dir
            for seq_dir in sequences
        }

        with tqdm(total=len(sequences), desc="Sequences", unit="seq") as pbar:
            for future in as_completed(future_to_seq):
                seq_dir = future_to_seq[future]
                seq_name = seq_dir.name

                try:
                    entry = future.result()
                except Exception:
                    tb = traceback.format_exc()
                    logger.error(
                        f"Unhandled error in {seq_name}:\n{tb}"
                    )
                    entry = {
                        "status": "error",
                        "error": tb,
                        "files": [],
                        "warnings": [],
                    }

                report["sequences"][seq_name] = entry
                save_report(report_path, report)
                pbar.update(1)

                if entry["status"] == "error":
                    logger.error(
                        f"FAIL {seq_name}: "
                        f"{entry.get('error', 'unknown error')}"
                    )
                    if stop_on_error:
                        should_stop = True
                        executor.shutdown(wait=True, cancel_futures=True)
                        break
                else:
                    n_files = len(entry["files"])
                    n_warn = len(entry["warnings"])
                    logger.info(
                        f"OK   {seq_name}: "
                        f"{n_files} files, {n_warn} warnings"
                    )

    if should_stop:
        logger.error(
            "--stop-on-error: aborting after current tasks finished."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
