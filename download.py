# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import click
from loguru import logger
from nymeria.definitions import DataGroups, VrsFiles
from nymeria.download_utils import DownloadManager

# et.vrs (eye-tracking) is always excluded from extraction — it is bundled inside
# the recording_head / recording_observer zips but stripped out on unzip.
_EXCLUDE_PATTERNS = ["et.vrs"]

# When --video is set, data.vrs (full-sensor, with video) replaces the
# motion-only motion.vrs for the head and observer recordings. These two
# recordings have a *standalone* data.vrs download group, so dropping their
# motion.vrs is safe — the equivalent (richer) data is fetched separately.
#
# Scope is deliberately head/observer only: the wrist recordings have NO
# separate data.vrs download, so a bare "motion.vrs" pattern would prune the
# wrist motion.vrs (and, on a skip-existing re-run, nothing would replace it,
# leaving recording_*wrist/data empty). VrsFiles.motion == "data/motion.vrs".
_VIDEO_EXCLUDE_PATTERNS = [
    f"{DataGroups.recording_head.value}/{VrsFiles.motion}",
    f"{DataGroups.recording_observer.value}/{VrsFiles.motion}",
]


def get_groups(video: bool = False) -> list[DataGroups]:
    """
    Full download: body, all recordings, narrations, raw Xsens.

    Files included per recording_head / recording_observer group (zip):
      - motion.vrs  — IMU + audio, no video (always downloaded)
      - et.vrs      — eye tracking (always excluded via _EXCLUDE_PATTERNS)
      - mps/slam/   — SLAM trajectories & calibration
      - mps/eye_gaze/ — gaze CSVs

    Pass video=True (--video flag) to also fetch data.vrs, which is the
    full-sensor VRS with video streams on top of motion data.

    semidense_observations (3D point cloud) is excluded by default.
    """
    groups = [
        DataGroups.LICENSE,
        DataGroups.metadata_json,
        DataGroups.body_motion,
        DataGroups.recording_head,       # motion.vrs + SLAM + gaze (et.vrs stripped)
        DataGroups.recording_lwrist,
        DataGroups.recording_rwrist,
        DataGroups.recording_observer,   # motion.vrs + SLAM + gaze (et.vrs stripped)
        DataGroups.narration_motion_narration_csv,
        DataGroups.narration_atomic_action_csv,
        DataGroups.narration_activity_summarization_csv,
        # DataGroups.semidense_observations,  # 3D point cloud — large, opt-in only
        DataGroups.body_xdata_mvnx,
    ]
    if video:
        # data.vrs is the full-sensor recording including all video streams;
        # it is a superset of motion.vrs (which is stripped of video).
        groups += [
            DataGroups.recording_head_data_data_vrs,
            DataGroups.recording_observer_data_data_vrs,
        ]
    return groups


def get_groups_IMU(video: bool = False) -> list[DataGroups]:
    """
    Minimal download for VIO post-processing: recordings + body motion only,
    no narrations.  Same et.vrs / semidense exclusion rules as get_groups().
    """
    groups = [
        DataGroups.LICENSE,
        DataGroups.metadata_json,
        DataGroups.body_motion,
        DataGroups.recording_head,
        DataGroups.recording_lwrist,
        DataGroups.recording_rwrist,
        DataGroups.recording_observer,
        DataGroups.body_xdata_mvnx,
    ]
    if video:
        groups += [
            DataGroups.recording_head_data_data_vrs,
            DataGroups.recording_observer_data_data_vrs,
        ]
    return groups


@click.command()
@click.option(
    "-i",
    "url_json",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    required=True,
    help="The json file contains download urls. Follow README.md instructions to access this file.",
)
@click.option(
    "-o",
    "rootdir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=None,
    help="The root directory to hold the downloaded dataset",
)
@click.option(
    "-m",
    "minimal",
    is_flag=True,
    help="Download only the minimum required groups for VIO post processing",
)
@click.option(
    "-f",
    "overwrite",
    is_flag=True,
    help="Ignore existing files and redownload",
)
@click.option(
    "-k",
    "match_key",
    default="2023",
    help=(
        "Partial key used to filter sequences for downloading. "
        "Default key value = 2023, which includes all available sequences."
    ),
)
@click.option(
    "--video",
    "include_video",
    is_flag=True,
    help=(
        "Also download data.vrs (includes video streams). "
        "Significantly larger than motion-only; omit for IMU/SLAM-only workflows."
    ),
)
@click.option(
    "--prune",
    "prune",
    is_flag=True,
    help=(
        "Before downloading, delete any files on disk that don't belong to the "
        "currently selected groups (frees space before the download starts). "
        "Shown in the confirmation prompt — a single [y/n] covers both prune and download."
    ),
)
@click.option(
    "--max-retry",
    "max_retries",
    default=3,
    show_default=True,
    type=int,
    help="Number of times to retry a stalled download before giving up.",
)
@click.option(
    "--stall-timeout",
    "stall_timeout",
    default=30,
    show_default=True,
    type=int,
    help=(
        "Seconds window used to detect a stall: if less than 1 MB is received "
        "within this window the download is aborted and retried."
    ),
)
def main(
    url_json: Path,
    rootdir: Path,
    minimal: bool,
    overwrite: bool,
    match_key: str,
    include_video: bool,
    prune: bool,
    max_retries: int,
    stall_timeout: int,
) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<level>{level: <7}</level> <blue>{name}.py:</blue><green>{function}</green><yellow>:{line}</yellow> {message}",
        level="INFO",
    )

    dl = DownloadManager(url_json, out_rootdir=rootdir)
    groups = get_groups_IMU(video=include_video) if minimal else get_groups(video=include_video)

    exclude_patterns = list(_EXCLUDE_PATTERNS)
    if include_video:
        # Drop head/observer motion.vrs; data.vrs supersedes it. With --prune the
        # already-on-disk motion.vrs is deleted; on a fresh unzip it is stripped.
        exclude_patterns += _VIDEO_EXCLUDE_PATTERNS

    dl.download(
        match_key=match_key,
        selected_groups=groups,
        ignore_existing=not overwrite,
        exclude_patterns=exclude_patterns,
        prune=prune,
        stall_timeout=stall_timeout,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    main()
