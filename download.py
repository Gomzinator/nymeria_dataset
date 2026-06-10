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


def get_groups(
    video: bool = False,
    body: bool = False,
    narration: bool = False,
    semidense: bool = False,
) -> list[DataGroups]:
    """
    Build the list of data groups to download.

    Core (always downloaded) — the minimal IMU/SLAM set:
      - LICENSE, metadata.json
      - recording_head / recording_observer  (motion.vrs + SLAM + gaze; et.vrs
        stripped via _EXCLUDE_PATTERNS)
      - recording_lwrist / recording_rwrist  (motion.vrs + SLAM)

    Everything else is opt-in via a flag (each flag just appends its groups):
      - video=True  (--video)      → data.vrs full-sensor recording (with video)
                                      for head/observer; motion.vrs is then
                                      pruned/stripped (see _VIDEO_EXCLUDE_PATTERNS).
      - body=True   (--body)       → the whole body/ dir: xdata.npz +
                                      xdata_blueman.glb (body_motion) and the raw
                                      xdata.mvnx (body_xdata_mvnx).
      - narration=True (--narration) → the three narration CSVs.
      - semidense=True (--semi-dense) → semidense point cloud: keeps the
                                      semidense_points.csv.gz bundled in each
                                      recording zip AND downloads the standalone
                                      semidense_observations.csv.gz (large).

    Anything not selected here is removed from disk by --prune.
    """
    groups = [
        DataGroups.LICENSE,
        DataGroups.metadata_json,
        DataGroups.recording_head,       # motion.vrs + SLAM + gaze (et.vrs stripped)
        DataGroups.recording_lwrist,
        DataGroups.recording_rwrist,
        DataGroups.recording_observer,   # motion.vrs + SLAM + gaze (et.vrs stripped)
    ]
    if video:
        # data.vrs is the full-sensor recording including all video streams;
        # it is a superset of motion.vrs (which is stripped of video).
        groups += [
            DataGroups.recording_head_data_data_vrs,
            DataGroups.recording_observer_data_data_vrs,
        ]
    if body:
        groups += [
            DataGroups.body_motion,      # xdata.npz + xdata_blueman.glb
            DataGroups.body_xdata_mvnx,  # raw xdata.mvnx
        ]
    if narration:
        groups += [
            DataGroups.narration_motion_narration_csv,
            DataGroups.narration_atomic_action_csv,
            DataGroups.narration_activity_summarization_csv,
        ]
    if semidense:
        # Selecting this group both keeps the (already-bundled) semidense_points
        # and downloads the standalone semidense_observations point cloud.
        groups += [DataGroups.semidense_observations]
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
        "Also download data.vrs (includes video streams) for head/observer. "
        "Significantly larger than motion-only; omit for IMU/SLAM-only workflows."
    ),
)
@click.option(
    "--body",
    "include_body",
    is_flag=True,
    help="Also download the body/ data (xdata.npz, xdata_blueman.glb, xdata.mvnx).",
)
@click.option(
    "--narration",
    "include_narration",
    is_flag=True,
    help="Also download the narration CSVs (motion / atomic action / activity summary).",
)
@click.option(
    "--semi-dense",
    "include_semidense",
    is_flag=True,
    help=(
        "Also keep/download the semi-dense point cloud: semidense_points.csv.gz "
        "(bundled in each recording zip) and the standalone "
        "semidense_observations.csv.gz (large). Without this flag they are pruned."
    ),
)
@click.option(
    "--all",
    "include_all",
    is_flag=True,
    help="Download everything: equivalent to --video --body --narration --semi-dense.",
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
    overwrite: bool,
    match_key: str,
    include_video: bool,
    include_body: bool,
    include_narration: bool,
    include_semidense: bool,
    include_all: bool,
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

    # --all is shorthand for turning on every opt-in group.
    if include_all:
        include_video = include_body = include_narration = include_semidense = True

    dl = DownloadManager(url_json, out_rootdir=rootdir)
    groups = get_groups(
        video=include_video,
        body=include_body,
        narration=include_narration,
        semidense=include_semidense,
    )

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
