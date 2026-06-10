# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-sequence capability manifest.

Two views are written to ``<out_rootdir>/manifest.json`` (write-only — this file
is never read back; both dicts are rebuilt from scratch on every run):

  has[seq][info]            -> bool
      Ground truth from the input master url json: does the sequence have a
      download link for that info's data group? Rebuilt in-memory from the json,
      independent of what we choose to download or of any prior manifest.

  was_downloaded[seq][info] -> bool
      Reality on disk: does the representative file for that info exist under
      <out_rootdir>/<seq>? Found by scanning the actual downloaded files (not the
      manifest), so it is robust to --prune and the motion.vrs -> data.vrs swap.

Only data-group-derivable infos are tracked (see refinement notes); metadata-only
explorer filters such as has_two_participants / *_slam / *_gaze / timesync are out
of scope because they cannot be recovered from the master url json alone.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from .definitions import (
    BodyFiles,
    DataGroups,
    SlamFiles,
    Subpaths,
    TextFiles,
    VrsFiles,
)

MANIFEST_FILENAME = "manifest.json"

_RECORDINGS = (
    Subpaths.recording_head,
    Subpaths.recording_lwrist,
    Subpaths.recording_rwrist,
    Subpaths.recording_observer,
)


@dataclass(frozen=True)
class InfoDef:
    """One tracked capability column."""

    name: str
    # DataGroups.name to look for among a sequence's url keys -> drives `has`.
    has_key: str
    # Representative relative path(s); ANY existing on disk -> was_downloaded True.
    files: tuple[str, ...]


# Order here is the column order in the manifest.
INFOS: tuple[InfoDef, ...] = (
    InfoDef(
        "head",
        DataGroups.recording_head.name,
        (
            f"{Subpaths.recording_head}/{VrsFiles.motion}",
            f"{Subpaths.recording_head}/{VrsFiles.data}",
        ),
    ),
    InfoDef(
        "left_wrist",
        DataGroups.recording_lwrist.name,
        (f"{Subpaths.recording_lwrist}/{VrsFiles.motion}",),
    ),
    InfoDef(
        "right_wrist",
        DataGroups.recording_rwrist.name,
        (f"{Subpaths.recording_rwrist}/{VrsFiles.motion}",),
    ),
    InfoDef(
        "observer",
        DataGroups.recording_observer.name,
        (
            f"{Subpaths.recording_observer}/{VrsFiles.motion}",
            f"{Subpaths.recording_observer}/{VrsFiles.data}",
        ),
    ),
    InfoDef(
        "body_motion",
        DataGroups.body_motion.name,
        (BodyFiles.xsens_processed,),  # body/xdata.npz
    ),
    InfoDef(
        "body_xdata_mvnx",
        DataGroups.body_xdata_mvnx.name,
        (BodyFiles.xsens_raw,),  # body/xdata.mvnx
    ),
    InfoDef(
        "video",
        DataGroups.recording_head_data_data_vrs.name,
        (f"{Subpaths.recording_head}/{VrsFiles.data}",),
    ),
    InfoDef(
        "semidense",
        DataGroups.semidense_observations.name,
        tuple(f"{rec}/{SlamFiles.semidense_observations}" for rec in _RECORDINGS),
    ),
    InfoDef(
        "atomic_action",
        DataGroups.narration_atomic_action_csv.name,
        (TextFiles.atomic_action,),
    ),
    InfoDef(
        "motion_narration",
        DataGroups.narration_motion_narration_csv.name,
        (TextFiles.motion_narration,),
    ),
    InfoDef(
        "activity_summarization",
        DataGroups.narration_activity_summarization_csv.name,
        (TextFiles.activity_summarization,),
    ),
)


def build_has(sequences: dict) -> dict[str, dict[str, bool]]:
    """has[seq][info] = the sequence has a download link for that info's group."""
    return {
        seq: {info.name: info.has_key in dgs for info in INFOS}
        for seq, dgs in sequences.items()
    }


def blank_was_downloaded(seq_names: Iterable[str]) -> dict[str, dict[str, bool]]:
    """All-False placeholder written before downloading (crash-safe manifest)."""
    return {seq: {info.name: False for info in INFOS} for seq in seq_names}


def scan_was_downloaded(
    out_rootdir: Path, seq_names: Iterable[str]
) -> dict[str, dict[str, bool]]:
    """was_downloaded[seq][info] = any representative file exists on disk."""
    result: dict[str, dict[str, bool]] = {}
    for seq in seq_names:
        seq_dir = out_rootdir / seq
        result[seq] = {
            info.name: any((seq_dir / rel).is_file() for rel in info.files)
            for info in INFOS
        }
    return result


def write_manifest(
    out_rootdir: Path,
    has: dict[str, dict[str, bool]],
    was_downloaded: dict[str, dict[str, bool]],
) -> Path:
    """Overwrite <out_rootdir>/manifest.json. Write-only; never read back."""
    path = out_rootdir / MANIFEST_FILENAME
    with open(path, "w") as f:
        json.dump({"has": has, "was_downloaded": was_downloaded}, f, indent=2)
    logger.info(f"wrote capability manifest to {path}")
    return path
