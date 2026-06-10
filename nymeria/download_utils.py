# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from zipfile import is_zipfile, ZipFile

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from .definitions import DataGroups, NYMERIA_VERSION
from .manifest import (
    blank_was_downloaded,
    build_has,
    scan_was_downloaded,
    write_manifest,
)


class DlConfig(Enum):
    CHUCK_SIZE_BYTE = 8192
    READ_BYTE = 4096
    RETRY = 5
    BACKOFF_FACTOR = 3
    CONNECT_TIMEOUT = 30          # seconds to establish connection
    STALL_MIN_BYTES = 1 << 20     # 1 MB — minimum progress per stall window


class DlStatus(Enum):
    UNKNOWN = None
    SUCCESS = "success"
    IGNORED = "ignored, file already downloaded"
    WARN_NOTFOUND = "warning, missing download link"
    ERR_SHA1SUM = "error, sha1sum mismatch"
    ERR_MEMORY = "error, insufficient disk space"
    ERR_NETWORK = "error, network"
    ERR_STALL = "error, download stalled — max retries exceeded"


class StallError(Exception):
    """Raised when download progress falls below the minimum threshold."""


@dataclass
class DlLink:
    filename: str
    sha1sum: str
    file_size_bytes: int
    download_url: str

    data_group: DataGroups
    status: DlStatus = DlStatus.UNKNOWN

    def __post_init__(self) -> None:
        prefix = f"Nymeria_{NYMERIA_VERSION}_"
        if prefix not in self.filename:
            self.status = (
                f"Version mismatch with the release {NYMERIA_VERSION}. "
                f"Please download the latest url json"
            )
            raise ValueError(self.status)
        self.filename = self.filename.replace(prefix, "")

    @property
    def seq_name(self) -> str:
        return "_".join(self.filename.split("_")[0:6])

    @property
    def logdir(self) -> str:
        return "logs"

    def __check_outdir(self, outdir: Path) -> None:
        assert outdir.name == self.seq_name, (
            f"Output directory name ({outdir.name}) mismatch with sequence {self.seq_name}"
        )
        outdir.mkdir(exist_ok=True)

    def get(
        self,
        outdir: Path,
        ignore_existing: bool = True,
        exclude_patterns: list[str] | None = None,
        stall_timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """
        Download with stall detection and retry.

        A stall is triggered when less than 1 MB is received within
        `stall_timeout` seconds (either a completely frozen connection or a
        slow drip).  On stall the attempt is aborted and retried up to
        `max_retries` times total.
        """
        flag = outdir / self.logdir / self.data_group.name
        if flag.is_file() and ignore_existing:
            self.status = DlStatus.IGNORED
            return

        self.__check_outdir(outdir)

        for attempt in range(1, max_retries + 1):
            try:
                self._attempt(outdir, exclude_patterns, stall_timeout)
                break  # success — exit retry loop
            except StallError as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Stall on {self.data_group.name} "
                        f"(attempt {attempt}/{max_retries}): {e} — retrying..."
                    )
                else:
                    # All attempts exhausted. Raise so the caller can log it;
                    # the except Exception in DownloadManager.download() catches
                    # this and continues with the next file.
                    # No flag is written → this file will be retried on the next run.
                    self.status = DlStatus.ERR_STALL
                    raise RuntimeError(
                        f"{self.data_group.name}: stalled after {max_retries} attempts — {e}"
                    ) from e

        logger.info(f"Downloaded {self.filename} → {outdir}")
        self.status = DlStatus.SUCCESS
        # Touch the "done" marker so subsequent runs skip this file.
        # flag = outdir/logs/<group_name>; the logs/ dir may not exist yet.
        flag.parent.mkdir(exist_ok=True)
        flag.touch()

    def _attempt(
        self,
        outdir: Path,
        exclude_patterns: list[str] | None,
        stall_timeout: int,
    ) -> None:
        """
        Single download attempt.  Raises StallError on stall; raises
        RuntimeError on unrecoverable errors (sha1, memory, HTTP).
        """
        session = requests.Session()
        # Retry will be triggered for the following HTTP codes:
        # 429 Too Many Requests, 500/502/503/504 server errors
        retries = Retry(
            total=DlConfig.RETRY.value,
            backoff_factor=DlConfig.BACKOFF_FACTOR.value,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_filename = Path(tmpdir) / self.filename
            logger.info(f"Download {self.filename} → {tmp_filename}")

            try:
                r_ctx = session.get(
                    self.download_url,
                    stream=True,
                    # read_timeout fires if no bytes arrive within stall_timeout seconds,
                    # covering the "completely frozen connection" case.
                    timeout=(DlConfig.CONNECT_TIMEOUT.value, stall_timeout),
                )
            except requests.exceptions.ConnectionError as e:
                self.status = DlStatus.ERR_NETWORK
                raise RuntimeError(f"Connection error: {e}") from e

            with r_ctx as r:
                free_outdir = shutil.disk_usage(outdir).free
                free_tmpdir = shutil.disk_usage(tmpdir).free
                if (
                    free_outdir < self.file_size_bytes
                    or free_tmpdir < self.file_size_bytes
                ):
                    self.status = DlStatus.ERR_MEMORY
                    raise RuntimeError(
                        "Insufficient disk space. "
                        f"Require {self.file_size_bytes}B, "
                        f"tmpdir available {free_tmpdir}B, outdir available {free_outdir}B"
                    )

                with open(tmp_filename, "wb") as f:
                    sha1 = hashlib.sha1()
                    progress_bar = tqdm(
                        total=self.file_size_bytes,
                        unit="iB",
                        unit_scale=True,
                        position=1,
                        leave=False,
                        desc=f"  {self.data_group.name}",
                        dynamic_ncols=True,
                    )

                    window_start = time.monotonic()
                    bytes_in_window = 0

                    try:
                        for chunk in r.iter_content(
                            chunk_size=DlConfig.CHUCK_SIZE_BYTE.value
                        ):
                            progress_bar.update(len(chunk))
                            f.write(chunk)
                            sha1.update(chunk)

                            # Rate check: slow-drip stall detection
                            bytes_in_window += len(chunk)
                            elapsed = time.monotonic() - window_start
                            if elapsed >= stall_timeout:
                                if bytes_in_window < DlConfig.STALL_MIN_BYTES.value:
                                    raise StallError(
                                        f"{bytes_in_window / 1024:.1f} KB in "
                                        f"{elapsed:.0f}s "
                                        f"(threshold: 1 MB / {stall_timeout}s)"
                                    )
                                bytes_in_window = 0
                                window_start = time.monotonic()

                    except requests.exceptions.ReadTimeout:
                        # Connection-level stall: no bytes at all for stall_timeout s
                        raise StallError(
                            f"no data received for {stall_timeout}s"
                        )
                    finally:
                        progress_bar.close()

                    computed = sha1.hexdigest()
                    if self.sha1sum != computed:
                        self.status = DlStatus.ERR_SHA1SUM
                        raise RuntimeError(
                            f"sha1sum mismatch, computed {computed}, expected {self.sha1sum}"
                        )

                try:
                    r.raise_for_status()
                except Exception as e:
                    self.status = DlStatus.ERR_NETWORK
                    raise RuntimeError(e) from e

            # move from tmp → dst
            if is_zipfile(tmp_filename):
                logger.info("unzip")
                with ZipFile(tmp_filename) as zf:
                    members = zf.namelist()
                    if exclude_patterns:
                        excluded = [
                            m for m in members
                            if any(pat in m for pat in exclude_patterns)
                        ]
                        members = [m for m in members if m not in excluded]
                        if excluded:
                            logger.info(
                                f"Skipping {len(excluded)} excluded file(s): {excluded}"
                            )
                    for member in members:
                        zf.extract(member, outdir)
            else:
                dst_file = outdir / self.data_group.value
                dst_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.move(src=tmp_filename, dst=dst_file)


class DownloadManager:
    def __init__(self, url_json: Path, out_rootdir: Path) -> None:
        self.url_json = url_json
        assert self.url_json.is_file(), f"{self.url_json} not found"

        self.out_rootdir = out_rootdir
        self.out_rootdir.mkdir(exist_ok=True)

        with open(self.url_json, "r") as f:
            data = json.load(f)
            self._sequences = data.get("sequences", {})
            assert len(self._sequences), (
                "No sequence found. Please check the json file is correct."
            )
        self.__get_data_summary()
        self._logs = {}

    @property
    def sequences(self) -> dict[str, any]:
        return self._sequences

    @property
    def logfile(self) -> Path:
        return self.out_rootdir / "download_summary.json"

    def __get_data_summary(self):
        missing = {x.name: {"count": 0, "sequences": []} for x in DataGroups}
        for seq, dgs in self.sequences.items():
            for dg in DataGroups:
                if dg.name not in dgs:
                    missing[dg.name]["count"] += 1
                    missing[dg.name]["sequences"].append(seq)
        fname = self.logfile.with_name("data_summary.json")
        with open(fname, "w") as f:
            json.dump(
                {
                    "missing_files": missing,
                    "available_sequences": list(self.sequences.keys()),
                },
                f,
                indent=2,
            )
        logger.info(f"save data summary to {fname}")

    def __prepare(
        self,
        match_key: str,
        selected_groups: list["DataGroups"],
        ignore_existing: bool = True,
        prune_targets: list[Path] | None = None,
        prune_freed_gb: float = 0.0,
    ) -> set["DataGroups"]:
        selected_groups += [DataGroups.LICENSE, DataGroups.metadata_json]
        selected_groups = set(selected_groups)

        num_seqs = 0
        num_files = 0        # files that will actually be downloaded
        num_skipped = 0      # files already on disk (flag present)
        total_gb = 0         # full project size (all selected files)
        pending_gb = 0       # size of files still to download
        self._logs = {}

        for seq, dgs in self.sequences.items():
            if match_key not in seq:
                continue

            num_seqs += 1
            self._logs[seq] = {}
            outdir = self.out_rootdir / seq
            for dg in selected_groups:
                if dg.name not in dgs:
                    self._logs[seq][dg.name] = DlStatus.WARN_NOTFOUND.value
                else:
                    self._logs[seq][dg.name] = None
                    dl = DlLink(**{**dgs.get(dg.name, {}), "data_group": dg})
                    file_gb = dl.file_size_bytes / (2**30)
                    total_gb += file_gb
                    flag = outdir / dl.logdir / dg.name
                    if flag.is_file() and ignore_existing:
                        num_skipped += 1
                    else:
                        pending_gb += file_gb
                        num_files += 1

        self._num_download_files = num_files + num_skipped  # total for outer bar

        # Space check
        free_disk_gb = shutil.disk_usage(self.out_rootdir).free / (2**30)
        effective_free_gb = free_disk_gb + prune_freed_gb
        has_space = effective_free_gb >= pending_gb

        # Build confirmation message
        msg = "\t" + "\n\t".join([x.value for x in selected_groups])
        skip_note = f" ({num_skipped} already downloaded, will be skipped)" if num_skipped else ""

        prune_lines = ""
        if prune_targets:
            prune_lines = (
                f"  Files to prune: {len(prune_targets)} ({prune_freed_gb:.2f} GB freed)\n"
                f"  Effective free space after pruning (GB): {effective_free_gb:.2f}\n"
            )

        space_warning = "" if has_space else (
            f"  ⚠  WARNING: effective free space ({effective_free_gb:.2f} GB) "
            f"< download size ({pending_gb:.2f} GB) — may run out of disk!\n"
        )

        confirm = (
            input(
                f"Download summary\n"
                f"  Output rootdir: {self.out_rootdir}\n"
                f"  Number of sequences: {num_seqs}\n"
                f"  Total size (GB): {total_gb:.2f}\n"
                f"  Files to download: {num_files}{skip_note}\n"
                f"  Total size to download (GB): {pending_gb:.2f}\n"
                f"{prune_lines}"
                f"  Available free disk space (GB): {free_disk_gb:.2f}\n"
                f"{space_warning}"
                f"  Selected data groups:\n{msg}\n"
                f"Proceed: [y/n] "
            ).lower()
            == "y"
        )
        if not confirm:
            exit(1)
        return selected_groups

    def _compute_prune_targets(
        self,
        match_key: str,
        selected_groups: set["DataGroups"],
        exclude_patterns: list[str] | None = None,
    ) -> tuple[list[Path], float]:
        """
        Dry-run: compute which files on disk don't belong to the selected groups.

        L_expected  = file paths that should exist for the given groups
        L_exists    = files currently on disk under out_rootdir
        L_delete    = L_exists - L_expected

        Returns (files_to_delete, freed_gb). Nothing is deleted.
        """
        from .definitions import get_group_definitions

        group_defs = get_group_definitions()

        # Build L_expected (posix paths relative to out_rootdir)
        expected: set[str] = {"download_summary.json", "data_summary.json"}
        for seq_name in self.sequences:
            if match_key not in seq_name:
                continue
            for dg in selected_groups:
                for rel in group_defs.get(dg.name, [dg.value]):
                    if exclude_patterns and any(pat in rel for pat in exclude_patterns):
                        continue
                    expected.add(f"{seq_name}/{rel}")
                # keep the "done" flag for this group
                expected.add(f"{seq_name}/logs/{dg.name}")

        # Build L_delete
        to_delete: list[Path] = [
            p for p in self.out_rootdir.rglob("*")
            if p.is_file()
            and p.relative_to(self.out_rootdir).as_posix() not in expected
        ]
        freed_gb = sum(p.stat().st_size for p in to_delete) / (2**30)
        return to_delete, freed_gb

    @staticmethod
    def _execute_prune(to_delete: list[Path], out_rootdir: Path) -> None:
        """Delete the given files and remove empty directories."""
        for p in to_delete:
            p.unlink()
        for dirpath in sorted(out_rootdir.rglob("*"), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                dirpath.rmdir()

    def prune(
        self,
        match_key: str,
        selected_groups: set["DataGroups"],
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """
        Standalone prune: list undesirable files, ask for confirmation, then delete.
        For integrated prune+download (prune first, single confirmation), use
        download(..., prune=True) instead.
        """
        to_delete, freed_gb = self._compute_prune_targets(
            match_key, selected_groups, exclude_patterns
        )

        if not to_delete:
            logger.info("Prune: nothing to remove.")
            return

        logger.warning(f"Prune: {len(to_delete)} files to delete ({freed_gb:.2f} GB will be freed)")
        for p in sorted(to_delete):
            logger.info(f"  - {p.relative_to(self.out_rootdir)}")

        confirm = input("\nConfirm deletion? [y/n] ").lower() == "y"
        if not confirm:
            logger.info("Prune cancelled.")
            return

        self._execute_prune(to_delete, self.out_rootdir)
        logger.info(f"Prune complete: {len(to_delete)} files deleted ({freed_gb:.2f} GB freed).")

    def __logging(self, **kwargs) -> None:
        self._logs.update(**kwargs)

        with open(self.logfile, "w") as f:
            json.dump(self._logs, f, indent=2)

    def download(
        self,
        match_key: str,
        selected_groups: list["DataGroups"],
        ignore_existing: bool = True,
        exclude_patterns: list[str] | None = None,
        prune: bool = False,
        stall_timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        # Compute prune targets upfront so __prepare can show the full picture
        # (freed space, effective free space, space check) in a single confirmation.
        prune_targets: list[Path] = []
        prune_freed_gb = 0.0
        if prune:
            # __prepare will add LICENSE + metadata_json; mirror that here so the
            # expected-file set matches exactly what __prepare will use.
            full_groups = set(list(selected_groups) + [DataGroups.LICENSE, DataGroups.metadata_json])
            prune_targets, prune_freed_gb = self._compute_prune_targets(
                match_key, full_groups, exclude_patterns
            )

        selected_groups = self.__prepare(
            match_key,
            selected_groups,
            ignore_existing=ignore_existing,
            prune_targets=prune_targets,
            prune_freed_gb=prune_freed_gb,
        )

        # Capability manifest (write-only; never read back). Build `has` from the
        # input json up front so it survives an interrupted download; the
        # `was_downloaded` view is a placeholder for now and gets the real
        # on-disk values at the end. Covers every sequence in the input json,
        # independent of match_key / selected_groups.
        seq_names = list(self.sequences.keys())
        manifest_has = build_has(self.sequences)
        write_manifest(
            self.out_rootdir, manifest_has, blank_was_downloaded(seq_names)
        )

        # Prune first to free space before the download starts
        if prune_targets:
            self._execute_prune(prune_targets, self.out_rootdir)
            logger.info(f"Pruned {len(prune_targets)} files ({prune_freed_gb:.2f} GB freed).")

        num_files = self._num_download_files
        outer_bar = tqdm(
            total=num_files,
            position=0,
            unit="file",
            desc="Overall",
            dynamic_ncols=True,
        )

        summary = {x.name: 0 for x in DlStatus}
        for seq_name, dgs in self.sequences.items():
            if match_key not in seq_name:
                continue

            outdir = self.out_rootdir / seq_name
            for dg in selected_groups:
                if dg.name not in dgs:
                    # Selected, but this sequence has no download link for it.
                    # Count + log it so missing data is visible in the summary
                    # instead of being silently skipped. (Not counted toward the
                    # outer progress bar, which only tracks downloadable files.)
                    summary[DlStatus.WARN_NOTFOUND.name] += 1
                    self._logs[seq_name][dg.name] = DlStatus.WARN_NOTFOUND.value
                    self.__logging()
                    continue

                dl = DlLink(**{**dgs[dg.name], "data_group": dg})
                outer_bar.set_description(f"[{seq_name[-24:]}] {dg.name}")
                try:
                    dl.get(
                        outdir,
                        ignore_existing=ignore_existing,
                        exclude_patterns=exclude_patterns,
                        stall_timeout=stall_timeout,
                        max_retries=max_retries,
                    )
                except Exception as e:
                    logger.error(f"downloading failure:, {e}")

                outer_bar.update(1)
                summary[dl.status.name] += 1
                self._logs[dl.seq_name][dl.data_group.name] = dl.status.value
                self.__logging()

        outer_bar.close()
        self.__logging(download_summary=summary)

        # Refresh the manifest with what actually landed on disk.
        write_manifest(
            self.out_rootdir,
            manifest_has,
            scan_was_downloaded(self.out_rootdir, seq_names),
        )

        logger.info(f"Dataset download to {self.out_rootdir}")
        logger.info(f"Brief download summary: {json.dumps(summary, indent=2)}")
        logger.info(f"Detailed summary saved to {self.logfile}")
