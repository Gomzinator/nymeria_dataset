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
from stat import S_ISDIR
from zipfile import is_zipfile, ZipFile

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from .definitions import DataGroups, FILENAME_PREFIXES, NYMERIA_VERSION
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
    ERR_EXPIRED = "error, download link expired — refresh the url json"


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
        # Accept any known release prefix (base Nymeria + NymeriaPlus, which is
        # where the wrist data.vrs links come from) and strip it, so seq_name
        # parsing below sees a bare <date>_<s?>_<first>_<last>_<act>_<uid>_...
        for prefix in FILENAME_PREFIXES:
            if prefix in self.filename:
                self.filename = self.filename.replace(prefix, "")
                return
        self.status = (
            f"Version mismatch with the release {NYMERIA_VERSION}. "
            f"Please download the latest url json"
        )
        raise ValueError(self.status)

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
            missing = self._missing_payload(outdir, exclude_patterns)
            if not missing:
                self.status = DlStatus.IGNORED
                return
            # The flag is stale: this group was downloaded under a different
            # selection (e.g. --wrist-video stripped motion.vrs, and we are now
            # running without it). Re-fetch so the swap is reversible.
            logger.info(
                f"{self.data_group.name}: re-downloading, missing {missing}"
            )

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

    def _missing_payload(
        self, outdir: Path, exclude_patterns: list[str] | None
    ) -> list[str]:
        """
        VRS files this group should have delivered but that are absent on disk.

        Used to detect a stale "done" flag when the selection changed between
        runs — the motion.vrs <-> data.vrs swap is the case that matters, since
        turning --wrist-video (or --video) back off must bring motion.vrs back
        rather than leave recording_*/data empty forever.

        Deliberately limited to .vrs files: they are the payload every recording
        zip always carries, whereas some MPS csvs are legitimately absent for
        some sequences (e.g. personalized_eye_gaze.csv), which would otherwise
        trigger a pointless re-download on every single run.
        """
        from .definitions import get_group_definitions

        rels = get_group_definitions().get(
            self.data_group.name, [self.data_group.value]
        )
        return [
            rel
            for rel in rels
            if rel.endswith(".vrs")
            and not (exclude_patterns and any(p in rel for p in exclude_patterns))
            and not (outdir / rel).is_file()
        ]

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
                # Check the HTTP status BEFORE streaming. These are signed CDN
                # urls with a short lifetime: once the json ages out the server
                # answers 403 "URL signature expired" with a 21-byte text body.
                # Hashing that body and reporting "sha1sum mismatch" (what this
                # code used to do, since raise_for_status ran last) sends you
                # hunting for corruption instead of refreshing the json.
                if r.status_code == 403:
                    self.status = DlStatus.ERR_EXPIRED
                    raise RuntimeError(
                        f"{r.status_code} {r.reason} for {self.filename} — the "
                        f"download links in the url json have expired. "
                        f"Re-download the url json from the Nymeria website."
                    )
                try:
                    r.raise_for_status()
                except Exception as e:
                    self.status = DlStatus.ERR_NETWORK
                    raise RuntimeError(e) from e

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
                # tempfile creates the staged file 0600 and shutil.move preserves
                # that, leaving non-zip downloads (data.vrs, mvnx, CSVs) owner-only
                # while zip-extracted files inherit the share's perms. Match the
                # parent directory so all files in the tree are consistent.
                shutil.copymode(dst_file.parent, dst_file)


class DownloadManager:
    def __init__(
        self,
        url_json: Path | list[Path],
        out_rootdir: Path,
        dry_run: bool = False,
    ) -> None:
        # One or more url jsons, merged per sequence per data group. The base
        # release and the NymeriaPlus release ship separate jsons that describe
        # disjoint data groups for the same 1100 sequences (the wrist data.vrs
        # links only exist in the plus json), so both are needed to download the
        # full set. On a (sequence, group) collision the later json wins.
        self.url_jsons = [url_json] if isinstance(url_json, Path) else list(url_json)
        assert self.url_jsons, "No url json provided"
        for p in self.url_jsons:
            assert p.is_file(), f"{p} not found"
        self.url_json = self.url_jsons[0]

        self.out_rootdir = out_rootdir
        self.out_rootdir.mkdir(exist_ok=True)

        self._sequences: dict[str, dict] = {}
        # One representative url per json, used by _preflight() to detect an
        # expired json before anything destructive happens. Signed urls in a
        # given json share an expiry window, so one sample is a good proxy.
        self._preflight_samples: dict[Path, str] = {}
        for p in self.url_jsons:
            with open(p, "r") as f:
                sequences = json.load(f).get("sequences", {})
            assert len(sequences), f"No sequence found in {p}. Please check the json file is correct."
            for seq, dgs in sequences.items():
                self._sequences.setdefault(seq, {}).update(dgs)
                for entry in dgs.values():
                    if p not in self._preflight_samples and "download_url" in entry:
                        self._preflight_samples[p] = entry["download_url"]
        if len(self.url_jsons) > 1:
            logger.info(
                f"merged {len(self.url_jsons)} url jsons → {len(self._sequences)} sequences"
            )

        # data_summary.json is a record of the input json(s) written into the
        # output dir. A --dry-run must leave the output dir untouched, so skip it.
        if not dry_run:
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
        exclude_patterns: list[str] | None = None,
        prune_targets: list[Path] | None = None,
        prune_freed_gb: float = 0.0,
        dry_run: bool = False,
    ) -> set["DataGroups"]:
        selected_groups += [DataGroups.LICENSE, DataGroups.metadata_json]
        selected_groups = set(selected_groups)

        num_seqs = 0
        num_files = 0        # files that will actually be downloaded
        num_skipped = 0      # files already on disk (flag present)
        total_gb = 0         # full project size (all selected files)
        pending_gb = 0       # size of files still to download
        # Per-action plan, only used to print the --dry-run breakdown. Each entry
        # is (seq, group.value[, missing_payload]) so the dry-run can spell out
        # the motion.vrs <-> data.vrs swaps rather than just a count.
        plan_new: list[tuple[str, str]] = []       # fresh downloads (no flag yet)
        plan_redl: list[tuple[str, str, list]] = []  # stale flag → re-download
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
                    # Mirror DlLink.download exactly: a "done" flag only counts
                    # as done if the group's vrs payload is actually on disk. A
                    # stale flag (selection changed between runs — e.g. reverting
                    # --wrist-video, where data.vrs must go back to motion.vrs)
                    # means the file WILL be re-downloaded, so it has to be
                    # counted here too. Otherwise the confirmation prompt and,
                    # worse, the free-space check understate the real transfer.
                    flag_ok = flag.is_file() and ignore_existing
                    missing = (
                        dl._missing_payload(outdir, exclude_patterns)
                        if flag_ok
                        else []
                    )
                    if flag_ok and not missing:
                        num_skipped += 1
                    else:
                        pending_gb += file_gb
                        num_files += 1
                        if flag_ok:  # stale flag → re-download the stripped payload
                            plan_redl.append((seq, dg.value, missing))
                        else:
                            plan_new.append((seq, dg.value))

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

        summary = (
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
        )

        if dry_run:
            # Print the same summary plus a per-file action breakdown, then let
            # download() return without touching disk or the network. This is the
            # supported way to preview the motion.vrs <-> data.vrs swaps: prune
            # (delete) + re-download (restore stripped payload) + new download.
            print(
                "DRY RUN — nothing will be downloaded, pruned, or modified.\n\n"
                + summary
                + self._format_plan(prune_targets or [], plan_redl, plan_new)
            )
            return selected_groups

        confirm = input(summary + "Proceed: [y/n] ").lower() == "y"
        if not confirm:
            exit(1)
        return selected_groups

    # Cap on how many individual paths each --dry-run section prints, so a
    # full-dataset preview does not scroll off thousands of lines. The counts in
    # the summary above are always exact; only the verbose listing is truncated.
    _DRY_RUN_MAX_LINES = 60

    def _format_plan(
        self,
        prune_targets: list[Path],
        plan_redl: list[tuple[str, str, list]],
        plan_new: list[tuple[str, str]],
    ) -> str:
        """Render the --dry-run action breakdown (prune / re-download / new)."""

        def section(title: str, lines: list[str]) -> str:
            shown = lines[: self._DRY_RUN_MAX_LINES]
            body = "\n".join(f"    {ln}" for ln in shown) if shown else "    (none)"
            more = len(lines) - len(shown)
            if more > 0:
                body += f"\n    ... and {more} more"
            return f"  {title}\n{body}\n"

        prune_lines = sorted(
            p.relative_to(self.out_rootdir).as_posix() for p in prune_targets
        )
        # Re-download == the swap's restore leg: flag present but the vrs payload
        # was stripped by a previous selection (e.g. reverting --wrist-video or
        # --video). Show which file is being brought back.
        redl_lines = [
            f"{seq}/{grp}"
            + (f"   (restores {', '.join(missing)})" if missing else "")
            for seq, grp, missing in sorted(plan_redl)
        ]
        new_lines = [f"{seq}/{grp}" for seq, grp in sorted(plan_new)]

        return (
            "\nPlanned actions\n"
            + section(f"Prune — delete {len(prune_lines)} file(s):", prune_lines)
            + section(
                f"Re-download — restore stripped payload, {len(redl_lines)} group(s):",
                redl_lines,
            )
            + section(f"Download new — {len(new_lines)} group(s):", new_lines)
        )

    def _compute_prune_targets(
        self,
        match_key: str,
        selected_groups: set["DataGroups"],
        exclude_patterns: list[str] | None = None,
    ) -> tuple[list[Path], float]:
        """
        Dry-run: compute which files on disk don't belong to the selected groups.

        L_expected  = file paths that should exist for the given groups
        L_exists    = files currently on disk under out_rootdir, restricted to
                      the sequences selected by match_key
        L_delete    = L_exists - L_expected

        Scoping to match_key matters: prune must never touch a sequence this run
        is not managing. Without it, `-k <one_seq> --prune` would treat every
        other sequence on disk as unexpected and delete it wholesale.

        Returns (files_to_delete, freed_gb). Nothing is deleted.
        """
        from .definitions import get_group_definitions

        group_defs = get_group_definitions()

        # Build L_expected (posix paths relative to out_rootdir) and, alongside,
        # the set of sequence dirs prune is allowed to descend into.
        expected: set[str] = set()
        pruneable_seqs: set[str] = set()
        for seq_name in self.sequences:
            if match_key not in seq_name:
                continue
            pruneable_seqs.add(seq_name)
            for dg in selected_groups:
                for rel in group_defs.get(dg.name, [dg.value]):
                    if exclude_patterns and any(pat in rel for pat in exclude_patterns):
                        continue
                    expected.add(f"{seq_name}/{rel}")
                # keep the "done" flag for this group
                expected.add(f"{seq_name}/logs/{dg.name}")

        # Build L_delete. Only files inside a matched sequence dir are eligible,
        # which also keeps the top-level bookkeeping files (download_summary.json,
        # data_summary.json, manifest.json) and any unrelated content safe.
        #
        # One stat per entry (not is_file() + a separate stat() pass): over a
        # slow SMB share a two-pass scan of 1100 sequences lets the session go
        # stale between passes, and each stat is a network round trip. Stat is
        # resilient (see _stat_resilient) so a transient blip does not abort the
        # whole scan — which is what crashed a real 1100-sequence NAS run.
        to_delete: list[Path] = []
        freed_bytes = 0
        unreadable: list[Path] = []
        for seq_name in sorted(pruneable_seqs):
            seq_dir = self.out_rootdir / seq_name
            st = self._stat_resilient(seq_dir, unreadable)
            if st is None or not S_ISDIR(st.st_mode):
                continue
            try:
                # Materialise this one sequence's walk so a mid-walk network
                # error is caught per-sequence instead of killing the whole scan.
                entries = list(seq_dir.rglob("*"))
            except OSError as e:
                logger.warning(f"prune scan: could not list {seq_name} ({e}); skipped")
                unreadable.append(seq_dir)
                self._abort_if_fs_unstable(unreadable)
                continue
            for p in entries:
                if p.relative_to(self.out_rootdir).as_posix() in expected:
                    continue
                st = self._stat_resilient(p, unreadable)
                if st is None:
                    self._abort_if_fs_unstable(unreadable)
                    continue
                if S_ISDIR(st.st_mode):
                    continue
                to_delete.append(p)
                freed_bytes += st.st_size

        if unreadable:
            logger.warning(
                f"prune scan: {len(unreadable)} path(s) stayed unreadable after "
                "retries (network/SMB blips); excluded from the freed-space "
                "estimate. Re-run if the reported prune size looks low."
            )
        freed_gb = freed_bytes / (2**30)
        return to_delete, freed_gb

    # Above this many unreadable paths, assume the output filesystem is down
    # (e.g. an SMB share dropped) rather than a scattered blip, and abort the
    # scan loudly instead of grinding through thousands of retrying stats. Safe:
    # this is the dry-run costing phase — nothing has been deleted yet.
    _MAX_PRUNE_SCAN_ERRORS = 25

    def _abort_if_fs_unstable(self, unreadable: list[Path]) -> None:
        if len(unreadable) > self._MAX_PRUNE_SCAN_ERRORS:
            raise RuntimeError(
                f"prune scan aborted: {len(unreadable)} paths under "
                f"{self.out_rootdir} were unreadable — the output filesystem "
                "looks unavailable (network share down?). Nothing was deleted; "
                "re-run once it is stable."
            )

    @staticmethod
    def _stat_resilient(p: Path, unreadable: list[Path]):
        """
        p.stat(), retrying transient network errors, for scanning over SMB.

        Windows surfaces a dropped/again-unreachable share as WinError 53/64/...
        which Python raises as FileNotFoundError/OSError — the same type as a
        genuinely absent file. Only ERROR_FILE_NOT_FOUND (2) / ERROR_PATH_NOT_
        FOUND (3), and a POSIX ENOENT, mean "really gone" (return None silently);
        anything else is treated as transient and retried. After the last retry
        the path is recorded in `unreadable` and None is returned, so the caller
        can keep scanning instead of crashing.
        """
        delays = (0.5, 1, 2, 4)
        for i, delay in enumerate(delays):
            try:
                return p.stat()
            except OSError as e:
                winerr = getattr(e, "winerror", None)
                really_missing = winerr in (2, 3) or (
                    winerr is None and isinstance(e, FileNotFoundError)
                )
                if really_missing:
                    return None
                if i == len(delays) - 1:
                    logger.warning(f"stat: giving up on {p} after retries ({e})")
                    unreadable.append(p)
                    return None
                time.sleep(delay)
        return None

    def _preflight(self) -> list[Path]:
        """
        Probe one url per input json; return the jsons whose links are dead.

        Signed CDN urls expire (403 "URL signature expired") a few weeks after
        the json is generated. That matters most right before --prune: prune
        deletes first and downloads second, so pruning against an expired json
        destroys files that cannot be re-fetched. One 1-byte ranged GET per json
        is enough to tell.
        """
        expired: list[Path] = []
        for path, url in self._preflight_samples.items():
            try:
                r = requests.get(
                    url,
                    stream=True,
                    headers={"Range": "bytes=0-0"},
                    timeout=DlConfig.CONNECT_TIMEOUT.value,
                )
                r.close()
            except requests.exceptions.RequestException as e:
                logger.warning(f"preflight for {path.name} could not complete: {e}")
                continue
            if r.status_code == 403:
                expired.append(path)
        return expired

    @staticmethod
    def _execute_prune(to_delete: list[Path], out_rootdir: Path) -> None:
        """Delete the given files and remove directories they leave empty."""
        # Tolerate per-file errors: a transient SMB blip must not abort a real
        # prune half-way (that is how a recording gets partially wiped). An
        # already-absent file is a no-op success.
        deleted_parents: set[Path] = set()
        for p in to_delete:
            try:
                p.unlink()
                deleted_parents.add(p.parent)
            except FileNotFoundError:
                deleted_parents.add(p.parent)
            except OSError as e:
                logger.warning(f"prune: could not delete {p} ({e})")

        # Remove now-empty dirs by walking UP from each deleted file's parent.
        # The old code rglob'd the entire out_rootdir, which over SMB re-walks
        # all 1100 sequences (minutes) and, on a scoped -k run, needlessly
        # touches unrelated sequences. rmdir only succeeds on an empty dir, so
        # this stops at the first non-empty ancestor.
        for parent in sorted(deleted_parents, key=lambda d: len(d.parts), reverse=True):
            cur = parent
            while cur != out_rootdir and cur.is_dir():
                try:
                    cur.rmdir()
                except OSError:
                    break  # not empty (or unreachable) — leave it and stop
                cur = cur.parent

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
        dry_run: bool = False,
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
            exclude_patterns=exclude_patterns,
            prune_targets=prune_targets,
            prune_freed_gb=prune_freed_gb,
            dry_run=dry_run,
        )

        # --dry-run stops here: the summary + action plan have been printed and
        # nothing on disk (manifest, prune, downloads) or the network (expiry
        # preflight) is touched. The real run repeats this preview as its [y/n]
        # confirmation prompt.
        if dry_run:
            return

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

        # Prune first to free space before the download starts. Because that
        # ordering is destructive-before-restorative, refuse to prune at all if
        # any input json has expired: otherwise we delete a motion.vrs (or a
        # data.vrs) and then fail to fetch its replacement, leaving the
        # recording empty with no way back.
        if prune_targets:
            expired = self._preflight()
            if expired:
                logger.error(
                    "Refusing to prune: the download links have expired in "
                    + ", ".join(p.name for p in expired)
                    + ". Pruning now would delete files that cannot be "
                    "re-downloaded. Re-download the url json(s) from the "
                    "Nymeria website, then re-run."
                )
                raise SystemExit(1)
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
