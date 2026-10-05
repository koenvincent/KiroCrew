"""Transfer a tracked project snapshot to a connected remote crew.

The hub builds the archive; this peer endpoint only verifies and installs it.
Archives are content-addressed, bounded before extraction, and may contain only
regular files and directories with relative POSIX names.  Symlink members (which
the hub's tracked-source tarball emits for tracked symlinks) are skipped, never
created; hardlinks, devices, FIFOs and traversal spellings are refused.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import secrets
import tarfile
import tempfile
import threading
import time
from collections.abc import Collection
from pathlib import Path, PurePosixPath

from aiohttp import web

from kiro_crew.atomic_write import atomic_write
from kiro_crew.dashboard.handlers._shared import _owner_denial_response
from kiro_crew.dashboard.handlers.source_providers import is_owner_dashboard_request
from kiro_crew.platform_compat import rmtree_force

_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_EXPANDED_BYTES = 512 * 1024 * 1024
_MAX_MEMBERS = 100_000
_DIGEST_RE = re.compile(r"^[a-f0-9]{64}\Z")
_COMMIT_RE = re.compile(r"^[a-f0-9]{40,64}\Z")
_MARKER = ".kirocrew-remote-workspace.json"
_WORKSPACE_TTL_SECS = 7 * 24 * 60 * 60
# ``<digest[:24]>-<nonce>``: one directory per install, never shared by two runs.
_WORKSPACE_NAME_RE = re.compile(r"^[a-f0-9]{24}-[a-f0-9]{8}\Z")
_UPLOAD_READ_CHUNK_BYTES = 256 * 1024
# install_workspace runs in worker threads; one lock keeps a prune from deleting
# a snapshot another upload is installing or handing out for reuse.
_INSTALL_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


class WorkspaceArchiveRejected(ValueError):
    """The supplied project archive is unsafe or inconsistent."""


def _reject_name(name: str) -> None:
    if not name or "\0" in name or "\\" in name:
        raise WorkspaceArchiveRejected("archive contains an unsafe member name")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise WorkspaceArchiveRejected("archive contains an absolute member path")
    path = PurePosixPath(name)
    if not path.parts or path.as_posix() != name or any(part == ".." for part in path.parts):
        raise WorkspaceArchiveRejected("archive contains path traversal")


def _member_filter(member: tarfile.TarInfo, _destination: str) -> tarfile.TarInfo | None:
    _reject_name(member.name)
    if member.issym():
        # Skipped, not created: a link target could point outside the snapshot.
        return None
    if not (member.isfile() or member.isdir()):
        raise WorkspaceArchiveRejected("archive contains a non-file member")
    scrubbed = member.replace(deep=True) if hasattr(member, "replace") else member
    scrubbed.uid = 0
    scrubbed.gid = 0
    scrubbed.uname = ""
    scrubbed.gname = ""
    if scrubbed.isdir():
        scrubbed.mode = 0o755
    else:
        scrubbed.mode = 0o755 if member.mode & 0o111 else 0o644
    return scrubbed


def _validate_archive(payload: bytes) -> None:
    count = 0
    expanded = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            while (member := archive.next()) is not None:
                count += 1
                if count > _MAX_MEMBERS:
                    raise WorkspaceArchiveRejected("archive contains too many members")
                _member_filter(member, "")
                if member.isfile():
                    expanded += max(0, int(member.size))
                    if expanded > _MAX_EXPANDED_BYTES:
                        raise WorkspaceArchiveRejected("archive expands beyond the workspace limit")
    except WorkspaceArchiveRejected:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise WorkspaceArchiveRejected("archive is not a valid gzip tarball") from exc
    if count == 0:
        raise WorkspaceArchiveRejected("archive is empty")


def _workspace_root() -> Path:
    return Path.home() / "workplace" / "kirocrew-remote-workspaces"


def _prune_workspaces(
    base: Path,
    *,
    keep: str,
    now: float | None = None,
    in_use: Collection[str] = (),
) -> int:
    """Remove expired, valid workspace snapshots except *keep* and *in_use*.

    *in_use* names the snapshots an active or queued run works in. File edits
    inside a snapshot do not refresh the directory's age, so age alone would
    delete a long-running worker's tree.
    """
    current = time.time() if now is None else now
    pruned = 0
    for child in base.iterdir():
        if (
            child.name == keep
            or child.name in in_use
            or not _WORKSPACE_NAME_RE.fullmatch(child.name)
            or child.is_symlink()
            or not child.is_dir()
        ):
            continue
        marker = child / _MARKER
        if marker.is_symlink() or not marker.is_file():
            continue
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            modified = child.stat().st_mtime
        except (OSError, ValueError, RecursionError):
            logger.warning("Remote workspace %s has an unreadable marker", child.name)
            continue
        marker_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        if (
            not isinstance(marker_digest, str)
            or not _DIGEST_RE.fullmatch(marker_digest)
            or marker_digest[:24] != child.name[:24]
            or current - modified < _WORKSPACE_TTL_SECS
        ):
            continue
        if rmtree_force(child):
            pruned += 1
        else:
            logger.warning("Could not prune expired remote workspace %s", child)
    return pruned


def install_workspace(
    payload: bytes,
    digest: str,
    commit: str,
    *,
    root: Path | None = None,
    in_use: Collection[str] = (),
) -> Path:
    """Verify and atomically install *payload* into a fresh directory; return it.

    Every install gets its own ``<digest[:24]>-<nonce>`` tree. Two runs from the
    same commit (always the case for a remote batch) produce the same digest, and
    a run edits and deletes inside its tree, so handing a later run the earlier
    run's directory would start it from a tree the hub never checked.
    """
    with _INSTALL_LOCK:
        return _install_workspace_locked(payload, digest, commit, root=root, in_use=in_use)


def _install_workspace_locked(
    payload: bytes,
    digest: str,
    commit: str,
    *,
    root: Path | None,
    in_use: Collection[str],
) -> Path:
    if not _DIGEST_RE.fullmatch(digest):
        raise WorkspaceArchiveRejected("invalid archive digest")
    if commit and not _COMMIT_RE.fullmatch(commit):
        raise WorkspaceArchiveRejected("invalid source commit")
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise WorkspaceArchiveRejected("archive exceeds the compressed workspace limit")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise WorkspaceArchiveRejected("archive digest mismatch")
    _validate_archive(payload)

    base = (root or _workspace_root()).resolve()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = base / f"{digest[:24]}-{secrets.token_hex(4)}"
    pruned = _prune_workspaces(base, keep=target.name, in_use=in_use)
    if pruned:
        logger.info("Pruned %d expired remote workspace snapshot(s)", pruned)
    if target.exists() or target.is_symlink():
        raise WorkspaceArchiveRejected("workspace target already exists in an unsafe form")

    staging = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=base))
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            try:
                archive.extractall(staging, filter=_member_filter)  # nosec B202
            except TypeError:
                members = [
                    kept
                    for member in archive.getmembers()
                    if (kept := _member_filter(member, str(staging))) is not None
                ]
                archive.extractall(staging, members=members)  # nosec B202
        atomic_write(
            staging / _MARKER,
            json.dumps({"sha256": digest, "commit": commit}, sort_keys=True) + "\n",
            fsync=True,
            mode=0o600,
        )
        os.replace(staging, target)
    except BaseException:
        if not rmtree_force(staging):
            logger.error("Remote workspace staging cleanup failed: %s", staging)
        raise
    return target


async def _read_upload(request: web.Request) -> bytes:
    """Read the upload body to EOF, stopping once it exceeds the archive cap.

    A single ``content.read(n)`` returns whatever is buffered, so an archive
    that arrives in several chunks would be cut short and fail its digest.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.content.iter_chunked(_UPLOAD_READ_CHUNK_BYTES):
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_ARCHIVE_BYTES:
            break
    return b"".join(chunks)[: _MAX_ARCHIVE_BYTES + 1]


def _workspaces_in_use(request: web.Request) -> frozenset[str]:
    """Snapshot names that a live or queued run on this gateway works in."""
    state = request.app.get("state") if hasattr(request.app, "get") else None
    manager = getattr(state, "subagents", None)
    agents = getattr(manager, "all_agents", None)
    if not isinstance(agents, list):
        return frozenset()
    bases = {_workspace_root(), _workspace_root().resolve()}
    names: set[str] = set()
    for info in agents:
        if getattr(info, "done", True):
            continue
        cwd = getattr(info, "cwd", "") or ""
        if not isinstance(cwd, str) or not cwd:
            continue
        for base in bases:
            try:
                relative = Path(cwd).relative_to(base)
            except ValueError:
                continue
            if relative.parts and _WORKSPACE_NAME_RE.fullmatch(relative.parts[0]):
                names.add(relative.parts[0])
    return frozenset(names)


async def api_remote_workspace_upload(request: web.Request) -> web.Response:
    """Install one tracked source snapshot sent through an instance tunnel."""
    if request.get("internal_auth") or request.get("app", ""):
        return _owner_denial_response(request, "dashboard owner required", "owner_only")
    if not is_owner_dashboard_request(request):
        return _owner_denial_response(request, "dashboard owner required", "owner_only")
    digest = str(request.query.get("sha256") or "")
    commit = str(request.query.get("commit") or "")
    payload = await _read_upload(request)
    if len(payload) > _MAX_ARCHIVE_BYTES:
        return web.json_response(
            {
                "error": "archive exceeds the compressed workspace limit",
                "code": "archive_too_large",
            },
            status=413,
        )
    in_use = _workspaces_in_use(request)
    try:
        path = await asyncio.to_thread(install_workspace, payload, digest, commit, in_use=in_use)
    except WorkspaceArchiveRejected as exc:
        return web.json_response(
            {"error": str(exc), "code": "workspace_archive_rejected"}, status=400
        )
    except OSError:
        return web.json_response(
            {"error": "workspace installation failed", "code": "workspace_install_failed"},
            status=500,
        )
    return web.json_response({"path": str(path), "sha256": digest, "commit": commit})
