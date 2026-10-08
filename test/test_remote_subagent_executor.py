from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import tarfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest.mock import AsyncMock

import pytest

import kiro_crew
from kiro_crew.dashboard.remote_subagents import (
    RemoteSubagentError,
    RemoteSubagentService,
)
from kiro_crew.dashboard.remote_workspaces import (
    WorkspaceArchiveRejected,
    api_remote_workspace_upload,
    install_workspace,
)
from kiro_crew.mcp_tools import spawn as spawn_tools
from kiro_crew.subagent import SubagentInfo, SubagentManager
from kiro_crew.subagent_manager.published_depth import PublishedQueueDepths
from kiro_crew.validation import SPAWN_RUN_SCHEMA, validate_tool_args

# The local-path cases drive SubagentManager.spawn; pin the host-memory reading
# so a memory-pressured runner does not queue them.
pytestmark = pytest.mark.usefixtures("healthy_host_memory")

# A monitor that never settles (a scripted peer reply list run dry, a retry loop
# that keeps re-polling) must fail its own test, not hang the xdist worker until
# pytest's global timeout kills it.
_MONITOR_SETTLE_SECONDS = 10.0


async def _settle_monitor(service: RemoteSubagentService, run_id: str) -> None:
    """Wait for one run's monitor, bounded; stop the service if it never settles."""
    try:
        await asyncio.wait_for(service._monitors[run_id], timeout=_MONITOR_SETTLE_SECONDS)
    except TimeoutError:
        await service.close()
        pytest.fail(
            f"remote monitor for {run_id} did not settle within "
            f"{_MONITOR_SETTLE_SECONDS}s (raise _MONITOR_SETTLE_SECONDS if the host is slow)"
        )


class _Content:
    """aiohttp StreamReader double that delivers the body in small chunks.

    Like the real reader, ``read(n)`` resolves with whatever is buffered -- here
    only the first chunk -- so code that does a single read sees a truncated
    body. Readers must drain ``iter_chunked`` to EOF.
    """

    def __init__(self, body: bytes, chunk: int = 7) -> None:
        self._body = body
        self._chunk = chunk

    async def read(self, limit: int) -> bytes:
        return self._body[: min(limit, self._chunk)]

    async def iter_chunked(self, size: int):
        step = max(1, min(size, self._chunk))
        for start in range(0, len(self._body), step):
            yield self._body[start : start + step]


class _Response:
    def __init__(self, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self.content = _Content(json.dumps(payload).encode("utf-8"))


class _Instances:
    def __init__(
        self,
        responses: list[tuple[int, dict[str, object]] | BaseException],
        *,
        connected: tuple[str, ...] = ("crew-a", "crew-b"),
        legacy_peer: bool = False,
    ) -> None:
        self._responses = list(responses)
        self._connected = connected
        # A current peer echoes what it enforced on an accepted spawn; a peer
        # without that change (same major.minor) silently drops the fields.
        self._legacy_peer = legacy_peer
        self.calls: list[tuple[str, str, str, dict[str, object] | bytes | None]] = []

    def status(self, instance_id: str) -> object | None:
        if instance_id not in self._connected:
            return None
        return SimpleNamespace(state=SimpleNamespace(value="connected"))

    def status_all(self) -> dict[str, object]:
        return {
            instance_id: SimpleNamespace(state=SimpleNamespace(value="connected"))
            for instance_id in self._connected
        }

    async def peer_version(self, _instance_id: str) -> tuple[bool, str]:
        return True, kiro_crew.__version__

    @asynccontextmanager
    async def proxy_request(
        self,
        instance_id: str,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        **_kwargs: object,
    ):
        try:
            body = json.loads(data) if data else None
        except (ValueError, UnicodeDecodeError):
            body = data
        self.calls.append((instance_id, method, path, body))
        scripted = self._responses.pop(0)
        if isinstance(scripted, BaseException):
            raise scripted
        status, payload = scripted
        if (
            not self._legacy_peer
            and method == "POST"
            and path == "api/spawn"
            and 200 <= status < 300
            and isinstance(body, dict)
            and "applied" not in payload
        ):
            payload = {
                **payload,
                "applied": {
                    "memory_mode": body.get("memory_mode") or "persistent",
                    "approval_floor": body.get("approval_floor", ""),
                },
            }
        yield _Response(status, payload)


class _Sessions:
    """The two session-store reads the hub's approval floor makes."""

    def __init__(self, policy: str = "", present: bool | None = True) -> None:
        self.policy = policy
        self.present = present

    def get_approval_policy(self, _key: str) -> str:
        return self.policy

    def has_session(self, _key: str) -> bool | None:
        return self.present


class _Manager:
    def __init__(self) -> None:
        self.external_agents: list[SubagentInfo] = []
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self.reported: list[SubagentInfo] = []
        self._sessions: Any = _Sessions()
        self._is_yolo: Callable[[], bool] | None = None
        self._global_approval_mode = ""
        self._completion_keep = "head"
        self._completion_keep_chars = 3
        self._result_ttl_secs = 3600
        self.external_placement: Any = None

    @property
    def _external_agents(self) -> dict[str, SubagentInfo]:
        return {info.id: info for info in self.external_agents}

    def bind_external_placement(self, placement: Any) -> None:
        self.external_placement = placement

    def get(self, agent_id: str) -> SubagentInfo | None:
        return next((info for info in self.external_agents if info.id == agent_id), None)

    def forget_external(self, agent_id: str) -> SubagentInfo | None:
        info = self.get(agent_id)
        if info is not None:
            self.external_agents.remove(info)
        return info

    def register_external(self, info: SubagentInfo) -> None:
        self.external_agents.append(info)

    async def _fire_event(self, event: str, info: SubagentInfo, extra: dict[str, object]) -> None:
        self.events.append((event, info.id, extra))

    async def report_external(self, info: SubagentInfo) -> bool:
        self.reported.append(info)
        return True


@pytest.mark.asyncio
async def test_remote_spawn_picks_least_loaded_peer_and_relays_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    instances = _Instances(
        [
            (200, {"id": "peer01"}),
            (200, {"done": False, "turns": 2, "last_tool": "Reading src"}),
            (
                200,
                {
                    "done": True,
                    "result": "remote-result",
                    "error": "",
                    "outcome": "completed",
                    "partial": False,
                },
            ),
        ]
    )
    manager = _Manager()
    manager.external_agents.append(
        SubagentInfo(
            id="occupied",
            task="existing",
            executor="remote",
            instance_id="crew-a",
            remote_id="peer00",
        )
    )
    state = SimpleNamespace(instances_manager=instances)
    service = RemoteSubagentService(state, manager)  # type: ignore[arg-type]

    info = await service.spawn(
        task="review exact refactor SHA",
        parent_session="dashboard:chat-1",
        agent="kirocrew",
        max_turns=50,
        cwd="/home/kirocrew/workplace/repo",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=True,
        memory_mode="temporary",
        batch_id="batch01",
        batch_total=1,
    )
    await _settle_monitor(service, info.id)

    assert info.instance_id == "crew-b"
    assert info.remote_id == "peer01"
    assert info.done is True
    assert info.result == "rem"
    assert info.result_truncated is True
    assert manager.reported == [info]
    assert manager.events[0][0] == "subagent_spawn"
    post = instances.calls[0]
    assert post[:3] == ("crew-b", "POST", "api/spawn")
    assert post[3] == {
        "task": "review exact refactor SHA",
        "parent_session": "",
        "silent": True,
        "include_memory": False,
        "include_lessons": False,
        "include_project": True,
        "memory_mode": "temporary",
        "agent": "kirocrew",
        "max_turns": 50,
        "cwd": "/home/kirocrew/workplace/repo",
        # A dashboard parent in Normal mode: the crew must not auto-approve.
        "approval_floor": "interactive",
    }
    await service.close()


@pytest.mark.asyncio
async def test_remote_spawn_refuses_disconnected_explicit_peer() -> None:
    manager = _Manager()
    service = RemoteSubagentService(
        cast(
            Any,
            SimpleNamespace(instances_manager=_Instances([], connected=("crew-a",))),
        ),
        cast(Any, manager),
    )

    with pytest.raises(RemoteSubagentError) as raised:
        await service.spawn(
            task="x",
            parent_session="dashboard:chat-1",
            agent="",
            max_turns=0,
            cwd="",
            model="",
            reasoning_effort="",
            include_memory=False,
            include_lessons=False,
            include_project=False,
            memory_mode="temporary",
            batch_id="",
            batch_total=0,
            instance_id="crew-z",
        )

    assert raised.value.code == "remote_instance_not_connected"
    assert manager.external_agents == []


@pytest.mark.asyncio
async def test_remote_cancel_targets_peer_run() -> None:
    instances = _Instances([(200, {"ok": True, "cancelled": True})])
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), _Manager()  # type: ignore[arg-type]
    )
    info = SubagentInfo(
        id="local01",
        task="x",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )

    assert await service.cancel(info) is True
    assert instances.calls == [("crew-a", "DELETE", "api/spawn/peer01", None)]


@pytest.mark.asyncio
async def test_an_accepted_remote_spawn_and_its_cancel_are_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A remote child leaves this gateway's approval ceiling, so its placement
    # and its stop must reach the SEL trail just like a local run's.
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "kiro_crew.sel.sel",
        lambda: SimpleNamespace(log_tool_invocation=lambda **kw: events.append(kw)),
    )
    instances = _Instances([(200, {"id": "peer01"}), (200, {"ok": True, "cancelled": True})])
    service = RemoteSubagentService(
        cast(Any, SimpleNamespace(instances_manager=instances)), cast(Any, _Manager())
    )
    info = await service.spawn(
        task="x",
        parent_session="dashboard:chat-1",
        agent="reviewer",
        max_turns=0,
        cwd="/home/kirocrew/workplace/repo",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=False,
        memory_mode="persistent",
        batch_id="",
        batch_total=0,
    )
    await service.close()  # the monitor must not consume the cancel reply
    assert await service.cancel(info) is True

    assert [e["outcome"] for e in events] == ["spawned", "cancelled"]
    for event in events:
        assert event["tool_name"] == "spawn_run"
        assert event["session_key"] == "dashboard:chat-1"
        assert event["metadata"] == {
            "subagent_id": info.id,
            "executor": "remote",
            "instance_id": info.instance_id,
            "remote_id": "peer01",
            "agent": "reviewer",
        }


def test_manager_external_inventory_does_not_consume_local_running_capacity() -> None:
    manager = SubagentManager.__new__(SubagentManager)
    manager._agents = {
        "local01": SubagentInfo(id="local01", task="local"),
    }
    manager._external_agents = {}
    manager._batch_submitted = {}
    manager._batch_progress_ts = {}
    remote = SubagentInfo(
        id="remote01",
        task="remote",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
        batch_id="batch01",
        batch_total=1,
    )

    manager.register_external(remote)

    assert manager.get("remote01") is remote
    assert {info.id for info in manager.all_agents} == {"local01", "remote01"}
    assert [info.id for info in manager.running] == ["local01"]
    assert manager._batch_submitted["batch01"] == [1, 1]


def test_spawn_schema_accepts_remote_pool_selection() -> None:
    cleaned = validate_tool_args(
        {"task": "review", "executor": "remote", "instance_id": "crew-a"},
        SPAWN_RUN_SCHEMA,
    )
    assert cleaned["executor"] == "remote"
    assert cleaned["instance_id"] == "crew-a"


def test_spawn_tool_forwards_remote_placement_and_reports_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(spawn_tools.mcp_core, "_resolve_session_key", lambda: "dashboard:chat-1")

    def post(path: str, body: dict[str, object]) -> dict[str, object]:
        assert path == "/api/spawn"
        calls.append(body)
        return {
            "id": f"run{len(calls)}",
            "executor": "remote",
            "instance_id": "crew-a",
        }

    monkeypatch.setattr(spawn_tools.mcp_core, "_post", post)
    result = spawn_tools.spawn_run(
        "spawn_run",
        {
            "tasks": ["review a", "review b"],
            "executor": "remote",
            "instance_id": "crew-a",
            "include_memory": False,
            "include_lessons": False,
        },
    )

    assert len(calls) == 2
    assert all(call["executor"] == "remote" for call in calls)
    assert all(call["instance_id"] == "crew-a" for call in calls)
    assert "[remote:crew-a]" in result


def _tar_payload(entries: list[tuple[str, bytes, int, str]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data, mode, kind in entries:
            member = tarfile.TarInfo(name)
            member.mode = mode
            if kind == "file":
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = data.decode("utf-8")
                archive.addfile(member)
    return buffer.getvalue()


def test_workspace_install_is_content_addressed_and_rejects_unsafe_members(
    tmp_path,
) -> None:
    payload = _tar_payload(
        [
            ("README.md", b"hello\n", 0o644, "file"),
            ("scripts/run.sh", b"#!/bin/sh\n", 0o755, "file"),
        ]
    )
    digest = hashlib.sha256(payload).hexdigest()

    path = install_workspace(payload, digest, "a" * 40, root=tmp_path)

    assert (path / "README.md").read_text(encoding="utf-8") == "hello\n"
    if os.name != "nt":
        # Windows has no POSIX execute bit; the mode is preserved only on POSIX.
        assert (path / "scripts/run.sh").stat().st_mode & 0o111
    # Same digest, fresh tree: a second run never starts in the first one's edits.
    (path / "README.md").write_text("edited by run 1\n", encoding="utf-8")
    second = install_workspace(payload, digest, "a" * 40, root=tmp_path)
    assert second != path and second.name[:24] == path.name[:24] == digest[:24]
    assert (second / "README.md").read_text(encoding="utf-8") == "hello\n"

    for bad in (
        _tar_payload([("../escape", b"x", 0o644, "file")]),
        _tar_payload([("nested//file", b"x", 0o644, "file")]),
        _tar_payload([("nested/./file", b"x", 0o644, "file")]),
        _tar_payload([("../link", b"/etc/passwd", 0o777, "symlink")]),
    ):
        with pytest.raises(WorkspaceArchiveRejected):
            install_workspace(bad, hashlib.sha256(bad).hexdigest(), "b" * 40, root=tmp_path)


def test_workspace_install_skips_tracked_symlinks_instead_of_rejecting(tmp_path) -> None:
    # build_source_tarball emits a symlink member for every tracked symlink; a
    # repo with one must still sync, and the link must never be created.
    payload = _tar_payload(
        [
            ("README.md", b"hello\n", 0o644, "file"),
            ("link", b"/etc/passwd", 0o777, "symlink"),
        ]
    )
    path = install_workspace(payload, hashlib.sha256(payload).hexdigest(), "a" * 40, root=tmp_path)

    assert (path / "README.md").is_file()
    assert not (path / "link").exists() and not (path / "link").is_symlink()


def test_workspace_prune_spares_snapshots_in_use(tmp_path) -> None:
    from kiro_crew.dashboard.remote_workspaces import _prune_workspaces

    def install(name: bytes) -> Path:
        payload = _tar_payload([("f", name, 0o644, "file")])
        return install_workspace(
            payload, hashlib.sha256(payload).hexdigest(), "a" * 40, root=tmp_path
        )

    busy, idle = install(b"busy"), install(b"idle")
    old = time.time() - 8 * 24 * 60 * 60
    for path in (busy, idle):
        os.utime(path, (old, old))

    assert _prune_workspaces(tmp_path, keep="", in_use={busy.name}) == 1
    assert busy.is_dir() and not idle.exists()


@pytest.mark.asyncio
async def test_remote_spawn_syncs_parent_project_when_no_remote_cwd(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    payload = b"tracked-source"
    digest = hashlib.sha256(payload).hexdigest()
    instances = _Instances(
        [
            (
                200,
                {
                    "path": "/srv/peer-snapshots/abc",
                    "sha256": digest,
                    "commit": "a" * 40,
                },
            ),
            (200, {"id": "peer01"}),
            (200, {"done": True, "result": "ok", "error": ""}),
        ],
        connected=("crew-a",),
    )
    manager = _Manager()
    manager.external_agents.clear()
    state = SimpleNamespace(
        instances_manager=instances,
        get_slot=lambda _name: SimpleNamespace(project=str(tmp_path)),
    )
    service = RemoteSubagentService(state, manager)  # type: ignore[arg-type]
    monkeypatch.setattr(
        service, "_build_project_archive", lambda _path: (payload, digest, "a" * 40)
    )

    info = await service.spawn(
        task="review",
        parent_session="dashboard:chat-1",
        agent="",
        max_turns=0,
        cwd="",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=True,
        memory_mode="temporary",
        batch_id="",
        batch_total=0,
    )
    await _settle_monitor(service, info.id)

    assert instances.calls[0][:3] == ("crew-a", "POST", "api/remote-workspaces")
    spawn_body = instances.calls[1][3]
    assert isinstance(spawn_body, dict)
    assert spawn_body["cwd"] == "/srv/peer-snapshots/abc"
    assert info.cwd == "/srv/peer-snapshots/abc"
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parent_session", "slot"),
    (
        ("slack:C123:1712793600.123456", None),
        ("cron:job-1", None),
        ("dashboard:chat-1", SimpleNamespace(project="")),
    ),
)
async def test_remote_spawn_without_a_parent_project_runs_without_a_snapshot(
    monkeypatch: pytest.MonkeyPatch, parent_session: str, slot: object
) -> None:
    """``include_project`` is on by default, so a parent with no project (a
    channel, cron or CLI parent, or a project-less chat) must still spawn:
    no snapshot is uploaded and the peer gets no cwd."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    instances = _Instances(
        [
            (200, {"id": "peer01"}),
            (200, {"done": True, "result": "ok", "error": ""}),
        ],
        connected=("crew-a",),
    )
    manager = _Manager()
    manager.external_agents.clear()
    state = SimpleNamespace(instances_manager=instances, get_slot=lambda _name: slot)
    service = RemoteSubagentService(state, manager)  # type: ignore[arg-type]

    def no_archive(_path: object) -> tuple[bytes, str, str]:
        raise AssertionError("no project, so nothing may be packaged")

    monkeypatch.setattr(service, "_build_project_archive", no_archive)

    info = await service.spawn(
        task="review",
        parent_session=parent_session,
        agent="",
        max_turns=0,
        cwd="",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=True,
        memory_mode="temporary",
        batch_id="",
        batch_total=0,
    )
    await _settle_monitor(service, info.id)

    assert [call[:3] for call in instances.calls][0] == ("crew-a", "POST", "api/spawn")
    assert not any(call[2] == "api/remote-workspaces" for call in instances.calls)
    spawn_body = instances.calls[0][3]
    assert isinstance(spawn_body, dict)
    assert "cwd" not in spawn_body
    assert info.cwd == ""
    await service.close()


@pytest.mark.asyncio
async def test_remote_spawn_refuses_a_parent_project_that_is_gone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "removed"
    instances = _Instances([], connected=("crew-a",))
    state = SimpleNamespace(
        instances_manager=instances,
        get_slot=lambda _name: SimpleNamespace(project=str(gone)),
    )
    service = RemoteSubagentService(state, _Manager())  # type: ignore[arg-type]

    with pytest.raises(RemoteSubagentError) as caught:
        await service.spawn(
            task="review",
            parent_session="dashboard:chat-1",
            agent="",
            max_turns=0,
            cwd="",
            model="",
            reasoning_effort="",
            include_memory=False,
            include_lessons=False,
            include_project=True,
            memory_mode="temporary",
            batch_id="",
            batch_total=0,
        )
    assert caught.value.code == "remote_project_unavailable"
    assert instances.calls == []
    await service.close()


@pytest.mark.asyncio
async def test_restart_restores_terminal_mapping_and_redelivers_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    run_dir = tmp_path / "subagents" / "remote" / "abcdef12"
    run_dir.mkdir(parents=True)
    result_path = run_dir / "result.txt"
    result_path.write_text("restored-result", encoding="utf-8")
    original = SubagentInfo(
        id="abcdef12",
        task="review",
        parent_session_key="dashboard:chat-1",
        agent="kirocrew",
        done=True,
        result="restored-result",
        result_path=str(result_path),
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    RemoteSubagentService._persist_info(original, delivered=False)

    manager = _Manager()
    service = RemoteSubagentService(SimpleNamespace(instances_manager=None), manager)  # type: ignore[arg-type]
    await service.ensure_restored()

    restored = manager.get("abcdef12")
    assert restored is not None
    assert restored.executor == "remote"
    assert restored.result == "res"
    assert restored.result_truncated is True
    assert manager.reported == [restored]
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["delivered"] is True
    await service.close()


@pytest.mark.asyncio
async def test_remote_monitor_recovers_from_disconnect_and_terminal_404(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._RETRY_SECONDS", 0.0)

    reconnecting = _Instances(
        [
            (200, {"id": "peer01"}),
            OSError("tunnel dropped"),
            (200, {"done": True, "turns": {}, "result": "reconnected", "error": ""}),
        ],
        connected=("crew-a",),
    )
    manager = _Manager()
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=reconnecting), manager  # type: ignore[arg-type]
    )
    info = await service.spawn(
        task="review",
        parent_session="dashboard:chat-1",
        agent="",
        max_turns=0,
        cwd="/remote/project",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=True,
        memory_mode="temporary",
        batch_id="",
        batch_total=0,
    )
    await _settle_monitor(service, info.id)
    assert info.done is True
    assert info.error == ""
    assert manager.reported == [info]
    await service.close()

    missing = _Instances(
        [(200, {"id": "peer02"}), (404, {}), (404, {}), (404, {})],
        connected=("crew-a",),
    )
    missing_manager = _Manager()
    missing_service = RemoteSubagentService(
        SimpleNamespace(instances_manager=missing), missing_manager  # type: ignore[arg-type]
    )
    missing_info = await missing_service.spawn(
        task="review",
        parent_session="dashboard:chat-1",
        agent="",
        max_turns=0,
        cwd="/remote/project",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=True,
        memory_mode="temporary",
        batch_id="",
        batch_total=0,
    )
    await missing_service._monitors[missing_info.id]
    assert missing_info.done is True
    assert "no longer available" in missing_info.error
    assert missing_manager.reported == [missing_info]
    await missing_service.close()


@pytest.mark.asyncio
async def test_corrupt_persistence_is_ignored_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    state_path = tmp_path / "subagents" / "remote" / "abcdef12" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "abcdef12",
                "executor": "remote",
                "instance_id": "crew-a",
                "remote_id": "../peer",
            }
        ),
        encoding="utf-8",
    )
    malformed_path = tmp_path / "subagents" / "remote" / "abcdef13" / "state.json"
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "abcdef13",
                "executor": "remote",
                "instance_id": "crew-a",
                "remote_id": "peer02",
                "max_turns": {"not": "an integer"},
            }
        ),
        encoding="utf-8",
    )
    manager = _Manager()
    service = RemoteSubagentService(SimpleNamespace(instances_manager=None), manager)  # type: ignore[arg-type]

    await service.ensure_restored()

    assert manager.external_agents == []
    assert manager.reported == []
    assert service._monitors == {}
    await service.close()


@pytest.mark.asyncio
async def test_remote_terminal_enters_normal_batch_digest_reporter() -> None:
    manager = SubagentManager.__new__(SubagentManager)
    info = SubagentInfo(
        id="remote01",
        task="remote",
        done=True,
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
        batch_id="batch01",
        batch_total=2,
    )
    manager._external_agents = {info.id: info}
    manager._claim_finalize = lambda candidate: candidate is info  # type: ignore[method-assign]
    reporter = AsyncMock(return_value=True)
    manager._run_terminal_report = reporter  # type: ignore[method-assign]

    assert await manager.report_external(info) is True
    reporter.assert_awaited_once_with(
        info,
        source="Remote subagent",
        injection_timeout_reason="remote completion delivery timed out",
        mark_delivered_on_success=False,
        settle_digest=True,
    )


def _manager_with_remote(info: SubagentInfo) -> tuple[SubagentManager, AsyncMock]:
    manager = cast(Any, SubagentManager.__new__(SubagentManager))
    manager._agents = {}
    manager._external_agents = {info.id: info}
    manager._queue = []
    manager._undurable_in_dispatch = {}
    manager._shutting_down = False
    manager._teardown_store_fences = {}
    manager._teardown_store_sweeps = []
    manager._teardown_sweeps_owed = []
    manager._teardown_fence_lock = threading.Lock()
    manager._teardown_cancelled_ids = set()
    manager._followup_watchers = {}
    manager._followup_watcher_parents = {}
    manager._followup_watcher_infos = {}
    manager._report_owners = {}
    manager._batch_submitted = {}
    manager._batch_progress_ts = {}
    manager._published_depths = PublishedQueueDepths()
    manager._admission = SimpleNamespace(
        taskq_pending_ids_for_async=AsyncMock(return_value=[]),
        taskq_store=lambda: None,
        _drain_queue_impl=lambda: None,
    )
    canceller = AsyncMock(return_value=True)
    manager.bind_external_placement(
        SimpleNamespace(
            cancel=canceller,
            dismiss=AsyncMock(return_value=True),
            acknowledge_delivered=AsyncMock(),
        )
    )
    return cast(SubagentManager, manager), canceller


@pytest.mark.asyncio
async def test_stop_all_and_parent_teardown_cancel_remote_children() -> None:
    stop_info = SubagentInfo(
        id="remote01",
        task="remote",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    stop_manager, stop_canceller = _manager_with_remote(stop_info)

    assert await stop_manager.cancel_for_parent("dashboard:chat-1") == (1, 0)
    stop_canceller.assert_awaited_once_with(stop_info)
    assert stop_info.user_stopped is True

    teardown_info = SubagentInfo(
        id="remote02",
        task="remote",
        parent_session_key="dashboard:chat-2",
        executor="remote",
        instance_id="crew-b",
        remote_id="peer02",
    )
    teardown_manager, teardown_canceller = _manager_with_remote(teardown_info)

    selected = teardown_manager.snapshot_teardown_children("dashboard:chat-2")
    assert selected == ("remote02",)
    assert "remote02" in teardown_manager._teardown_cancelled_ids
    assert (
        await teardown_manager.cancel_for_teardown(
            selected,
            parent_session_key="dashboard:chat-2",
            verb="remove",
        )
        == 1
    )
    teardown_canceller.assert_awaited_once_with(teardown_info)


class _WorkspaceRequest:
    def __init__(self, *, user: str, app: str = "", internal_auth: bool = False) -> None:
        self.app = {"state": SimpleNamespace(owner_id="owner")}
        self.query = {"sha256": "a" * 64, "commit": "b" * 40}
        self.content = _Content(b"archive")
        self._values = {"user": user, "app": app, "internal_auth": internal_auth}

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)


@pytest.mark.asyncio
async def test_workspace_route_accepts_only_owner_cookie_requests(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for request in (
        _WorkspaceRequest(user="owner", internal_auth=True),
        _WorkspaceRequest(user="owner", app="notes"),
        _WorkspaceRequest(user="other"),
    ):
        response = await api_remote_workspace_upload(cast(Any, request))
        assert response.status == 403

    installed = tmp_path / "workspace"
    monkeypatch.setattr(
        "kiro_crew.dashboard.remote_workspaces.install_workspace",
        lambda _payload, _digest, _commit, **_kwargs: installed,
    )
    response = await api_remote_workspace_upload(cast(Any, _WorkspaceRequest(user="owner")))
    assert response.status == 200
    assert isinstance(response.body, (bytes, bytearray))
    payload = json.loads(response.body)
    assert payload["path"] == str(installed)
    assert "reused" not in payload


@pytest.mark.asyncio
async def test_delivered_remote_records_expire_at_configured_ttl(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    info = SubagentInfo(
        id="abcdef12",
        task="done",
        done=True,
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    RemoteSubagentService._persist_info(info, delivered=True)
    manager = _Manager()
    manager._result_ttl_secs = 0
    service = RemoteSubagentService(SimpleNamespace(instances_manager=None), manager)  # type: ignore[arg-type]

    await service.ensure_restored()

    assert manager.get("abcdef12") is None
    assert not (tmp_path / "subagents" / "remote" / "abcdef12").exists()
    await service.close()


def test_workspace_install_prunes_only_expired_marked_snapshots(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_workspaces._WORKSPACE_TTL_SECS", 1)
    first = _tar_payload([("one.txt", b"one", 0o644, "file")])
    first_path = install_workspace(
        first,
        hashlib.sha256(first).hexdigest(),
        "a" * 40,
        root=tmp_path,
    )
    old = time.time() - 10
    os.utime(first_path, (old, old))

    unmarked = tmp_path / ("f" * 24 + "-" + "0" * 8)
    unmarked.mkdir()
    os.utime(unmarked, (old, old))

    second = _tar_payload([("two.txt", b"two", 0o644, "file")])
    second_path = install_workspace(
        second,
        hashlib.sha256(second).hexdigest(),
        "b" * 40,
        root=tmp_path,
    )

    assert second_path.is_dir()
    assert not first_path.exists()
    assert unmarked.is_dir()


@pytest.mark.asyncio
async def test_restart_resumes_polling_an_unfinished_remote_mapping(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    original = SubagentInfo(
        id="abcdef12",
        task="review",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    RemoteSubagentService._persist_info(original, delivered=False)
    instances = _Instances(
        [(200, {"done": True, "result": "after-restart", "error": ""})],
        connected=("crew-a",),
    )
    manager = _Manager()
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), manager  # type: ignore[arg-type]
    )

    await service.ensure_restored()
    await _settle_monitor(service, "abcdef12")

    restored = manager.get("abcdef12")
    assert restored is not None and restored.done is True
    assert manager.reported == [restored]
    assert instances.calls == [("crew-a", "GET", "api/spawn/peer01", None)]
    await service.close()


@pytest.mark.asyncio
async def test_a_failed_terminal_write_publishes_nothing_and_retries(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist before publish: no completion reaches the parent, and ``done``
    stays False, until the terminal mapping is on disk; the monitor re-polls
    the peer and retries instead of logging the write failure and reporting."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._RETRY_SECONDS", 0.0)
    original = SubagentInfo(
        id="abcdef12",
        task="review",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    RemoteSubagentService._persist_info(original, delivered=False)
    instances = _Instances(
        [
            (200, {"done": True, "result": "first", "error": ""}),
            (200, {"done": True, "result": "second", "error": ""}),
        ],
        connected=("crew-a",),
    )
    manager = _Manager()
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), manager  # type: ignore[arg-type]
    )
    real_persist = RemoteSubagentService._persist_info.__func__
    seen: list[tuple[bool, bool, int]] = []

    def flaky(cls, info, *, delivered, done=None):
        committed = info.done if done is None else done
        seen.append((committed, info.done, len(manager.reported)))
        if committed and len([s for s in seen if s[0]]) == 1:
            raise OSError("disk full")
        return real_persist(cls, info, delivered=delivered, done=done)

    monkeypatch.setattr(RemoteSubagentService, "_persist_info", classmethod(flaky))

    await service.ensure_restored()
    await _settle_monitor(service, "abcdef12")

    restored = manager.get("abcdef12")
    assert restored is not None and restored.done is True
    # The failed commit happened with nothing published and done still False.
    first_commit = next(entry for entry in seen if entry[0])
    assert first_commit[1:] == (False, 0)
    assert manager.reported == [restored]
    assert len(instances.calls) == 2
    state = json.loads((tmp_path / "subagents" / "remote" / "abcdef12" / "state.json").read_text())
    assert state["done"] is True
    await service.close()


def _git_project(root) -> str:
    import subprocess

    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    run = lambda *argv: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *argv], check=True, capture_output=True, env=env
    )
    run("init", "-q")
    (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
    run("add", "app.py")
    run(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "init",
    )
    return run("rev-parse", "HEAD").stdout.decode().strip()


def test_build_project_archive_uses_the_staged_tarball(tmp_path) -> None:
    import hashlib

    from kiro_crew.dashboard.remote_subagents import RemoteSubagentService

    head = _git_project(tmp_path)

    payload, digest, commit = RemoteSubagentService._build_project_archive(tmp_path)

    assert payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert commit == head
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert any(name.endswith("app.py") for name in archive.getnames())


def _commit_all(root, *paths: str) -> None:
    import subprocess

    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    run = lambda *argv: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *argv], check=True, capture_output=True, env=env
    )
    run("add", "-f", *paths)
    run(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "more",
    )


def _names(payload: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        return set(archive.getnames())


def test_a_planted_fsmonitor_never_runs_on_the_gateway(tmp_path) -> None:
    """An agent-written ``core.fsmonitor`` must not execute while packaging."""
    import subprocess

    from kiro_crew.dashboard.remote_subagents import RemoteSubagentService

    _git_project(tmp_path)
    marker = tmp_path.parent / f"{tmp_path.name}-fsmonitor-ran"
    hook = tmp_path.parent / f"{tmp_path.name}-fsmonitor.sh"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.fsmonitor", str(hook)], check=True)
    # Dirty a tracked file too: a status refresh is where git would ask the monitor.
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")

    payload, _digest, _commit = RemoteSubagentService._build_project_archive(tmp_path)

    assert not marker.exists()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        member = archive.extractfile("app.py")
        assert member is not None and member.read() == b"print('changed')\n"


def test_a_symlinked_parent_cannot_pull_files_from_outside(tmp_path) -> None:
    """``cache/config.json`` is tracked, then ``cache`` becomes a link elsewhere."""
    from kiro_crew.dashboard.remote_subagents import RemoteSubagentService

    project = tmp_path / "project"
    project.mkdir()
    _git_project(project)
    (project / "cache").mkdir()
    (project / "cache" / "config.json").write_text("{}\n", encoding="utf-8")
    (project / "notes.txt").write_text("kept\n", encoding="utf-8")
    (project / "leaf-link").symlink_to(project / "notes.txt")
    _commit_all(project, "cache/config.json", "notes.txt", "leaf-link")

    outside = tmp_path / "credentials"
    outside.mkdir()
    (outside / "config.json").write_text("SECRET\n", encoding="utf-8")
    (project / "cache" / "config.json").unlink()
    (project / "cache").rmdir()
    (project / "cache").symlink_to(outside, target_is_directory=True)

    payload, _digest, _commit = RemoteSubagentService._build_project_archive(project)

    names = _names(payload)
    assert "cache/config.json" not in names
    assert "leaf-link" not in names  # a tracked symlink is never packaged
    assert {"app.py", "notes.txt"} <= names
    assert b"SECRET" not in gzip_decompress(payload)


def gzip_decompress(payload: bytes) -> bytes:
    import gzip

    return gzip.decompress(payload)


def test_tracked_credentials_and_protected_paths_are_left_out(tmp_path, monkeypatch) -> None:
    from kiro_crew.dashboard import remote_subagents
    from kiro_crew.dashboard.remote_subagents import RemoteSubagentService

    _git_project(tmp_path)
    (tmp_path / ".env").write_text("TOKEN=x\n", encoding="utf-8")
    (tmp_path / "fenced.txt").write_text("fenced\n", encoding="utf-8")
    _commit_all(tmp_path, ".env", "fenced.txt")
    fenced = str(tmp_path / "fenced.txt")

    import kiro_crew.security.paths as paths

    real = paths.is_sensitive_resolved_path
    monkeypatch.setattr(paths, "is_sensitive_resolved_path", lambda p: p == fenced or real(p))

    names = _names(RemoteSubagentService._build_project_archive(tmp_path)[0])
    assert ".env" not in names and "fenced.txt" not in names
    assert "app.py" in names

    monkeypatch.setattr(paths, "is_sensitive_resolved_path", lambda p: True)
    with pytest.raises(RemoteSubagentError) as raised:
        RemoteSubagentService._build_project_archive(tmp_path)
    assert raised.value.code == "remote_project_protected"
    assert remote_subagents._GIT_NEUTRALISED_CONFIG[0] == "core.fsmonitor="


@pytest.mark.asyncio
async def test_registered_upload_route_reaches_the_real_handler(monkeypatch) -> None:
    """The route the gateway REGISTERS must import the module where it lives.

    The handler tests above import ``api_remote_workspace_upload`` directly, so
    they stayed green while the registered route resolved it under
    ``dashboard.handlers`` and raised ``ModuleNotFoundError`` -- a bare 500 on
    every project-synced remote spawn, caught only by an end-to-end canary.
    """
    from aiohttp import web

    from kiro_crew.dashboard import remote_workspaces, server

    sentinel = web.json_response({"reached": True})

    async def _stub(_request: Any) -> web.Response:
        return sentinel

    monkeypatch.setattr(remote_workspaces, "api_remote_workspace_upload", _stub)
    app = web.Application()
    server._register_mcp_routes(app)
    route = next(
        r
        for r in app.router.routes()
        if r.method == "POST"
        and r.resource is not None
        and r.resource.canonical == "/api/remote-workspaces"
    )

    assert await route.handler(cast(Any, object())) is sentinel


def _spawn_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "task": "review the secret plan",
        "parent_session": "dashboard:chat-1",
        "agent": "",
        "max_turns": 0,
        "cwd": "/remote/project",
        "model": "",
        "reasoning_effort": "",
        "include_memory": False,
        "include_lessons": False,
        "include_project": True,
        "memory_mode": "persistent",
        "batch_id": "",
        "batch_total": 0,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.asyncio
async def test_unpersisted_mapping_cancels_the_peer_run_instead_of_publishing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    instances = _Instances([(200, {"id": "peer01"}), (200, {"cancelled": True})])
    manager = _Manager()
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), manager  # type: ignore[arg-type]
    )

    def disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(service, "_persist_info", disk_full)

    with pytest.raises(RemoteSubagentError) as caught:
        await service.spawn(**_spawn_kwargs())  # type: ignore[arg-type]

    assert caught.value.code == "remote_mapping_unpersisted"
    assert manager.external_agents == []
    assert [call[1:3] for call in instances.calls] == [
        ("POST", "api/spawn"),
        ("DELETE", "api/spawn/peer01"),
    ]
    assert not service._monitors
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "applied", "refused"),
    [
        # An older peer drops both fields without saying so.
        ({"memory_mode": "incognito", "approval_mode": "auto"}, None, True),
        ({"approval_mode": ""}, None, True),
        # A peer that echoes something weaker than what was sent.
        (
            {"memory_mode": "temporary", "approval_mode": "auto"},
            {"memory_mode": "incognito", "approval_floor": ""},
            True,
        ),
        ({"approval_mode": ""}, {"memory_mode": "persistent", "approval_floor": ""}, True),
        # Nothing to tighten: an older peer is fine.
        ({"approval_mode": "auto"}, None, False),
        # A peer that applied something stricter is fine.
        (
            {"memory_mode": "incognito", "approval_mode": "auto"},
            {"memory_mode": "temporary", "approval_floor": ""},
            False,
        ),
    ],
    ids=[
        "legacy-incognito",
        "legacy-floor",
        "weaker-mode",
        "dropped-floor",
        "legacy-nothing-to-tighten",
        "stricter-mode",
    ],
)
async def test_a_peer_that_does_not_confirm_the_tightening_is_refused(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    applied: dict[str, str] | None,
    refused: bool,
) -> None:
    """Version parity is major.minor only, so the peer's echo is the proof."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    accepted: dict[str, object] = {"id": "peer01"}
    if applied is not None:
        accepted["applied"] = applied
    instances = _Instances(
        [(200, accepted), (200, {"cancelled": True})],
        connected=("crew-a",),
        legacy_peer=True,
    )
    manager = _Manager()
    service = RemoteSubagentService(
        cast(Any, SimpleNamespace(instances_manager=instances)), cast(Any, manager)
    )
    monkeypatch.setattr(service, "_monitor", AsyncMock())

    if refused:
        with pytest.raises(RemoteSubagentError) as caught:
            await service.spawn(**_spawn_kwargs(**overrides))  # type: ignore[arg-type]
        assert caught.value.code == "remote_peer_unenforced"
        assert caught.value.status == 409
        assert [call[1:3] for call in instances.calls] == [
            ("POST", "api/spawn"),
            ("DELETE", "api/spawn/peer01"),
        ]
        assert manager.external_agents == []
        assert not (tmp_path / "subagents" / "remote").exists() or not any(
            (tmp_path / "subagents" / "remote").iterdir()
        )
    else:
        info = await service.spawn(**_spawn_kwargs(**overrides))  # type: ignore[arg-type]
        assert info.remote_id == "peer01"
        assert [call[1:3] for call in instances.calls] == [("POST", "api/spawn")]
    await service.close()


@pytest.mark.asyncio
async def test_privacy_mode_runs_leave_no_task_or_result_on_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    instances = _Instances(
        [(200, {"id": "peer01"}), (200, {"done": True, "result": "private answer"})]
    )
    manager = _Manager()
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), manager  # type: ignore[arg-type]
    )

    info = await service.spawn(**_spawn_kwargs(memory_mode="incognito"))  # type: ignore[arg-type]
    await _settle_monitor(service, info.id)

    run_dir = tmp_path / "subagents" / "remote" / info.id
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["task"] == ""
    assert state["remote_id"] == "peer01"
    assert not (run_dir / "result.txt").exists()
    assert info.result_path == ""
    assert "secret plan" not in (run_dir / "state.json").read_text(encoding="utf-8")
    await service.close()


@pytest.mark.asyncio
async def test_a_permanently_unreachable_peer_ends_the_shadow_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._RETRY_SECONDS", 0.0)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._UNREACHABLE_DEADLINE_SECONDS", 0.0)
    instances = _Instances([(200, {"id": "peer01"}), ConnectionError("gone")])
    manager = _Manager()
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), manager  # type: ignore[arg-type]
    )

    info = await service.spawn(**_spawn_kwargs())  # type: ignore[arg-type]
    await _settle_monitor(service, info.id)

    assert info.done is True
    assert "unreachable" in info.error
    assert manager.reported == [info]
    await service.close()


@pytest.mark.asyncio
async def test_a_digest_held_completion_is_not_acknowledged(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._POLL_SECONDS", 0.0)
    instances = _Instances([(200, {"id": "peer01"}), (200, {"done": True, "result": "ok"})])
    manager = _Manager()

    async def held(info: SubagentInfo) -> bool:
        # The gateway parked this wave member in the digest: the reporter
        # returns True, but nothing has reached the parent yet.
        info._digest_held = True
        manager.reported.append(info)
        return True

    manager.report_external = held  # type: ignore[method-assign]
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), manager  # type: ignore[arg-type]
    )

    info = await service.spawn(**_spawn_kwargs(batch_id="w1", batch_total=2))  # type: ignore[arg-type]
    await _settle_monitor(service, info.id)

    state = json.loads(
        (tmp_path / "subagents" / "remote" / info.id / "state.json").read_text(encoding="utf-8")
    )
    assert state["done"] is True
    assert state["delivered"] is False

    # The digest later reaches the parent; its settle acknowledges the run, so
    # a restart does not deliver the same result a second time.
    assert manager.external_placement is service
    await service.acknowledge_delivered([info.id, "notmine1"])
    state = json.loads(
        (tmp_path / "subagents" / "remote" / info.id / "state.json").read_text(encoding="utf-8")
    )
    assert state["delivered"] is True
    assert not (tmp_path / "subagents" / "remote" / "notmine1").exists()
    await service.close()


@pytest.mark.asyncio
async def test_the_service_binds_once_and_close_unbinds_every_duty(tmp_path, monkeypatch) -> None:
    """One reference carries cancel, dismiss and delivery acknowledgement.

    Closing drops all three together: before, ``close`` unbound the canceller
    and the dismisser but left the delivery listener pointing at a closed service.
    """
    from kiro_crew.subagent import ExternalPlacement

    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    manager = SubagentManager(sessions=SimpleNamespace(), ctx_builder=None)
    service = RemoteSubagentService(
        cast(Any, SimpleNamespace(instances_manager=_Instances([]))), manager
    )
    placement: ExternalPlacement = service
    assert cast(Any, manager)._external_placement is placement

    await service.close()

    assert cast(Any, manager)._external_placement is None
    info = SubagentInfo(
        id="remote01",
        task="t",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    manager.register_external(info)
    with pytest.raises(RuntimeError, match="unavailable"):
        await manager.cancel_external("remote01")


@pytest.mark.asyncio
async def test_settles_hand_only_delivered_external_ids_to_the_listener() -> None:
    from kiro_crew.subagent import SubagentDelivery

    info = SubagentInfo(
        id="remote01", task="t", parent_session_key="dashboard:chat-1", executor="remote"
    )
    manager, _canceller = _manager_with_remote(info)
    raw = cast(Any, manager)
    raw._waves = SimpleNamespace(
        settle_queued_delivery_impl=AsyncMock(), _settle_digest_holds_impl=AsyncMock()
    )
    listener = cast(Any, manager)._external_placement.acknowledge_delivered
    deliveries = [
        SubagentDelivery("remote01", 1.0, 0.0),
        SubagentDelivery("local001", 1.0, 0.0),
        SubagentDelivery("remote01", 0.0, 0.0, report_owed=True),
    ]

    await manager.settle_queued_delivery(deliveries)
    listener.assert_awaited_once_with(["remote01"])

    listener.reset_mock()
    holder = SubagentInfo(id="member01", task="t", parent_session_key="dashboard:chat-1")
    holder._digest_settle_deliveries = [SubagentDelivery("remote01", 1.0, 0.0)]
    await manager._settle_digest_holds(holder)
    listener.assert_awaited_once_with(["remote01"])

    # A digest whose injection the gateway gave up on did not reach the parent.
    listener.reset_mock()
    holder._digest_settle_deliveries = [SubagentDelivery("remote01", 1.0, 0.0)]
    holder._report_undelivered = True
    await manager._settle_digest_holds(holder)
    listener.assert_not_awaited()

    # A failing acknowledgement never fails the settle itself.
    listener.side_effect = RuntimeError("disk full")
    await manager.settle_queued_delivery([SubagentDelivery("remote01", 1.0, 0.0)])


@pytest.mark.asyncio
async def test_version_skew_is_a_typed_refusal_not_a_bare_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unreachable(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("version read failed")

    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents._ensure_version_parity", unreachable)
    instances = _Instances([])
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=instances), _Manager()  # type: ignore[arg-type]
    )

    with pytest.raises(RemoteSubagentError) as caught:
        await service.spawn(**_spawn_kwargs())  # type: ignore[arg-type]

    assert caught.value.status == 409
    assert instances.calls == []
    await service.close()


@pytest.mark.asyncio
async def test_a_project_that_cannot_be_packaged_is_a_typed_refusal(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kiro_crew.cloud.aws import AWSError

    state = SimpleNamespace(
        instances_manager=_Instances([], connected=("crew-a",)),
        get_slot=lambda _name: SimpleNamespace(project=str(tmp_path)),
    )
    service = RemoteSubagentService(state, _Manager())  # type: ignore[arg-type]

    def not_git(_path: Path) -> tuple[bytes, str, str]:
        raise AWSError("not a git checkout")

    monkeypatch.setattr(service, "_build_project_archive", not_git)

    with pytest.raises(RemoteSubagentError) as caught:
        await service.spawn(**_spawn_kwargs(cwd=""))  # type: ignore[arg-type]

    assert caught.value.code == "remote_project_snapshot_failed"
    assert caught.value.status == 400
    await service.close()


def test_a_swallowed_injection_failure_is_not_read_as_delivered() -> None:
    from kiro_crew.dashboard.remote_subagents import _reached_parent

    info = SubagentInfo(id="remote01", task="t", parent_session_key="cron:job-1")
    assert _reached_parent(info) is True
    # A channel or cron parent whose injection attempts all failed: the
    # reporter returned True, but the parent never saw the result.
    info._report_undelivered = True
    assert _reached_parent(info) is False


@pytest.mark.asyncio
async def test_a_failed_redelivery_does_not_strand_live_runs_on_restore(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    finished = SubagentInfo(
        id="aaaaaaa1",
        task="t",
        parent_session_key="dashboard:chat-1",
        done=True,
        result="r",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
    )
    live = SubagentInfo(
        id="bbbbbbb2",
        task="t",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer02",
    )
    RemoteSubagentService._persist_info(finished, delivered=False)
    RemoteSubagentService._persist_info(live, delivered=False)

    manager = _Manager()

    async def broken(_info: SubagentInfo) -> bool:
        raise OSError("disk full")

    manager.report_external = broken  # type: ignore[method-assign]
    service = RemoteSubagentService(SimpleNamespace(instances_manager=None), manager)  # type: ignore[arg-type]
    started: list[str] = []

    async def monitor(info: SubagentInfo) -> None:
        started.append(info.id)

    monkeypatch.setattr(service, "_monitor", monitor)
    await service.ensure_restored()
    await asyncio.sleep(0)

    assert started == ["bbbbbbb2"]
    state = json.loads(
        (tmp_path / "subagents" / "remote" / "aaaaaaa1" / "state.json").read_text(encoding="utf-8")
    )
    assert state["delivered"] is False  # retried on the next start
    await service.close()


@pytest.mark.asyncio
async def test_one_bad_record_does_not_fail_the_whole_restore(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record with an empty crew binding, or one the manager refuses, is
    skipped; the other runs still restore and the restore itself succeeds,
    so the spawn routes that await it do not error."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    runs = {
        run_id: SubagentInfo(
            id=run_id,
            task="t",
            parent_session_key="dashboard:chat-1",
            executor="remote",
            instance_id="crew-a",
            remote_id=f"peer0{index}",
        )
        for index, run_id in enumerate(("aaaaaaa1", "bbbbbbb2", "ccccccc3"), start=1)
    }
    for info in runs.values():
        RemoteSubagentService._persist_info(info, delivered=False)
    blank = tmp_path / "subagents" / "remote" / "aaaaaaa1" / "state.json"
    state = json.loads(blank.read_text(encoding="utf-8"))
    state["instance_id"] = ""
    blank.write_text(json.dumps(state), encoding="utf-8")

    manager = _Manager()
    accept = manager.register_external

    def picky(info: SubagentInfo) -> None:
        if info.id == "bbbbbbb2":
            raise ValueError("subagent id already registered: bbbbbbb2")
        accept(info)

    manager.register_external = picky  # type: ignore[method-assign]
    service = RemoteSubagentService(SimpleNamespace(instances_manager=None), manager)  # type: ignore[arg-type]
    started: list[str] = []

    async def monitor(info: SubagentInfo) -> None:
        started.append(info.id)

    monkeypatch.setattr(service, "_monitor", monitor)
    await service.ensure_restored()
    await asyncio.sleep(0)

    assert [info.id for info in manager.external_agents] == ["ccccccc3"]
    assert started == ["ccccccc3"]
    await service.close()


@pytest.mark.asyncio
async def test_a_failed_restore_is_retried_on_the_next_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    info = SubagentInfo(
        id="ddddddd4",
        task="t",
        parent_session_key="dashboard:chat-1",
        done=True,
        result="r",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer04",
    )
    RemoteSubagentService._persist_info(info, delivered=True)
    real_load = RemoteSubagentService._load_records.__func__
    calls: list[int] = []

    def flaky(cls):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return real_load(cls)

    monkeypatch.setattr(RemoteSubagentService, "_load_records", classmethod(flaky))
    manager = _Manager()
    service = RemoteSubagentService(SimpleNamespace(instances_manager=None), manager)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        await service.ensure_restored()
    await service.ensure_restored()

    assert len(calls) == 2
    assert [restored.id for restored in manager.external_agents] == ["ddddddd4"]
    await service.close()


@pytest.mark.asyncio
async def test_automatic_placement_skips_a_version_skewed_crew(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    instances = _Instances([(200, {"id": "peer01"})], connected=("crew-a", "crew-b"))

    async def version(instance_id: str) -> tuple[bool, str]:
        # crew-a is the least-loaded (alphabetical tie winner) but a feature
        # release ahead; crew-b matches.
        return True, ("99.0.0" if instance_id == "crew-a" else kiro_crew.__version__)

    instances.peer_version = version  # type: ignore[method-assign]
    manager = _Manager()
    service = RemoteSubagentService(SimpleNamespace(instances_manager=instances), manager)  # type: ignore[arg-type]
    monkeypatch.setattr(service, "_monitor", AsyncMock())

    info = await service.spawn(**_spawn_kwargs())  # type: ignore[arg-type]

    assert info.instance_id == "crew-b"
    assert [call[0] for call in instances.calls] == ["crew-b"]

    # An explicitly requested skewed crew is still a typed refusal.
    with pytest.raises(RemoteSubagentError) as caught:
        await service.spawn(**_spawn_kwargs(instance_id="crew-a"))  # type: ignore[arg-type]
    assert caught.value.status == 409
    await service.close()


@pytest.mark.asyncio
async def test_dismissing_a_finished_remote_run_waits_for_its_delivery() -> None:
    info = SubagentInfo(
        id="remote01",
        task="t",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        done=True,
    )
    manager, _canceller = _manager_with_remote(info)

    info._delivery_queued = True
    assert await manager.settle_before_delete("remote01") == "pending"
    assert manager.get("remote01") is info

    info._delivery_queued = False
    assert await manager.settle_before_delete("remote01") == "delivered"
    # Gone from the inventory, so the spawn list does not show it.
    assert cast(Any, manager)._external_agents == {}


@pytest.mark.asyncio
async def test_a_dismissed_remote_run_stays_dismissed_across_a_restart(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dismissal is committed to the mapping before the pop, so a restart
    does not restore it; a failed commit leaves the run listed and retryable."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    info = SubagentInfo(
        id="abcdef12",
        task="t",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
        done=True,
    )
    RemoteSubagentService._persist_info(info, delivered=True)
    manager, _canceller = _manager_with_remote(info)
    service = RemoteSubagentService(
        SimpleNamespace(instances_manager=_Instances([], connected=("crew-a",))),
        manager,  # type: ignore[arg-type]
    )
    real_persist = RemoteSubagentService._persist_info.__func__

    def unwritable(cls, *args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(RemoteSubagentService, "_persist_info", classmethod(unwritable))
    assert await manager.settle_before_delete("abcdef12") == "pending"
    assert manager.get("abcdef12") is info

    monkeypatch.setattr(RemoteSubagentService, "_persist_info", classmethod(real_persist))
    assert await manager.settle_before_delete("abcdef12") == "delivered"
    assert manager.get("abcdef12") is None
    assert RemoteSubagentService._load_records() == []
    await service.close()


def test_a_late_acknowledgement_cannot_undo_a_committed_dismissal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivery acknowledgement already in flight when the operator dismisses
    the run must not write ``dismissed: false`` back: mapping writes for one run
    are serialized and a committed dismissal is sticky."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    info = SubagentInfo(
        id="abcdef12",
        task="t",
        parent_session_key="dashboard:chat-1",
        executor="remote",
        instance_id="crew-a",
        remote_id="peer01",
        done=True,
    )
    RemoteSubagentService._persist_info(info, delivered=False)

    # Sequential: an acknowledgement after the dismissal keeps it.
    RemoteSubagentService._persist_info(info, delivered=True, dismissed=True)
    RemoteSubagentService._persist_info(info, delivered=True)
    assert RemoteSubagentService._load_records() == []

    # Interleaved: the acknowledgement has read the mapping (not yet dismissed)
    # and is about to write when the dismissal arrives on another thread.
    # Start from a fresh, undismissed mapping (the dismissal above is sticky).
    (tmp_path / "subagents" / "remote" / "abcdef12" / "state.json").unlink()
    RemoteSubagentService._persist_info(info, delivered=False)
    assert len(RemoteSubagentService._load_records()) == 1
    real_stored = RemoteSubagentService._stored_dismissed
    ack_has_read = threading.Event()
    release_ack = threading.Event()

    def paused_read(run_dir: Path) -> bool:
        value = real_stored(run_dir)
        if threading.current_thread().name == "ack":
            ack_has_read.set()
            release_ack.wait(timeout=5)
        return value

    monkeypatch.setattr(RemoteSubagentService, "_stored_dismissed", staticmethod(paused_read))
    ack = threading.Thread(
        target=RemoteSubagentService._persist_info,
        args=(info,),
        kwargs={"delivered": True},
        name="ack",
    )
    ack.start()
    assert ack_has_read.wait(timeout=5)
    dismiss = threading.Thread(
        target=RemoteSubagentService._persist_info,
        args=(info,),
        kwargs={"delivered": True, "dismissed": True},
        name="dismiss",
    )
    dismiss.start()
    dismiss.join(timeout=0.3)  # blocked behind the acknowledgement's lock
    release_ack.set()
    ack.join(timeout=5)
    dismiss.join(timeout=5)
    assert not ack.is_alive() and not dismiss.is_alive()

    state = json.loads(
        (tmp_path / "subagents" / "remote" / "abcdef12" / "state.json").read_text(encoding="utf-8")
    )
    assert state["dismissed"] is True
    assert RemoteSubagentService._load_records() == []


@pytest.mark.parametrize("record", ("abcdef12/state.json", "remote/abcdef12/state.json"))
def test_remote_run_mappings_are_write_protected_like_run_records(record: str) -> None:
    """A mapping names the parent session a restart restores, which decides who
    may read or cancel the run and where its result lands. It lives inside the
    ``subagents`` registry, so the same seal holds it: agent file edits are
    refused, reads stay open."""
    from kiro_crew import security

    path = os.path.expanduser(f"~/.kiro/crew/subagents/{record}")
    assert security.is_sensitive_write_path(path) is True
    assert security.is_sensitive_path(path) is False


def test_remote_records_live_inside_the_sealed_run_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The records root must stay under ``subagents/``: a sibling leaf would need
    its own seal, and an unsealed one lets an agent forge a run's parent session."""
    from kiro_crew.dashboard import remote_subagents

    monkeypatch.setattr(remote_subagents, "data_home", lambda: tmp_path)
    root = remote_subagents._records_root()
    assert root.parent == tmp_path / "subagents"
    assert not re.fullmatch(r"[a-f0-9]+", root.name), "the root must never look like a run id"


def test_local_registry_walks_skip_the_remote_records_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every local walk of ``subagents/`` must pass over ``remote/``: it has no
    ``state.json`` of its own, so it is neither an orphan to recover, a panel
    record, a legacy queue row, nor a tombstoned folder to prune."""
    from kiro_crew import subagent_persistence
    from kiro_crew.taskq.migrate import legacy_subagent_records

    registry = tmp_path / "subagents"
    record = registry / "remote" / "abcdef12"
    record.mkdir(parents=True)
    (record / "state.json").write_text(
        json.dumps({"version": 1, "id": "abcdef12", "executor": "remote"}), encoding="utf-8"
    )
    monkeypatch.setattr(subagent_persistence, "_SUBAGENTS_DIR", registry)

    assert subagent_persistence.list_orphans() == []
    assert subagent_persistence.prune_stale_tombstones(max_age_days=0, delivered_ttl_secs=0) == 0
    assert subagent_persistence.read_panel_records(keep=10, max_age_secs=86400).records == []
    assert legacy_subagent_records(registry, now=time.time()) == []
    assert (record / "state.json").is_file()


def _raising_sessions() -> Any:
    class _Broken:
        def get_approval_policy(self, _key: str) -> str:
            raise RuntimeError("session store unreadable")

        def has_session(self, _key: str) -> bool | None:
            raise RuntimeError("session store unreadable")

    return _Broken()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configure", "approval_mode", "expected"),
    [
        (lambda m: None, "", "interactive"),
        (lambda m: setattr(m, "_sessions", _Sessions(policy="auto")), "", ""),
        (lambda m: None, "auto", ""),
        (lambda m: setattr(m, "_is_yolo", lambda: True), "", ""),
        # Global approval_mode=auto counts only for a parent that is gone ...
        (
            lambda m: (
                setattr(m, "_global_approval_mode", "auto"),
                setattr(m, "_sessions", _Sessions(present=False)),
            ),
            "",
            "",
        ),
        # ... never for a live parent that is intentionally not on auto.
        (lambda m: setattr(m, "_global_approval_mode", "auto"), "", "interactive"),
        # An unreadable posture must not loosen the child.
        (lambda m: setattr(m, "_sessions", _raising_sessions()), "auto", "interactive"),
    ],
    ids=[
        "normal-parent",
        "auto-parent",
        "sdk-auto",
        "hub-yolo",
        "global-auto-parent-gone",
        "global-auto-parent-alive",
        "store-unreadable",
    ],
)
async def test_remote_spawn_forwards_the_parents_approval_floor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    configure: Callable[[Any], object],
    approval_mode: str,
    expected: str,
) -> None:
    """The crew gets the posture the same child would have here, as a floor."""
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    instances = _Instances([(200, {"id": "peer01"})], connected=("crew-a",))
    manager = _Manager()
    configure(manager)
    service = RemoteSubagentService(
        cast(Any, SimpleNamespace(instances_manager=instances)), cast(Any, manager)
    )
    monkeypatch.setattr(service, "_monitor", AsyncMock())

    info = await service.spawn(
        task="review",
        parent_session="dashboard:chat-1",
        agent="",
        max_turns=0,
        cwd="/remote/project",
        model="",
        reasoning_effort="",
        include_memory=False,
        include_lessons=False,
        include_project=True,
        memory_mode="persistent",
        batch_id="",
        batch_total=0,
        approval_mode=approval_mode,
    )

    body = instances.calls[0][3]
    assert isinstance(body, dict)
    if expected:
        assert body["approval_floor"] == expected
    else:
        assert "approval_floor" not in body
    assert "approval_mode" not in body
    assert info.approval_floor == expected
    state = json.loads(
        (tmp_path / "subagents" / "remote" / info.id / "state.json").read_text(encoding="utf-8")
    )
    assert state["approval_floor"] == expected
    await service.close()


@pytest.mark.asyncio
async def test_restored_remote_run_keeps_its_approval_floor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.dashboard.remote_subagents.data_home", lambda: tmp_path)
    for run_id, floor in (("abcdef12", "interactive"), ("abcdef13", "bogus")):
        RemoteSubagentService._persist_info(
            SubagentInfo(
                id=run_id,
                task="review",
                parent_session_key="dashboard:chat-1",
                done=True,
                executor="remote",
                instance_id="crew-a",
                remote_id="peer01",
                approval_floor=floor,
            ),
            delivered=True,
        )
    manager = _Manager()
    service = RemoteSubagentService(
        cast(Any, SimpleNamespace(instances_manager=None)), cast(Any, manager)
    )
    await service.ensure_restored()

    kept = manager.get("abcdef12")
    unknown = manager.get("abcdef13")
    assert kept is not None and kept.approval_floor == "interactive"
    # Only the one known value survives a reload; anything else reads as none.
    assert unknown is not None and unknown.approval_floor == ""
    await service.close()
