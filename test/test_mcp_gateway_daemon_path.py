"""gatewayd's own PATH must carry the MCP launcher dirs, not just ``~/.local/bin``.

The field failure this pins: the dashboard reported an MCP server as "failed to
start in this session" with ``JSON-RPC error: -32000: backend gone: exit
rc=127`` for ``slack-mcp``, ``talos-web-mcp`` and ``local-chorus-mcp``, while
every other pooled server on the same daemon came up fine.

rc=127 is ``/bin/sh``'s "command not found", and the split between the failing
and working servers is what names the cause. A server whose spec command is an
absolute binary (``~/.toolbox/bin/builder-mcp``) needs no PATH lookup at all.
The three failures are the ones whose command is a thin wrapper script that
exec's a BARE tool name::

    #!/bin/sh
    exec <tool> <subcommand> local-chorus-mcp

``GatewayManager._spawn_once`` built the daemon's env by prepending exactly ONE
hand-picked directory (``~/.local/bin``), so a gateway launched from the desktop
app handed gatewayd a PATH without the dir holding that tool. Every such wrapper
then failed its exec, the backend exited 127 before writing a byte of protocol,
and the only thing the session saw was "backend gone" -- no mention of PATH, of
the wrapper, or of the tool it could not find.

Three properties are pinned:

* the daemon's PATH is built by :func:`kiro_crew.env.augmented_path` -- the same
  helper the kiro-cli spawn path uses -- so the launcher dirs cannot drift apart
  between the two spawn sites,
* that build runs OFF the event loop, because on a cold cache it globs every
  version-manager root for Node bin dirs, and
* the mechanism itself: a bare-name wrapper really does exit 127 under the old
  one-directory PATH and really does start under the fixed one. Without this last
  test the first is just an assertion about a string.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kiro_crew.env import augmented_path
from kiro_crew.mcp_gateway import manager as manager_mod
from kiro_crew.mcp_gateway.manager import GatewayManager, GatewaySpec


class _SpawnCaptured(RuntimeError):
    """Raised from the fake spawn so the test stops at the env we came for."""


def _captured_daemon_env(monkeypatch, tmp_path: Path, inherited_path: str) -> dict[str, str]:
    """Run ``_spawn_once`` far enough to capture the env it would spawn with."""
    captured: dict[str, str] = {}

    async def _fake_spawn(*_argv, env=None, **_kwargs):
        captured.update(env or {})
        raise _SpawnCaptured

    monkeypatch.setenv("PATH", inherited_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    manager = GatewayManager(GatewaySpec(socket_path=tmp_path / "gateway.sock"))
    monkeypatch.setattr(manager, "_gatewayd_log_path", lambda: tmp_path / "gatewayd.stdout")
    monkeypatch.setattr(manager, "_credential_watch_paths", lambda: [])

    with pytest.raises(_SpawnCaptured):
        asyncio.run(manager._spawn_once())
    return captured


# ------------------------------------------------- the daemon's PATH is augmented


def test_the_daemon_inherits_the_augmented_launcher_path(monkeypatch, tmp_path) -> None:
    """A minimal inherited PATH is extended with EVERY well-known launcher dir.

    Asserting equality against ``augmented_path`` rather than spot-checking one
    directory is the point: it makes this spawn site and the kiro-cli spawn site
    share one definition, so a dir added to ``_EXTRA_PATH_DIRS`` for one of them
    can never silently skip the other.
    """
    minimal = os.pathsep.join(["/usr/bin", "/bin"])
    env = _captured_daemon_env(monkeypatch, tmp_path, minimal)

    assert env["PATH"] == augmented_path(minimal)


def test_the_toolbox_bin_dir_reaches_the_daemon(monkeypatch, tmp_path) -> None:
    """The specific directory the field failure was missing.

    ``~/.toolbox/bin`` holds the tool that every launcher wrapper exec's by bare
    name. This is the assertion that goes red if someone narrows the
    augmentation back to a hand-picked subset.
    """
    minimal = os.pathsep.join(["/usr/bin", "/bin"])
    env = _captured_daemon_env(monkeypatch, tmp_path, minimal)

    toolbox_bin = str(Path(os.path.expanduser("~")) / ".toolbox" / "bin")
    assert toolbox_bin in env["PATH"].split(os.pathsep)


def test_an_operator_path_entry_is_not_dropped(monkeypatch, tmp_path) -> None:
    """Augmenting must ADD dirs, never replace the inherited ones -- an operator
    who put a launcher on the gateway's own PATH must still be able to reach it.
    """
    inherited = os.pathsep.join(["/opt/site/bin", "/usr/bin", "/bin"])
    env = _captured_daemon_env(monkeypatch, tmp_path, inherited)

    parts = env["PATH"].split(os.pathsep)
    assert "/opt/site/bin" in parts


def test_the_path_is_built_off_the_event_loop(monkeypatch, tmp_path) -> None:
    """The PATH build must not run on the loop.

    On a cold cache ``augmented_path`` globs every version-manager root for Node
    bin dirs, which is filesystem work with no ceiling on a slow or remote home.
    This method already offloads its other blocking calls, so a regression that
    drops the ``to_thread`` would be silent -- the value would be identical and
    every other test here would still pass.
    """
    loop_thread: dict[str, int] = {}
    build_thread: dict[str, int] = {}

    def _spy(base: str) -> str:
        build_thread["ident"] = threading.get_ident()
        return augmented_path(base)

    monkeypatch.setattr(manager_mod, "augmented_path", _spy)

    async def _record_loop_thread() -> None:
        loop_thread["ident"] = threading.get_ident()

    captured: dict[str, str] = {}

    async def _fake_spawn(*_argv, env=None, **_kwargs):
        captured.update(env or {})
        raise _SpawnCaptured

    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_spawn)

    mgr = GatewayManager(GatewaySpec(socket_path=tmp_path / "gateway.sock"))
    monkeypatch.setattr(mgr, "_gatewayd_log_path", lambda: tmp_path / "gatewayd.stdout")
    monkeypatch.setattr(mgr, "_credential_watch_paths", lambda: [])

    async def _drive() -> None:
        await _record_loop_thread()
        with pytest.raises(_SpawnCaptured):
            await mgr._spawn_once()

    asyncio.run(_drive())

    assert build_thread.get("ident") is not None, "augmented_path was never called"
    assert build_thread["ident"] != loop_thread["ident"]
    # ...and the value the worker produced is still what the child receives.
    assert captured["PATH"] == augmented_path(os.pathsep.join(["/usr/bin", "/bin"]))


# ------------------------------------------- the mechanism the rc=127 came from


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX #! wrapper + /bin/sh exec")
def test_a_bare_name_wrapper_exits_127_without_the_launcher_dir(tmp_path) -> None:
    """Reproduce the failure and its fix with a real wrapper and a real exec.

    ``tool_dir`` stands in for the launcher tool's own bin dir and ``wrapper``
    for the one-line script that exec's it by bare name. Under a PATH missing
    ``tool_dir`` the exec fails and the process exits 127 -- exactly the code
    the daemon reported as "backend gone". Adding the directory is the whole
    difference.
    """
    tool_dir = tmp_path / "toolbox-bin"
    tool_dir.mkdir()
    tool = tool_dir / "faux-tool"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)

    wrapper = tmp_path / "faux-mcp-server"
    wrapper.write_text('#!/bin/sh\nexec faux-tool start-server x "$@"\n')
    wrapper.chmod(0o755)

    without = subprocess.run(
        [str(wrapper)],
        env={"PATH": os.pathsep.join(["/usr/bin", "/bin"])},
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    assert without.returncode == 127

    with_dir = subprocess.run(
        [str(wrapper)],
        env={"PATH": os.pathsep.join([str(tool_dir), "/usr/bin", "/bin"])},
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    assert with_dir.returncode == 0
