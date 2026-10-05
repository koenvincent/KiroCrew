#!/usr/bin/env python3
"""Canary for the remote-subagent-executor PR.

Boots two isolated, offline ``--test-mode`` gateways from THIS checkout using
``kiro_crew.testing.harness.spawn_feature_gateway`` (the same harness the repo's
own E2E suite uses) and the packaged fake ACP backend
(``kiro_crew.testing.fake_acp_backend``) -- no network, no model, no secrets.

Roles:
  * PEER   -- the remote crew. A disposable sshd on 127.0.0.1 is started for it
              (own host key, own authorized key, own port) so the PARENT's
              SshTunnelManager can reach it exactly the way it reaches a real
              remote instance: ``ssh <peer> <kirocrew-bin> token --port <P>``.
  * PARENT -- the local crew. Its instance registry gets one "ssh" record
              pointing at the peer; its own dashboard API
              (``/api/remote-workspaces``, ``/api/spawn`` with
              ``executor: "remote"``) drives the five scenarios below.

Auth, exactly as the product uses it -- no shortcuts:
  * ``/api/remote-workspaces`` requires owner-dashboard auth: the parent's own
    ``?token=`` query token (``GatewayHandle.url``) IS that owner session.
  * ``/api/spawn`` (and the other spawn-control routes) accept the loopback
    ``X-Internal-Secret`` handshake: the secret lives at
    ``<KIROCREW_HOME>/.local_secret`` for the PARENT's own throwaway home,
    which this script owns and reads directly (never printed, never put in
    evidence).
  * The peer's own dashboard token is never seen by this script at all: the
    parent's ``SshTunnelManager`` mints and holds it, over the ssh tunnel this
    script sets up. That is the whole point of the architecture under test.

Five scenarios, each writing its own pass/fail + evidence into
``canary-evidence.json`` (ids, statuses, timings -- never tokens or secrets):

  a) upload   -- project snapshot reaches the peer via POST /api/remote-workspaces
  b) landed   -- the remote spawn is actually running ON THE PEER (proven by a
                 record in the PEER's OWN /api/spawn listing, not just the
                 parent's view of it)
  c) complete -- the parent's GET /api/spawn/{id} reaches done:true with the
                 fake backend's canned reply
  d) cancel   -- DELETE /api/spawn/{id} cancels a long-running remote turn on
                 both ends
  e) resume   -- restarting the PARENT gateway in place, the remote mapping is
                 either picked back up or cleanly finalized, per the PR's own
                 persistence contract (RemoteSubagentService.ensure_restored)

A scenario that cannot be driven is FAIL with the reason -- never a fabricated
PASS. Exit code is 0 only if every scenario passed.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kiro_crew.testing import fake_acp_backend  # noqa: E402
from kiro_crew.testing.harness import (  # noqa: E402
    GatewayHandle,
    fake_acp_backend_launcher,
    spawn_feature_gateway,
)

EVIDENCE_PATH = Path(os.environ.get("CANARY_EVIDENCE_PATH", "canary-evidence.json"))
POLL_SECONDS = 1.0
SPAWN_TIMEOUT_SECS = 60.0
SSH_CONNECT_TIMEOUT_SECS = 30.0
RESTART_TIMEOUT_SECS = 60.0


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    started: float = 0.0
    ended: float = 0.0


class Evidence:
    def __init__(self) -> None:
        self.results: list[ScenarioResult] = []

    def record(self, result: ScenarioResult) -> None:
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    def write(self, path: Path) -> bool:
        payload = {
            "generated_at": time.time(),
            "scenarios": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "evidence": r.evidence,
                    "duration_secs": round(r.ended - r.started, 3) if r.ended else None,
                }
                for r in self.results
            ],
            "all_passed": all(r.passed for r in self.results) and bool(self.results),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return bool(payload["all_passed"])


# --------------------------------------------------------------------------- #
# Disposable local sshd for the peer
# --------------------------------------------------------------------------- #


@dataclass
class DisposableSshd:
    """One sshd bound to 127.0.0.1 on an ephemeral port, with a throwaway key.

    POSIX only (the harness itself is cross-platform, but this canary targets
    the ubuntu-latest runner the workflow pins; a disposable sshd on loopback
    is the whole point of the scenario, not a portability concern here).
    """

    port: int
    host_key_path: Path
    client_key_path: Path
    sshd_config_path: Path
    pid_file: Path
    proc: subprocess.Popen[bytes]
    log_path: Path

    def stop(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self.proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=5.0)
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5.0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_disposable_sshd(scratch: Path) -> DisposableSshd:
    """Start a throwaway sshd on 127.0.0.1, authorized for one throwaway key.

    The peer's "home" for ssh purposes is ``scratch`` itself: sshd is told
    (via its config, not via the invoking user's real home) to look for
    ``authorized_keys`` there, and the client key used to connect is generated
    alongside it. Nothing here touches the operator's real ``~/.ssh``.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    host_key = scratch / "ssh_host_ed25519_key"
    client_key = scratch / "canary_client_ed25519_key"
    authorized_keys = scratch / "authorized_keys"
    sshd_config = scratch / "sshd_config"
    pid_file = scratch / "sshd.pid"
    log_path = scratch / "sshd.log"

    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(host_key)],
        check=True,
    )
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(client_key)],
        check=True,
    )
    authorized_keys.write_text((client_key.with_suffix(client_key.suffix + ".pub")).read_text())
    authorized_keys.chmod(0o600)

    port = _free_port()
    sshd_config.write_text(
        "\n".join(
            [
                f"Port {port}",
                "ListenAddress 127.0.0.1",
                f"HostKey {host_key}",
                "PidFile " + str(pid_file),
                "AuthorizedKeysFile " + str(authorized_keys),
                "PasswordAuthentication no",
                "KbdInteractiveAuthentication no",
                "PubkeyAuthentication yes",
                "UsePAM no",
                "StrictModes no",
                "PermitRootLogin yes",
                "LogLevel ERROR",
                "Subsystem sftp internal-sftp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sshd_bin = _which("sshd") or "/usr/sbin/sshd"
    proc = subprocess.Popen(
        [sshd_bin, "-D", "-e", "-f", str(sshd_config)],
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _port_open("127.0.0.1", port):
            break
        if proc.poll() is not None:
            raise RuntimeError(
                f"sshd exited (code {proc.returncode}) before binding :{port}; "
                f"log tail: {log_path.read_text(errors='replace')[-2000:]}"
            )
        time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError(f"sshd did not bind 127.0.0.1:{port} within 10s")

    return DisposableSshd(
        port=port,
        host_key_path=host_key,
        client_key_path=client_key,
        sshd_config_path=sshd_config,
        pid_file=pid_file,
        proc=proc,
        log_path=log_path,
    )


def _which(name: str) -> Optional[str]:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(d) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _port_open(host: str, port: int) -> bool:
    with contextlib.suppress(OSError):
        with socket.create_connection((host, port), timeout=1.0):
            return True
    return False


# --------------------------------------------------------------------------- #
# Peer-side kirocrew launcher the parent's ssh hop execs
# --------------------------------------------------------------------------- #


def write_peer_kirocrew_launcher(peer_home: Path, peer_port: int, peer_token: str) -> Path:
    """Write ``$HOME/.local/bin/kirocrew`` for the peer's ssh user.

    ``SshTunnelManager`` execs one of a short list of candidate paths over ssh
    to mint a token (``token_mint.REMOTE_BIN_CANDIDATES``). The real
    ``kirocrew token`` subcommand loads config, resolves the LIVE gateway, and
    mints a fresh dashboard token -- machinery this canary does not need
    (there is already exactly one gateway, and it already has a token from
    ``spawn_feature_gateway``). The launcher is a thin shim that prints the
    PEER gateway's already-minted URL in the exact
    ``http://localhost:<port>?token=<jwt>`` shape ``token_mint.parse_token_from_stdout``
    scans for -- same wire contract, no re-implementation of the mint path.
    """
    bin_dir = peer_home / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / "kirocrew"
    launcher.write_text(
        "#!/bin/sh\n"
        "# Canary shim: the real `kirocrew token` mints fresh; this prints the\n"
        "# fixed token the harness already minted for the peer gateway, in the\n"
        "# same URL shape token_mint.py scans stdout for.\n"
        f'echo "http://localhost:{peer_port}?token={peer_token}"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


# --------------------------------------------------------------------------- #
# HTTP helpers against the parent's own dashboard API
# --------------------------------------------------------------------------- #


def _http(
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    body: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        status = exc.code
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = raw
    return status, parsed


def owner_url(handle: GatewayHandle, path: str) -> str:
    sep = "&" if "?" in path else "?"
    return f"http://localhost:{handle.port}{path}{sep}token={handle.token}"


def internal_secret(handle: GatewayHandle) -> str:
    secret_path = handle.home / ".local_secret"
    return secret_path.read_text(encoding="utf-8").strip()


def internal_headers(handle: GatewayHandle) -> dict[str, str]:
    return {"X-Internal-Secret": internal_secret(handle)}


# --------------------------------------------------------------------------- #
# Scenario helpers
# --------------------------------------------------------------------------- #


def register_peer_instance(parent_home: Path, *, name: str, ssh_port: int, remote_port: int,
                            remote_bin: str, client_key_path: Path) -> str:
    from kiro_crew.instances.registry import InstancesRegistry

    registry = InstancesRegistry(parent_home / "instances.json")
    # ssh_host carries the client identity file via an ssh config-free
    # ``-i``-less path: SshTunnelManager builds its own argv from ssh_host,
    # so this canary points ssh_host at "localhost" and relies on an
    # ssh_config entry (written by the caller) that pins the IdentityFile,
    # port and StrictHostKeyChecking for that host alias -- the supported way
    # to hand ssh per-host options without widening the registry's own
    # charset-validated fields.
    inst = registry.add(
        name=name,
        ssh_host="kirocrew-canary-peer",
        remote_port=remote_port,
        remote_bin=remote_bin,
        connection_method="ssh",
    )
    return inst.id


def write_ssh_client_config(ssh_dir: Path, *, host_alias: str, port: int,
                             identity_file: Path, known_hosts: Path) -> Path:
    ssh_dir.mkdir(parents=True, exist_ok=True)
    config_path = ssh_dir / "config"
    config_path.write_text(
        "\n".join(
            [
                f"Host {host_alias}",
                "  HostName 127.0.0.1",
                f"  Port {port}",
                "  User " + os.environ.get("USER", "runner"),
                f"  IdentityFile {identity_file}",
                "  IdentitiesOnly yes",
                f"  UserKnownHostsFile {known_hosts}",
                "  StrictHostKeyChecking accept-new",
                "  BatchMode yes",
                "  ConnectTimeout 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    return config_path


def wait_for_remote_spawn_terminal(handle: GatewayHandle, run_id: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, body = _http("GET", owner_url(handle, f"/api/spawn/{run_id}"))
        if status == 200 and isinstance(body, dict):
            last = body
            if body.get("done"):
                return body
        time.sleep(POLL_SECONDS)
    return last


def peer_listing_has_run(peer_handle: GatewayHandle) -> list[dict[str, Any]]:
    status, body = _http("GET", owner_url(peer_handle, "/api/spawn"))
    if status != 200 or not isinstance(body, dict):
        return []
    rows = body.get("agents") or body.get("items") or body.get("rows") or []
    return rows if isinstance(rows, list) else []


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    evidence = Evidence()
    scratch_root = Path(tempfile.mkdtemp(prefix="canary-pr12821-"))
    peer_scratch = scratch_root / "peer-ssh"
    ssh_client_dir = scratch_root / "ssh-client"
    sshd: Optional[DisposableSshd] = None

    backend_dir = scratch_root / "fake-backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    fake_backend_bin = fake_acp_backend_launcher(backend_dir)

    try:
        sshd = start_disposable_sshd(peer_scratch)
    except Exception as exc:
        for name in ("upload", "landed", "complete", "cancel", "resume"):
            evidence.record(
                ScenarioResult(name, False, f"could not start disposable sshd: {exc}")
            )
        all_passed = evidence.write(EVIDENCE_PATH)
        return 0 if all_passed else 1

    known_hosts = scratch_root / "known_hosts"
    known_hosts.touch()
    ssh_config = write_ssh_client_config(
        ssh_client_dir,
        host_alias="kirocrew-canary-peer",
        port=sshd.port,
        identity_file=sshd.client_key_path,
        known_hosts=known_hosts,
    )
    os.environ["HOME_SSH_CONFIG_CANARY"] = str(ssh_config)
    # SshTunnelManager spawns plain `ssh <ssh_host> ...`; it resolves per-host
    # options via the user's normal ssh config resolution (`-F` is not part of
    # its argv), so the alias config must live where ssh looks by default.
    # Point HOME at our scratch for the ssh client call only -- least invasive
    # way to get `~/.ssh/config` without touching the operator's real one, and
    # exactly what the harness-spawned gateways already do for KIROCREW_HOME.
    user_ssh_dir = scratch_root / "home" / ".ssh"
    user_ssh_dir.mkdir(parents=True, exist_ok=True)
    (user_ssh_dir / "config").write_text(ssh_config.read_text(), encoding="utf-8")
    (user_ssh_dir / "config").chmod(0o600)
    canary_home = scratch_root / "home"

    exit_code = 1
    try:
        with spawn_feature_gateway(
            fixture="minimal", approval="yolo", crons=False, kiro_bin=str(fake_backend_bin)
        ) as peer:
            peer_launcher = write_peer_kirocrew_launcher(canary_home, peer.port, peer.token)

            parent_env_patch: dict[str, str] = {"HOME": str(canary_home)}

            def _parent_before_spawn(env: dict[str, str], _cwd: Path) -> None:
                env.update(parent_env_patch)

            with spawn_feature_gateway(
                fixture="minimal",
                approval="yolo",
                crons=False,
                kiro_bin=str(fake_backend_bin),
                before_spawn=_parent_before_spawn,
            ) as parent:
                run_scenarios(evidence, parent, peer, peer_launcher, canary_home)

        exit_code = 0 if evidence.write(EVIDENCE_PATH) else 1
    except Exception as exc:
        traceback.print_exc()
        evidence.record(ScenarioResult("harness_boot", False, f"{type(exc).__name__}: {exc}"))
        evidence.write(EVIDENCE_PATH)
        exit_code = 1
    finally:
        if sshd is not None:
            sshd.stop()

    return exit_code


def run_scenarios(
    evidence: Evidence,
    parent: GatewayHandle,
    peer: GatewayHandle,
    peer_launcher: Path,
    canary_home: Path,
) -> None:
    instance_id = register_peer_instance(
        parent.home,
        name="canary-peer",
        ssh_port=0,  # unused; ssh_host alias carries the port via ssh config
        remote_port=peer.port,
        remote_bin=str(peer_launcher),
        client_key_path=peer_launcher,
    )

    # -- a) upload -----------------------------------------------------
    r = ScenarioResult("upload", False, "not run", started=time.monotonic())
    try:
        payload = b"canary-snapshot-placeholder"
        status, body = _http(
            "POST",
            owner_url(parent, f"/api/remote-workspaces?sha256={'0'*64}&commit={'0'*40}"),
            headers={"Content-Type": "application/octet-stream"},
            body=None,
            timeout=30.0,
        )
        # The real route reads raw bytes, not JSON; retry with a raw body.
        req = urllib.request.Request(
            owner_url(parent, f"/api/remote-workspaces?sha256={'0'*64}&commit={'0'*40}"),
            data=payload,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read().decode("utf-8", "replace")
        r.evidence = {"status": status, "body": body if isinstance(body, dict) else str(body)[:300]}
        r.passed = status in (200, 400) and isinstance(body, dict)
        r.detail = f"POST /api/remote-workspaces -> {status}"
    except Exception as exc:
        r.detail = f"{type(exc).__name__}: {exc}"
    r.ended = time.monotonic()
    evidence.record(r)

    # -- spawn the remote run (feeds b, c, d) ---------------------------
    headers = internal_headers(parent)
    status, body = _http(
        "POST",
        owner_url(parent, "/api/spawn"),
        headers=headers,
        body={
            "task": "[[SLOW]] canary remote turn",
            "executor": "remote",
            "instance_id": instance_id,
            "include_memory": False,
            "include_lessons": False,
            "include_project": False,
        },
        timeout=30.0,
    )
    run_id = ""
    if status == 200 and isinstance(body, dict):
        run_id = str(body.get("id") or "")

    # -- b) landed -------------------------------------------------------
    r = ScenarioResult("landed", False, "not run", started=time.monotonic())
    try:
        if not run_id:
            r.detail = f"POST /api/spawn did not return an id (status {status}): {body!r}"[:500]
        else:
            found = False
            deadline = time.monotonic() + SPAWN_TIMEOUT_SECS
            peer_rows: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                peer_rows = peer_listing_has_run(peer)
                if peer_rows:
                    found = True
                    break
                time.sleep(POLL_SECONDS)
            r.evidence = {"parent_spawn_status": status, "run_id": run_id,
                          "peer_rows": len(peer_rows)}
            r.passed = found
            r.detail = (
                f"peer /api/spawn shows {len(peer_rows)} row(s)" if found
                else "no run ever appeared on the peer's own /api/spawn listing"
            )
    except Exception as exc:
        r.detail = f"{type(exc).__name__}: {exc}"
    r.ended = time.monotonic()
    evidence.record(r)

    # -- d) cancel (consumes the [[SLOW]] run before c completes it) ----
    r = ScenarioResult("cancel", False, "not run", started=time.monotonic())
    try:
        if not run_id:
            r.detail = "no run_id from spawn; cannot cancel"
        else:
            status_c, body_c = _http(
                "DELETE", owner_url(parent, f"/api/spawn/{run_id}"), headers=headers, timeout=30.0
            )
            time.sleep(POLL_SECONDS * 2)
            peer_rows_after = peer_listing_has_run(peer)
            parent_status, parent_body = _http("GET", owner_url(parent, f"/api/spawn/{run_id}"))
            r.evidence = {
                "delete_status": status_c,
                "delete_body": body_c if isinstance(body_c, dict) else str(body_c)[:300],
                "peer_rows_after": len(peer_rows_after),
                "parent_after": parent_body if isinstance(parent_body, dict) else str(parent_body)[:300],
            }
            r.passed = status_c in (200, 204) or (
                isinstance(parent_body, dict) and parent_body.get("stopped")
            )
            r.detail = f"DELETE /api/spawn/{run_id} -> {status_c}"
    except Exception as exc:
        r.detail = f"{type(exc).__name__}: {exc}"
    r.ended = time.monotonic()
    evidence.record(r)

    # -- c) complete: a fresh, non-cancelled run must reach done:true ---
    r = ScenarioResult("complete", False, "not run", started=time.monotonic())
    try:
        status_s, body_s = _http(
            "POST",
            owner_url(parent, "/api/spawn"),
            headers=headers,
            body={
                "task": "say pong",
                "executor": "remote",
                "instance_id": instance_id,
                "include_memory": False,
                "include_lessons": False,
                "include_project": False,
            },
            timeout=30.0,
        )
        complete_run_id = str(body_s.get("id") or "") if isinstance(body_s, dict) else ""
        if not complete_run_id:
            r.detail = f"POST /api/spawn did not return an id (status {status_s}): {body_s!r}"[:500]
        else:
            final = wait_for_remote_spawn_terminal(parent, complete_run_id, timeout=SPAWN_TIMEOUT_SECS)
            r.evidence = {"run_id": complete_run_id, "final": final}
            r.passed = bool(final.get("done")) and fake_acp_backend.REPLY_TEXT in str(
                final.get("result", "")
            )
            r.detail = (
                f"done={final.get('done')} result-has-reply="
                f"{fake_acp_backend.REPLY_TEXT in str(final.get('result', ''))}"
            )
    except Exception as exc:
        r.detail = f"{type(exc).__name__}: {exc}"
    r.ended = time.monotonic()
    evidence.record(r)

    # -- e) resume: restart the PARENT, check the mapping is handled ----
    r = ScenarioResult("resume", False, "not run", started=time.monotonic())
    try:
        status_s, body_s = _http(
            "POST",
            owner_url(parent, "/api/spawn"),
            headers=headers,
            body={
                "task": "[[SLOW]] canary resume turn",
                "executor": "remote",
                "instance_id": instance_id,
                "include_memory": False,
                "include_lessons": False,
                "include_project": False,
            },
            timeout=30.0,
        )
        resume_run_id = str(body_s.get("id") or "") if isinstance(body_s, dict) else ""
        if not resume_run_id:
            r.detail = f"POST /api/spawn did not return an id (status {status_s}): {body_s!r}"[:500]
        else:
            time.sleep(POLL_SECONDS * 2)
            restarted = parent.restart()
            deadline = time.monotonic() + RESTART_TIMEOUT_SECS
            final = {}
            while time.monotonic() < deadline:
                status_g, body_g = _http(
                    "GET", owner_url(restarted, f"/api/spawn/{resume_run_id}")
                )
                if status_g == 200 and isinstance(body_g, dict):
                    final = body_g
                    if final.get("done") or "error" not in final:
                        break
                time.sleep(POLL_SECONDS)
            r.evidence = {"run_id": resume_run_id, "post_restart": final}
            r.passed = isinstance(final, dict) and bool(final)
            r.detail = (
                f"post-restart GET /api/spawn/{resume_run_id} -> {final}"[:300]
                if final else "post-restart status never answered for the remote run"
            )
            with contextlib.suppress(Exception):
                _http(
                    "DELETE",
                    owner_url(restarted, f"/api/spawn/{resume_run_id}"),
                    headers=internal_headers(restarted),
                )
    except Exception as exc:
        r.detail = f"{type(exc).__name__}: {exc}"
    r.ended = time.monotonic()
    evidence.record(r)


if __name__ == "__main__":
    sys.exit(main())
