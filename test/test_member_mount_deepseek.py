"""A deepseek member DM session holds the member session-control tools and the panel.

deepseek has no mirror, so its array is composed by
``AcpClient._mirrorless_member_mounts`` rather than by a projection. These drive the
REAL resolver over a real agent spec and read the elements that come out, the same
way the goose and opencode array tests do. Driving a live deepseek session would
need the harness binary and a model, which this test host does not have; the mount
itself is pinned by the committed round trip in
``test/fixtures/acp_frames/deepseek/mcp-stdio-mount-live.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kiro_crew.validation  # noqa: F401 - break the legacy import cycle first
from kiro_crew import acp_tool_gate
from kiro_crew.acp.client import AcpClient
from kiro_crew.acp.types import (
    ACP_BACKEND_DEEPSEEK,
    ACP_BACKEND_PI,
    ACP_BACKENDS_MEMBER_DISPATCH,
    ACP_BACKENDS_MEMBER_PANEL,
    ACP_BACKENDS_SESSION_MCP_ARRAY,
)
from kiro_crew.mcp_gateway.claim import STUB_SESSION_TOKEN_ENV
from kiro_crew.members import MEMBER_DISPATCH_SERVER, MEMBER_PANEL_SERVER

MEMBER_KEY = "dashboard_member-autofix"
FIXTURES = Path(__file__).parent / "fixtures" / "acp_frames" / "deepseek"


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    """An agent spec on disk that DECLARES the dashboard server, and the panel on."""
    import kiro_crew.agent as agent_mod
    import kiro_crew.members as members_mod
    from kiro_crew.acp import session_mcp

    d = tmp_path / "agents"
    d.mkdir()
    monkeypatch.setattr(agent_mod, "KIRO_AGENTS_DIR", d)
    monkeypatch.setattr(agent_mod, "_KIRO_MCP_JSON", tmp_path / "settings-mcp.json")
    monkeypatch.setattr(session_mcp, "ensure_agent_materialized", lambda _a: True)
    monkeypatch.setattr(session_mcp, "_mcp_registry_mode", lambda: False)
    monkeypatch.setattr(members_mod, "crew_panel_enabled", lambda: True)
    (d / "kirocrew.json").write_text(
        json.dumps(
            {
                "name": "kirocrew",
                "mcpServers": {
                    "kirocrew-core": {"command": "/opt/kirocrew", "args": ["mcp-core"]},
                    MEMBER_DISPATCH_SERVER: {
                        "command": "/opt/kirocrew",
                        "args": ["mcp-dashboard"],
                    },
                },
                "tools": ["@kirocrew-core", "@" + MEMBER_DISPATCH_SERVER],
            }
        ),
        encoding="utf-8",
    )
    return d


def _client(tmp_path: Path, session_key: str, backend: str = ACP_BACKEND_DEEPSEEK):
    """A minimally constructed ``AcpClient``: only what the resolver reads."""
    client = object.__new__(AcpClient)
    client._acp_backend = backend
    client._mcp_gateway_overlay = None  # shared gateway off: no broker stubs
    client._agent = "kirocrew"
    client._work_dir = tmp_path / "work"
    client._session_key = session_key
    client._channel_id = "dashboard"
    client._stub_session_token = "9" * 64
    client._claude_settings_authored = False
    client._session_mcp_cache = None
    return client


def _names(out: list[dict]) -> list[str]:
    return [e["name"] for e in out]


class TestTheDecision:
    def test_deepseek_holds_both_things_membership_needs(self):
        """The mount it rides, and the gate that makes the mount safe, for THIS harness."""
        assert ACP_BACKEND_DEEPSEEK in ACP_BACKENDS_MEMBER_DISPATCH
        assert ACP_BACKEND_DEEPSEEK in ACP_BACKENDS_MEMBER_PANEL
        assert ACP_BACKEND_DEEPSEEK in ACP_BACKENDS_SESSION_MCP_ARRAY
        assert acp_tool_gate.is_enforced(ACP_BACKEND_DEEPSEEK) is True

    def test_the_round_trip_the_rationale_names_is_committed(self):
        body = (FIXTURES / "mcp-stdio-mount-live.jsonl").read_text(encoding="utf-8")
        assert "tools/call" in body
        assert "mcp__crew-probe__crew_probe_echo" in body


class TestDeepseekSessionArray:
    def test_a_member_session_carries_both_member_elements(self, agents_dir, tmp_path):
        client = _client(tmp_path, MEMBER_KEY)
        out = client._resolve_session_mcp_servers()
        assert _names(out) == [MEMBER_DISPATCH_SERVER, MEMBER_PANEL_SERVER]
        for element in out:
            env = {p["name"]: p["value"] for p in element["env"]}
            assert env["KIROCREW_SESSION_KEY"] == MEMBER_KEY
            assert env[STUB_SESSION_TOKEN_ENV] == client._stub_session_token
        assert client.member_dispatch_mounted is True

    def test_the_wire_accessor_hands_out_the_same_array(self, agents_dir, tmp_path):
        client = _client(tmp_path, MEMBER_KEY)
        assert _names(client._deepseek_session_mcp_servers()) == [
            MEMBER_DISPATCH_SERVER,
            MEMBER_PANEL_SERVER,
        ]

    def test_an_ordinary_session_carries_nothing(self, agents_dir, tmp_path):
        """The spec DECLARED the dashboard server; an ordinary session still gets none."""
        client = _client(tmp_path, "dashboard_abc123")
        assert client._resolve_session_mcp_servers() == []
        assert client.member_dispatch_mounted is False

    def test_a_disabled_dashboard_server_withholds_only_that_mount(self, agents_dir, tmp_path):
        (tmp_path / "settings-mcp.json").write_text(
            json.dumps({"mcpServers": {MEMBER_DISPATCH_SERVER: {"disabled": True}}}),
            encoding="utf-8",
        )
        client = _client(tmp_path, MEMBER_KEY)
        assert _names(client._resolve_session_mcp_servers()) == [MEMBER_PANEL_SERVER]

    def test_a_switched_off_dashboard_tool_withholds_the_whole_mount(self, agents_dir, tmp_path):
        """No mirror means no per-tool deny channel, so the narrowed server stays off."""
        (tmp_path / "settings-mcp.json").write_text(
            json.dumps(
                {"mcpServers": {MEMBER_DISPATCH_SERVER: {"disabledTools": ["session_stop"]}}}
            ),
            encoding="utf-8",
        )
        client = _client(tmp_path, MEMBER_KEY)
        assert _names(client._resolve_session_mcp_servers()) == [MEMBER_PANEL_SERVER]

    def test_the_panel_switch_withholds_only_the_panel(self, agents_dir, tmp_path, monkeypatch):
        import kiro_crew.members as members_mod

        monkeypatch.setattr(members_mod, "crew_panel_enabled", lambda: False)
        client = _client(tmp_path, MEMBER_KEY)
        assert _names(client._resolve_session_mcp_servers()) == [MEMBER_DISPATCH_SERVER]

    def test_an_unreadable_spec_mounts_nothing(self, agents_dir, tmp_path, monkeypatch):
        from kiro_crew.acp import session_mcp

        def _boom(*_a, **_k):
            raise OSError("unreadable")

        monkeypatch.setattr(session_mcp, "session_mcp_projection", _boom)
        client = _client(tmp_path, MEMBER_KEY)
        assert client._resolve_session_mcp_servers() == []

    def test_pi_stays_plain_chat(self, agents_dir, tmp_path):
        """The mirror-less branch is gated on membership, and pi is not a member."""
        client = _client(tmp_path, MEMBER_KEY, backend=ACP_BACKEND_PI)
        assert client._resolve_session_mcp_servers() == []


class TestPooledStubsBesideTheMount:
    def test_a_same_named_stub_is_dropped(self, agents_dir, tmp_path, monkeypatch):
        client = _client(tmp_path, MEMBER_KEY)
        client._session_mcp_cache = client._resolve_session_mcp_servers()
        stubs = [{"name": MEMBER_DISPATCH_SERVER}, {"name": "kirocrew-core"}]
        monkeypatch.setattr(AcpClient, "_pooled_broker_stubs", lambda self: list(stubs))
        assert _names(client._pooled_mcp_servers()) == ["kirocrew-core"]

    def test_an_ordinary_session_keeps_every_stub(self, agents_dir, tmp_path, monkeypatch):
        client = _client(tmp_path, "dashboard_abc123")
        client._session_mcp_cache = client._resolve_session_mcp_servers()
        stubs = [{"name": MEMBER_DISPATCH_SERVER}, {"name": "kirocrew-core"}]
        monkeypatch.setattr(AcpClient, "_pooled_broker_stubs", lambda self: list(stubs))
        assert _names(client._pooled_mcp_servers()) == [MEMBER_DISPATCH_SERVER, "kirocrew-core"]
