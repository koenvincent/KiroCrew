"""``task.split`` in the runner: who gets asked, and where the receipt lands.

Three halves.

WHO gets asked. The point is for a message the dashboard owner typed, so an
app-token send, a cron, a sub-agent, a gateway stage, a channel relay and a
nudge-loop wake must ask nothing at all -- checked against the real
``_run_chat``, not against the predicate alone, because the predicate is only
worth what its four arguments are wired to.

WHAT the agent did. The baseline arm is counted from the TRUSTED ``_meta.kiro``
identity of each tool call, so a scripted turn that calls two spawns reads as
``split`` while the same turn calling a shell tool named ``spawn_run`` reads as
``single``. That mutation is the whole claim of the comparison: nothing the model
writes may move the arm its own suggestion is scored against.

WHERE the receipt lands. On the turn's FINAL assistant row, through both doors a
client reads, and BESIDE a ``skills.select`` strip on the same reply rather than
displacing it -- the collision that kept this point off ``decisions.outcomes``.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from chat_test_helpers import _make_ready_kiro_prerequisite

from kiro_crew.acp.types import EVENT_COMPLETE, EVENT_TEXT_CHUNK, EVENT_TOOL_CALL
from kiro_crew.dashboard import chat_runner
from kiro_crew.dashboard.chat_persistence import _build_message_entry_uncached
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.decisions.points import task_split as ts
from kiro_crew.history import ConversationLog
from kiro_crew.providers.base import LLMEvent

#: What the point returns for a decided turn -- patched in, so the wiring under
#: test is the call site rather than the oracle.
DECIDED = {"turn_id": "ts-7", "choice": ts.CHOICE_SPLIT, "p": 0.84, "latency_ms": 190}

#: The row the point writes for it, as the caller stamps it.
RECORD = {
    "ts": "2026-09-21T07:00:00+00:00",
    "point": "task.split",
    "session": "0123456789ab",
    "latency_ms": 190,
    "scrubbed": False,
    "answers": None,
    "error": None,
    "turn_id": "ts-7",
    "jev_choice": "split",
    "agent_choice": "split",
    "spawn_calls": 2,
    "agree": True,
    "p": 0.84,
}


def _state(tmp_path) -> DashboardState:
    sessions = MagicMock(count=0)
    sessions.get_pid = MagicMock(return_value=None)
    sessions.get_slack_link = MagicMock(return_value=(None, None))
    sessions.get_mirror_link = MagicMock(return_value=None)
    sessions.get_provider = MagicMock(return_value=None)
    sessions.resumable_sid = MagicMock(return_value=None)
    sessions.check_context_usage = MagicMock()
    sessions.reset = AsyncMock()
    sessions.remove = AsyncMock()
    sessions.record_failure = AsyncMock()
    state = DashboardState(
        sessions=sessions,
        crons=MagicMock(list_jobs=MagicMock(return_value=[]), status=MagicMock(return_value={})),
        lessons=MagicMock(load_all=MagicMock(return_value=[])),
        start_time=0.0,
        conversation_log=ConversationLog(base_dir=tmp_path),
    )
    state.kiro_prerequisite_service = _make_ready_kiro_prerequisite()
    state.broadcast_ws = MagicMock()
    state.push_slots_update = MagicMock()
    state.push_refresh = MagicMock()
    state.refresh_slot_source_status = MagicMock()
    state.broadcast_context_usage = MagicMock()
    return state


def _runner(tmp_path):
    """``(state, client)`` wired for a scripted ``_run_chat`` turn."""
    state = _state(tmp_path)
    client = AsyncMock()
    client.context_usage_pct = MagicMock(return_value=0.0)
    client.context_window_tokens = MagicMock(return_value=0)
    client.context_used_tokens = MagicMock(return_value=0)
    client.last_prompt_stats = None
    client._client = client
    client.exit_code = None
    state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
    state.sessions.consume_replay_suppression = MagicMock(return_value=False)
    state.sessions.consume_needs_reinjection = MagicMock(return_value=False)
    state._hook_store = MagicMock(fire=AsyncMock(return_value=[]))
    return state, client


def _slot(key: str = "chat-split-1") -> _ChatSlot:
    slot = _ChatSlot(key)
    slot._titled = True
    return slot


@contextmanager
def _quiet_sel():
    with patch.object(chat_runner, "sel") as mock_sel:
        mock_sel.return_value = MagicMock()
        yield mock_sel


async def _settle(slot) -> None:
    task = slot.task
    if task is None or not hasattr(task, "cancel"):
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:  # pragma: no cover - draining, never the assertion
        pass


def _spawn_call(tool_call_id: str, *, server: str = ts.CORE_MCP_SERVER, tool: str = "spawn_run"):
    return LLMEvent(
        kind=EVENT_TOOL_CALL,
        title=tool,
        tool_name=tool,
        mcp_server_name=server,
        tool_call_id=tool_call_id,
        tool_kind="other",
        tool_input="{}",
    )


def _scripts(client, events) -> None:
    """Script ``client.stream`` so one turn yields *events* then completes."""

    async def _once():
        for event in events:
            yield event
        yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

    client.stream = MagicMock(side_effect=lambda *_a, **_k: _once())


@contextmanager
def _answering(decided: dict | None, *, record: dict | None = RECORD):
    """Patch the POINT's two halves and report what each was handed."""
    asked: list[dict] = []
    recorded: list[dict] = []

    async def _suggest(text, **kwargs):
        asked.append({"text": text, **kwargs})
        return decided

    async def _record(**kwargs):
        recorded.append(kwargs)
        return record

    with patch.object(ts, "suggest", _suggest), patch.object(ts, "record_outcome", _record):
        yield asked, recorded


def _assistant_rows(slot: _ChatSlot) -> list[dict]:
    return [m for m in slot.messages if m.get("role") == "assistant"]


def _record_of(msg: dict) -> object:
    return (msg.get("meta") or {}).get("decisions_split")


def _text(body: str) -> LLMEvent:
    return LLMEvent(kind=EVENT_TEXT_CHUNK, text=body)


# ── who gets asked ────────────────────────────────────────────────────────────


class TestOnlyAnOwnerTurnIsAsked:
    @pytest.mark.asyncio
    async def test_an_owner_send_is_asked_and_the_hint_is_prepended(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("on it")])

        with _quiet_sel(), _answering(DECIDED) as (asked, _recorded):
            await chat_runner._run_chat(
                state, slot, "audit these four modules", _directive_user_origin=True
            )
        await _settle(slot)

        assert len(asked) == 1, "one question per owner turn"
        assert asked[0]["text"] == "audit these four modules"
        prompt = client.stream.call_args[0][0]
        assert "Jev suggests:" in prompt, "one advisory line, prepended"
        assert "You decide." in prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            # An app token: the middleware stamps the actor and clears user origin.
            {"_directive_user_origin": False, "_turn_actor": "app"},
            {"_directive_user_origin": True, "_turn_actor": "cron"},
            {"_directive_user_origin": True, "_turn_actor": "subagent"},
            {"_directive_user_origin": True, "_turn_actor": "gateway"},
            # A nudge/monitor loop waking the slot.
            {"_directive_user_origin": True, "_directive_self_wake": True},
            # A Slack or Discord relay.
            {"_directive_user_origin": True, "_directive_channel_origin": True},
            # No dispatch claimed a person at all.
            {},
        ],
    )
    async def test_nothing_else_is_asked(self, tmp_path, kwargs):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("done")])

        with _quiet_sel(), _answering(DECIDED) as (asked, recorded):
            await chat_runner._run_chat(state, slot, "do the thing", **kwargs)
        await _settle(slot)

        assert asked == [], "asked on a turn nobody typed"
        assert recorded == []
        assert "Jev suggests:" not in client.stream.call_args[0][0]
        assert _record_of(_assistant_rows(slot)[-1]) is None

    @pytest.mark.asyncio
    async def test_a_refusal_prepends_nothing_and_records_nothing(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("done")])

        with _quiet_sel(), _answering(None) as (asked, recorded):
            await chat_runner._run_chat(state, slot, "do the thing", _directive_user_origin=True)
        await _settle(slot)

        assert len(asked) == 1, "the point was called; it declined"
        assert recorded == [], "no suggestion, so no arm to compare and no row"
        assert "Jev suggests:" not in client.stream.call_args[0][0]
        assert _record_of(_assistant_rows(slot)[-1]) is None


class TestTheOwnerPredicateIsStructural:
    """The four arguments, held one at a time.

    Read directly as well as through the runner: the runner proves the wiring, and
    this proves the rule, so a future dispatch that stamps a new actor is refused
    by the same line rather than by a branch somebody remembered.
    """

    def test_an_owner_turn_is_all_four(self):
        assert chat_runner._is_owner_turn(
            user_origin=True, turn_actor="", self_wake=False, channel_origin=False
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"user_origin": False},
            {"turn_actor": "app"},
            {"turn_actor": "cron"},
            {"self_wake": True},
            {"channel_origin": True},
        ],
    )
    def test_any_one_of_them_refuses(self, kwargs):
        base = {
            "user_origin": True,
            "turn_actor": "",
            "self_wake": False,
            "channel_origin": False,
        }
        assert chat_runner._is_owner_turn(**{**base, **kwargs}) is False


# ── what the agent did ────────────────────────────────────────────────────────


class TestTheBaselineArmIsCountedFromTheTurn:
    @pytest.mark.asyncio
    async def test_two_core_spawn_calls_are_counted(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_spawn_call("tc-1"), _spawn_call("tc-2"), _text("both running")])

        with _quiet_sel(), _answering(DECIDED) as (_asked, recorded):
            await chat_runner._run_chat(
                state, slot, "audit these four modules", _directive_user_origin=True
            )
        await _settle(slot)

        assert len(recorded) == 1
        assert recorded[0]["spawn_calls"] == 2
        assert ts.agent_choice_for(recorded[0]["spawn_calls"]) == ts.CHOICE_SPLIT

    @pytest.mark.asyncio
    async def test_a_batch_spawn_and_a_single_one_are_both_counted(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(
            client,
            [
                _spawn_call("tc-1", tool="spawn_sub_agents"),
                _spawn_call("tc-2", tool="mcp__kirocrew-core__spawn_run"),
                _text("ok"),
            ],
        )

        with _quiet_sel(), _answering(DECIDED) as (_asked, recorded):
            await chat_runner._run_chat(state, slot, "split this up", _directive_user_origin=True)
        await _settle(slot)

        assert recorded[0]["spawn_calls"] == 2

    @pytest.mark.asyncio
    async def test_no_spawn_call_reads_as_a_single_worker(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("here is the answer")])

        with _quiet_sel(), _answering(DECIDED) as (_asked, recorded):
            await chat_runner._run_chat(
                state, slot, "audit these four modules", _directive_user_origin=True
            )
        await _settle(slot)

        assert recorded[0]["spawn_calls"] == 0
        assert ts.agent_choice_for(recorded[0]["spawn_calls"]) == ts.CHOICE_SINGLE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "server,tool",
        [
            # A shell tool the MODEL named `spawn_run`: no MCP server identity.
            ("", "spawn_run"),
            # A third-party MCP server exposing a same-named tool.
            ("other-server", "spawn_run"),
            # A core tool that is not a spawn.
            (ts.CORE_MCP_SERVER, "send_message"),
        ],
    )
    async def test_nothing_the_model_can_write_moves_the_arm(self, tmp_path, server, tool):
        """The mutation the comparison rests on.

        Two calls that LOOK like spawns still read as ``single``, because the count
        is taken from the trusted ``_meta.kiro`` identity rather than from a title.
        """
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(
            client,
            [
                _spawn_call("tc-1", server=server, tool=tool),
                _spawn_call("tc-2", server=server, tool=tool),
                _text("ok"),
            ],
        )

        with _quiet_sel(), _answering(DECIDED) as (_asked, recorded):
            await chat_runner._run_chat(state, slot, "do the thing", _directive_user_origin=True)
        await _settle(slot)

        assert recorded[0]["spawn_calls"] == 0

    @pytest.mark.asyncio
    async def test_an_unasked_turn_counts_nothing(self, tmp_path):
        """The count is only kept while a suggestion is outstanding."""
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_spawn_call("tc-1"), _text("ok")])

        with _quiet_sel(), _answering(None) as (_asked, recorded):
            await chat_runner._run_chat(state, slot, "do the thing", _directive_user_origin=True)
        await _settle(slot)

        assert recorded == []


# ── where the receipt lands ───────────────────────────────────────────────────


class TestTheRecordRidesTheFinalReply:
    @pytest.mark.asyncio
    async def test_it_lands_on_the_last_assistant_row(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        # Text, then a spawn, then the final answer: the FIRST segment flushes
        # before the spawn, so a receipt claimed there would compare the
        # suggestion against a count that had not finished.
        _scripts(client, [_text("planning"), _spawn_call("tc-1"), _text("the answer")])

        with _quiet_sel(), _answering(DECIDED):
            await chat_runner._run_chat(
                state, slot, "audit these modules", _directive_user_origin=True
            )
        await _settle(slot)

        rows = _assistant_rows(slot)
        assert len(rows) >= 2, f"expected a pre-tool segment and a final one, got {rows}"
        assert _record_of(rows[-1]) == RECORD
        assert all(_record_of(row) is None for row in rows[:-1]), "one receipt per turn"

    @pytest.mark.asyncio
    async def test_it_reaches_both_doors_from_one_write(self, tmp_path):
        """The live frame ``slot.append`` broadcasts, and the persisted line."""
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("the answer")])

        with _quiet_sel(), _answering(DECIDED):
            await chat_runner._run_chat(state, slot, "do the thing", _directive_user_origin=True)
        await _settle(slot)

        row = _assistant_rows(slot)[-1]
        assert _record_of(row) == RECORD
        entry = _build_message_entry_uncached(row)
        assert (entry.get("meta") or {}).get("decisions_split") == RECORD

    @pytest.mark.asyncio
    async def test_a_row_the_writer_refused_stamps_nothing(self, tmp_path):
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("the answer")])

        with _quiet_sel(), _answering(DECIDED, record=None):
            await chat_runner._run_chat(state, slot, "do the thing", _directive_user_origin=True)
        await _settle(slot)

        assert _record_of(_assistant_rows(slot)[-1]) is None

    @pytest.mark.asyncio
    async def test_it_sits_beside_a_skills_select_strip_rather_than_replacing_it(self, tmp_path):
        """The collision that kept this point off ``decisions.outcomes``.

        That registry holds ONE outcome per session and ``consume`` pops, so a
        ``skills.select`` turn's strip and this receipt would be two claims on one
        slot. Two keys, one row.
        """
        from kiro_crew.decisions import outcomes

        strip = {"turn_id": "sk-1", "point": "skills.select", "baseline": [], "jev": ["a"]}
        state, client = _runner(tmp_path)
        slot = _slot()
        _scripts(client, [_text("the answer")])

        outcomes.reset()
        try:
            with _quiet_sel(), _answering(DECIDED):
                # Published from inside the turn, the way prompt assembly does it:
                # after `_discard_stale_decision` has run on the way in.
                original = chat_runner._task_split_suggestion

                async def _publishing(*args, **kwargs):
                    outcomes.publish(chat_runner.effective_session_key(slot), strip)
                    return await original(*args, **kwargs)

                with patch.object(chat_runner, "_task_split_suggestion", _publishing):
                    await chat_runner._run_chat(
                        state, slot, "do the thing", _directive_user_origin=True
                    )
            await _settle(slot)
        finally:
            outcomes.reset()

        meta = _assistant_rows(slot)[-1]["meta"]
        assert meta["decisions_split"] == RECORD
        assert meta["decisions_strip"] == strip, "the strip was not displaced"


class TestAnOrdinaryRowIsUnchanged:
    def test_no_fragment_writes_no_meta_key(self):
        """``slot.append`` writes no ``meta`` for ``None``, which is what keeps a
        turn with no decision byte-identical to one this build appends today."""
        assert chat_runner._merged_row_meta(None, None) is None
        assert chat_runner._merged_row_meta({}, None) is None

    def test_both_fragments_are_merged(self):
        merged = chat_runner._merged_row_meta({"decisions_strip": 1}, {"decisions_split": 2})
        assert merged == {"decisions_strip": 1, "decisions_split": 2}

    def test_the_record_key_is_reserved_to_the_gateway(self):
        """A request's own ``meta`` cannot carry one: it rides verbatim onto the
        persisted row, so a caller could otherwise stamp a decision nobody made."""
        from kiro_crew.dashboard.chat_handlers import RESERVED_ROW_META_KEYS

        assert "decisions_split" in RESERVED_ROW_META_KEYS
