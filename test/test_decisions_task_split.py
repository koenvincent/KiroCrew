"""``task.split``: the three shapes, the hint, the baseline count, and the row.

The load-bearing groups are :class:`TestEveryRefusalPrependsNothing` -- the reason
this point is safe on the path that assembles every owner turn -- and
:class:`TestTheBaselineArmIsTheAgentsOwnBehaviour`, which pins that the arm the
suggestion is scored against comes from the TRUSTED tool identity, so nothing the
model writes can move it.

The runner wiring -- which turns get asked, and where the record lands -- is
``test_decisions_task_split_apply.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from kiro_crew import credential_patterns as _cred
from kiro_crew.decisions import log as log_mod
from kiro_crew.decisions.points import skills_select as ss
from kiro_crew.decisions.points import task_split as ts
from kiro_crew.decisions.types import Answer

#: An AWS key id assembled from the prefix list rather than written out, for the
#: reason ``test_decisions_gate`` gives: a contiguous key-shaped literal is refused
#: by the repo's own secret scanners, correctly, since neither they nor Semgrep can
#: tell a test vector from a leak.
_AWS_KEY = _cred.AWS_KEY_ID_PREFIXES.split("|")[0] + "A2B3C4D5E6F7G8H9"


@pytest.fixture
def answer(monkeypatch):
    """Install one ``decide`` answer and record what was asked. Returns a setter."""
    asked: list[dict] = []

    def _install(value, *, p=0.84):
        async def _decide(point, state, questions, **kwargs):
            asked.append({"point": point, "state": state, "questions": questions, "kwargs": kwargs})
            if value is None:
                return None
            return {ts.QUESTION_ID: Answer(id=ts.QUESTION_ID, value=value, p=p)}

        monkeypatch.setattr("kiro_crew.decisions.decide", _decide)
        return asked

    return _install


@pytest.fixture(autouse=True)
def no_history_budget(monkeypatch):
    """The shipped ceiling: 0, so a test that wants prior turns raises it itself."""
    monkeypatch.setattr("kiro_crew.decisions.history_budget_chars", lambda: 0)


@pytest.fixture
def written_rows(monkeypatch):
    """Capture every row the point hands the writer; report it as written."""
    rows: list[dict] = []

    def _append(row):
        rows.append(row)
        return True

    monkeypatch.setattr(log_mod, "append", _append)
    return rows


def _turns(*pairs):
    """Prior transcript rows in the shape ``ConversationLog.recent`` returns."""
    return [{"role": role, "content": content} for role, content in pairs]


class TestTheAnswerIsTheSuggestion:
    @pytest.mark.asyncio
    async def test_split_comes_back_as_the_split_shape(self, answer):
        asked = answer(ts.CHOICE_SPLIT)
        decided = await ts.suggest("audit these four modules independently")
        assert decided is not None
        assert decided["choice"] == ts.CHOICE_SPLIT
        assert decided["p"] == pytest.approx(0.84)
        assert isinstance(decided["turn_id"], str) and decided["turn_id"]
        assert decided["latency_ms"] >= 0
        assert asked[0]["point"] == ts.POINT

    @pytest.mark.asyncio
    async def test_each_shape_round_trips(self, answer):
        for shape in ts.CHOICES:
            answer(shape)
            decided = await ts.suggest("do the thing")
            assert decided is not None and decided["choice"] == shape

    @pytest.mark.asyncio
    async def test_the_three_shapes_are_the_whole_declared_domain(self, answer):
        answer(ts.CHOICE_SINGLE)
        asked = answer(ts.CHOICE_SINGLE)
        await ts.suggest("rename one function")
        questions = asked[0]["questions"]
        assert len(questions) == 1, "one Choice: the answer is consumed as one line"
        assert questions[0].id == ts.QUESTION_ID
        assert questions[0].options == list(ts.CHOICES)
        for shape in ts.CHOICES:
            assert shape in questions[0].prompt, "each option is defined in the rubric"


class TestEveryRefusalPrependsNothing:
    @pytest.mark.asyncio
    async def test_no_answer_is_none_not_a_shape(self, answer):
        answer(None)
        assert await ts.suggest("do the thing") is None

    @pytest.mark.asyncio
    async def test_an_answer_outside_the_domain_is_none(self, answer):
        answer("fan-out")
        assert await ts.suggest("do the thing") is None

    @pytest.mark.asyncio
    async def test_a_raising_provider_is_none(self, monkeypatch):
        async def _decide(*args, **kwargs):
            raise RuntimeError("transport")

        monkeypatch.setattr("kiro_crew.decisions.decide", _decide)
        assert await ts.suggest("do the thing") is None

    @pytest.mark.asyncio
    async def test_a_call_past_the_callers_budget_is_none(self, monkeypatch):
        monkeypatch.setattr(ts, "wait_budget", lambda: 0.01)

        async def _decide(*args, **kwargs):
            await asyncio.sleep(5)

        monkeypatch.setattr("kiro_crew.decisions.decide", _decide)
        assert await ts.suggest("do the thing") is None

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self, monkeypatch):
        async def _decide(*args, **kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr("kiro_crew.decisions.decide", _decide)
        with pytest.raises(asyncio.CancelledError):
            await ts.suggest("do the thing")

    @pytest.mark.asyncio
    async def test_a_refusal_writes_no_outcome_row(self, answer, written_rows):
        answer(None)
        assert await ts.suggest("do the thing") is None
        assert written_rows == [], "the gate writes its own row; the point writes none"


class TestTheHintIsAdviceAndNamesJev:
    def test_each_shape_has_its_own_sentence(self):
        seen = set()
        for shape in ts.CHOICES:
            line = ts.hint_line({"choice": shape, "p": 0.84})
            assert line.startswith("Jev suggests:"), "attributed, never an instruction"
            assert "You decide." in line, "the agent still chooses"
            assert "0.84" in line, "the score is readable so a weak hint can be discounted"
            seen.add(line)
        assert len(seen) == len(ts.CHOICES), "three distinct sentences"

    def test_the_split_sentence_names_parallel_sub_tasks(self):
        line = ts.hint_line({"choice": ts.CHOICE_SPLIT, "p": 0.84})
        assert "independent sub-tasks" in line and "parallel" in line

    def test_a_missing_probability_still_reads_as_a_sentence(self):
        line = ts.hint_line({"choice": ts.CHOICE_DELEGATE, "p": None})
        assert line.startswith("Jev suggests:") and line.endswith("You decide.")
        assert "(" not in line, "no empty parenthetical"

    def test_an_unreadable_choice_prepends_nothing(self):
        for decided in ({}, {"choice": ""}, {"choice": "fan-out"}, {"choice": None}):
            assert ts.hint_line(decided) == ""

    def test_the_hint_is_not_localised(self):
        """Prompt input for a model, not interface copy for a person.

        A translated hint would make the advice a function of the reader's UI
        language, so the text lives in this module rather than in a catalog.
        """
        assert set(ts._HINT_TEXT) == set(ts.CHOICES)
        for text in ts._HINT_TEXT.values():
            assert text.isascii()


class TestTheBaselineArmIsTheAgentsOwnBehaviour:
    @pytest.mark.parametrize(
        "calls,expected",
        [
            (0, ts.CHOICE_SINGLE),
            (1, ts.CHOICE_DELEGATE),
            (2, ts.CHOICE_SPLIT),
            (7, ts.CHOICE_SPLIT),
        ],
    )
    def test_the_count_maps_onto_a_shape(self, calls, expected):
        assert ts.agent_choice_for(calls) == expected

    @pytest.mark.parametrize("bad", [-1, -9, None, "2", 1.5, True])
    def test_an_unusable_count_reads_as_no_spawns(self, bad):
        """Never raises: this runs while a reply is being persisted.

        ``True`` is included deliberately -- a bool is not a count, and Python
        would otherwise read it as 1 and report a delegation nobody made.
        """
        assert ts.agent_choice_for(bad) == ts.CHOICE_SINGLE

    @pytest.mark.parametrize("tool", sorted(ts.SPAWN_TOOLS))
    def test_a_core_spawn_call_counts(self, tool):
        assert ts.is_spawn_call(ts.CORE_MCP_SERVER, tool) is True

    @pytest.mark.parametrize(
        "tool",
        [
            "kirocrew-core___spawn_run",
            "mcp__kirocrew-core__spawn_sub_agents",
        ],
    )
    def test_a_server_qualified_name_counts(self, tool):
        """Transports disagree on the separator; both shipped spellings resolve."""
        assert ts.is_spawn_call(ts.CORE_MCP_SERVER, tool) is True

    @pytest.mark.parametrize(
        "server,tool",
        [
            # A shell call: no MCP server, a canonical name of its own.
            ("", "execute_bash"),
            # A third-party server exposing a same-named tool.
            ("other-server", "spawn_run"),
            ("", "spawn_run"),
            # A crafted tail that is not a >= 2 underscore separator.
            (ts.CORE_MCP_SERVER, "do_spawn_run"),
            (ts.CORE_MCP_SERVER, "a/b/spawn_run"),
            # A core tool that is not a spawn.
            (ts.CORE_MCP_SERVER, "send_message"),
            (ts.CORE_MCP_SERVER, ""),
        ],
    )
    def test_nothing_else_can_move_the_count(self, server, tool):
        """The count is the arm the suggestion is scored against.

        So it is read from the trusted ``_meta.kiro`` identity alone, and absent
        identity fails closed -- which here means not counted.
        """
        assert ts.is_spawn_call(server, tool) is False


class TestTheRequestIsBoundedAndTheHistoryIsCeilinged:
    @pytest.mark.asyncio
    async def test_the_message_is_clipped_to_the_shared_bound(self, answer):
        asked = answer(ts.CHOICE_SINGLE)
        await ts.suggest("x" * (ts.MAX_MESSAGE_CHARS + 500))
        assert len(asked[0]["state"]["message"]) == ts.MAX_MESSAGE_CHARS
        assert ts.message_chars("x" * 5000) == ts.MAX_MESSAGE_CHARS

    @pytest.mark.asyncio
    async def test_the_shipped_ceiling_sends_the_request_alone(self, answer):
        asked = answer(ts.CHOICE_SINGLE, p=0.5)
        await ts.suggest("do the thing", history=_turns(("user", "earlier"), ("assistant", "ok")))
        assert "history" not in asked[0]["state"], "omitted, never an empty list"
        assert asked[0]["kwargs"]["extra"]["history_chars"] == 0

    @pytest.mark.asyncio
    async def test_a_raised_ceiling_buys_prior_turns(self, answer, monkeypatch):
        monkeypatch.setattr("kiro_crew.decisions.history_budget_chars", lambda: 500)
        asked = answer(ts.CHOICE_SPLIT)
        await ts.suggest("do the thing", history=_turns(("user", "earlier"), ("assistant", "ok")))
        state = asked[0]["state"]
        assert [row["text"] for row in state["history"]] == ["ok", "earlier"], "newest first"
        assert asked[0]["kwargs"]["extra"]["history_chars"] == len("ok") + len("earlier")

    @pytest.mark.asyncio
    async def test_the_ceiling_is_read_before_the_history_is_used(self, answer):
        """At 0 the caller's rows contribute nothing, whatever it passed."""
        asked = answer(ts.CHOICE_SINGLE)
        await ts.suggest("do the thing", history=_turns(("user", "y" * 4000)))
        assert "history" not in asked[0]["state"]

    def test_the_history_walk_is_the_one_skills_select_owns(self):
        """One spender of the consented ceiling, not two nearly-equal walks."""
        assert ts.MAX_HISTORY_MESSAGES == ss.MAX_HISTORY_MESSAGES
        assert ts.MAX_MESSAGE_CHARS == ss.MAX_MESSAGE_CHARS

    @pytest.mark.asyncio
    async def test_a_secret_in_a_prior_turn_refuses_the_whole_request(self, monkeypatch):
        """The state is what the gate renders, so history is inside the scrub."""
        monkeypatch.setattr("kiro_crew.decisions.history_budget_chars", lambda: 500)
        from kiro_crew.decisions import gate

        state = ts.build_state(
            "do the thing", _turns(("user", f"key {_AWS_KEY}")), history_budget_chars=500
        )
        assert gate.scrub_reason(state, ts.questions()) == gate.ERROR_SCRUBBED_CREDENTIAL


class TestTheWaitIsTheShapeEveryPointUses:
    def test_the_three_values_are_held_equal_across_the_package(self):
        """No point invents its own ceiling on the turn's critical path."""
        assert ts.WAIT_MARGIN_SECS == ss.WAIT_MARGIN_SECS
        assert ts.MIN_WAIT_SECS == ss.MIN_WAIT_SECS
        assert ts.MAX_WAIT_SECS == ss.MAX_WAIT_SECS

    def test_a_hand_edited_timeout_cannot_hold_the_reply(self, monkeypatch):
        monkeypatch.setattr("kiro_crew.decisions.timeout_secs", lambda: 3600.0)
        assert ts.wait_budget() == ts.MAX_WAIT_SECS

    def test_an_unreadable_budget_is_the_floor(self, monkeypatch):
        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr("kiro_crew.decisions.timeout_secs", _boom)
        assert ts.wait_budget() == ts.MIN_WAIT_SECS

    def test_a_non_finite_budget_is_the_floor(self, monkeypatch):
        monkeypatch.setattr("kiro_crew.decisions.timeout_secs", lambda: float("inf"))
        assert ts.wait_budget() == ts.MIN_WAIT_SECS

    def test_an_unreadable_ceiling_sends_no_prior_turns(self, monkeypatch):
        def _boom():
            raise RuntimeError("no keystone")

        monkeypatch.setattr("kiro_crew.decisions.history_budget_chars", _boom)
        assert ts.history_budget() == 0


class TestTheRowHoldsBothArms:
    @pytest.mark.asyncio
    async def test_the_outcome_row_names_both_choices_and_the_count(self, written_rows):
        decided = {"turn_id": "ts-1", "choice": ts.CHOICE_SPLIT, "p": 0.84, "latency_ms": 190}
        row = await ts.record_outcome(session_key="s", decided=decided, spawn_calls=2)
        assert row is not None
        assert row["point"] == ts.POINT
        assert row["jev_choice"] == ts.CHOICE_SPLIT
        assert row["agent_choice"] == ts.CHOICE_SPLIT
        assert row["spawn_calls"] == 2
        assert row["agree"] is True
        assert row["p"] == pytest.approx(0.84)
        assert row["latency_ms"] == 190, "a core field, carried at top level"
        assert written_rows == [row], "the row returned IS the row written"

    @pytest.mark.asyncio
    async def test_a_suggestion_the_agent_ignored_reads_as_a_difference(self, written_rows):
        decided = {"turn_id": "ts-2", "choice": ts.CHOICE_SPLIT, "p": 0.6, "latency_ms": 12}
        row = await ts.record_outcome(session_key="s", decided=decided, spawn_calls=0)
        assert row["agent_choice"] == ts.CHOICE_SINGLE
        assert row["agree"] is False

    def test_agree_is_derived_from_the_two_choices(self):
        """Never asserted by a caller: a flag that disagreed with the words beside
        it would hide a real divergence behind one of them."""
        built = ts.build_outcome({"turn_id": "t", "choice": ts.CHOICE_DELEGATE}, 1)
        assert built["agree"] is True
        built = ts.build_outcome({"turn_id": "t", "choice": ts.CHOICE_DELEGATE}, 4)
        assert built["agree"] is False

    def test_latency_is_not_an_extra(self):
        """It is a core row field, so an ``extra`` naming it would be dropped."""
        assert "latency_ms" not in ts.build_outcome({"choice": ts.CHOICE_SINGLE}, 0)

    @pytest.mark.asyncio
    async def test_a_row_the_writer_refused_stamps_nothing(self, monkeypatch):
        monkeypatch.setattr(log_mod, "append", lambda row: False)
        decided = {"turn_id": "ts-3", "choice": ts.CHOICE_SINGLE, "p": 0.5, "latency_ms": 1}
        assert await ts.record_outcome(session_key="s", decided=decided, spawn_calls=0) is None

    @pytest.mark.asyncio
    async def test_a_raising_writer_stamps_nothing(self, monkeypatch):
        def _boom(row):
            raise OSError("sealed")

        monkeypatch.setattr(log_mod, "append", _boom)
        decided = {"turn_id": "ts-4", "choice": ts.CHOICE_SINGLE, "p": 0.5, "latency_ms": 1}
        assert await ts.record_outcome(session_key="s", decided=decided, spawn_calls=0) is None

    @pytest.mark.asyncio
    async def test_a_write_past_its_budget_stamps_nothing(self, monkeypatch):
        monkeypatch.setattr(ts, "LOG_BUDGET_SECS", 0.01)

        def _slow(row):
            import time as _t

            _t.sleep(0.5)
            return True

        monkeypatch.setattr(log_mod, "append", _slow)
        decided = {"turn_id": "ts-5", "choice": ts.CHOICE_SINGLE, "p": 0.5, "latency_ms": 1}
        assert await ts.record_outcome(session_key="s", decided=decided, spawn_calls=0) is None


class TestThePointIsRegistered:
    def test_the_gate_knows_the_name(self):
        from kiro_crew.decisions import DECISION_POINT_NAMES

        assert ts.POINT in DECISION_POINT_NAMES

    def test_it_needs_no_tool_argument_scope(self):
        """It sends a message excerpt and prior turns -- the category the main
        consent already names -- so it is not a second consent."""
        from kiro_crew.decisions.gate import POINTS_NEEDING_TOOL_ARGS

        assert ts.POINT not in POINTS_NEEDING_TOOL_ARGS
