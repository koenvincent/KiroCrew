"""``compaction.keep`` -- the shadow point that scores an automatic compaction.

Four properties this file pins, because each of them is a way the measurement could
be wrong rather than merely absent:

* the FITTING ladder picks the first stage under the ceiling, and a transcript that
  fits nowhere is recorded rather than sent;
* the redaction runs BEFORE the clip, and no tool RESULT byte is on the wire at any
  stage;
* the questions are batched, every batch carries the SAME state, and a partial
  answer publishes nothing;
* the per-call answers map onto the three options and onto the two character arms.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kiro_crew.decisions import consent as consent_mod
from kiro_crew.decisions import gate as gate_mod
from kiro_crew.decisions.points import compaction_keep as point
from kiro_crew.decisions.types import Answer, Choice

SECRET = "AKIAIOSFODNN7EXAMPLE"


def tool_row(index: int, *, tool: str = "execute_bash", inp: str = "ls", out: str = "ok") -> dict:
    return {
        "role": "tool",
        "content": f"\U0001f527 {tool}",
        "meta": {"tool_call_id": f"tc{index}", "input": inp, "output": out},
    }


def transcript(calls: int = 3, *, inp: str = "ls -la", out: str = "a" * 100) -> list[dict]:
    """A transcript with *calls* tool rows and a conversation around them.

    Deliberately longer than ``PINNED_TAIL_ROWS`` so the pinning split is exercised
    rather than making every row pinned.
    """
    rows: list[dict] = [{"role": "user", "content": "please do the thing"}]
    for index in range(calls):
        rows.append(tool_row(index, inp=inp, out=out))
        rows.append({"role": "assistant", "content": f"step {index} done"})
    return rows


@pytest.fixture(autouse=True)
def _no_leftover_records():
    point.reset_records()
    yield
    point.reset_records()


class TestTranscriptReading:
    def test_tool_rows_carry_their_real_input_and_result_sizes(self):
        built = point.build_transcript(transcript(calls=2, inp="x" * 30, out="y" * 70))
        assert [call.input_chars for call in built.calls] == [30, 30]
        assert [call.result_chars for call in built.calls] == [70, 70]

    def test_chars_today_counts_only_what_the_replay_keeps(self):
        # The comparison arm is today's recycle replay, which admits user/assistant/
        # inject and drops every tool row. A chars_today that counted tool traffic
        # would make the two arms describe the same thing.
        rows = [
            {"role": "user", "content": "abcde"},
            tool_row(0, inp="i" * 100, out="o" * 100),
            {"role": "assistant", "content": "fgh"},
            {"role": "chunk", "content": "ignored entirely"},
        ]
        built = point.build_transcript(rows)
        assert built.chars_today == len("abcde") + len("fgh")
        assert built.chars_all == len("abcde") + len("fgh") + 200

    def test_first_row_and_the_newest_six_are_pinned(self):
        rows = [tool_row(index) for index in range(12)]
        built = point.build_transcript(rows)
        pinned = [call.index for call in built.calls if call.pinned]
        assert pinned == [0, 6, 7, 8, 9, 10, 11]

    def test_a_pinned_call_is_never_asked_about(self):
        built = point.build_transcript([tool_row(index) for index in range(12)])
        asked = {question.id for question in point.questions_for(built.calls)}
        assert asked == {f"call_{index}" for index in range(1, 6)}

    def test_an_unreadable_row_is_skipped_rather_than_raising(self):
        rows = [{"role": "user", "content": "ok"}, "not a row", {"role": "tool", "meta": 7}]
        built = point.build_transcript(rows)  # type: ignore[arg-type]
        assert len(built.calls) == 1
        assert built.calls[0].input_chars == 0

    def test_the_call_count_is_bounded(self, monkeypatch):
        monkeypatch.setattr(point, "MAX_CALLS", 4)
        built = point.build_transcript([tool_row(index) for index in range(40)])
        assert len(built.calls) == 4


class TestStateShape:
    def test_no_tool_result_byte_is_sent_at_any_stage(self):
        built = point.build_transcript(transcript(calls=4, out="TOPSECRETOUTPUT" * 20))
        for stage in point.STAGES:
            rendered = json.dumps(point.build_state(built, stage))
            assert "TOPSECRETOUTPUT" not in rendered
            assert "(omitted)" in rendered

    def test_a_result_is_replaced_by_its_own_size(self):
        built = point.build_transcript(transcript(calls=1, out="z" * 4096))
        state = point.build_state(built, point.STAGES[0])
        assert state["calls"][0]["result"] == "ok, 4096 chars (omitted)"

    def test_redaction_runs_before_the_clip(self):
        # The clip is 6 characters and the secret starts at offset 0, so a clip-first
        # implementation would send a 6-character fragment of the key that neither
        # redactor matches. Redaction-first replaces it, and the placeholder is what
        # gets clipped.
        built = point.build_transcript(transcript(calls=1, inp=SECRET))
        stage = point.Stage("tiny", 6, 1.0)
        sent = point.build_state(built, stage)["calls"][0]["input"]
        assert SECRET[:6] not in sent
        assert SECRET not in sent

    def test_a_pinned_call_says_so_in_the_state(self):
        built = point.build_transcript([tool_row(index) for index in range(12)])
        state = point.build_state(built, point.STAGES[0])
        assert state["calls"][0].get("pinned") is True
        assert "pinned" not in state["calls"][3]

    def test_conversation_text_is_verbatim_until_a_stage_cuts_it(self):
        rows = [{"role": "user", "content": "q" * 400}, tool_row(0)]
        built = point.build_transcript(rows)
        assert point.build_state(built, point.STAGES[0])["messages"][0]["text"] == "q" * 400
        halved = point.build_state(built, point.STAGES[3])["messages"][0]["text"]
        assert len(halved) < 400
        assert halved.startswith("q") and halved.endswith("q")

    def test_one_line_stage_leaves_no_newline_in_an_input(self):
        built = point.build_transcript(transcript(calls=1, inp="a\nb\nc"))
        state = point.build_state(built, point.STAGES[-1])
        assert "\n" not in state["calls"][0]["input"]


class TestFitting:
    def test_the_first_stage_under_the_ceiling_wins(self):
        built = point.build_transcript(transcript(calls=2, inp="i" * 50))
        fitted = point.fit_state(built)
        assert fitted is not None
        assert fitted[1].name == point.STAGES[0].name

    def test_the_ladder_shrinks_monotonically(self):
        # Each rung must be no larger than the one above it, or "the first stage that
        # fits" would not be the mildest one that fits.
        built = point.build_transcript(transcript(calls=8, inp="i" * 900))
        sizes = [point.estimated_tokens(point.build_state(built, s)) for s in point.STAGES]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[-1] < sizes[0]

    def test_a_large_transcript_walks_down_the_ladder(self, monkeypatch):
        # The ceiling is derived from the SECOND rung's own measurement rather than
        # guessed, so the test says "the first rung is too big and the second is not"
        # without encoding today's character counts.
        built = point.build_transcript(transcript(calls=8, inp="i" * 900))
        second = point.estimated_tokens(point.build_state(built, point.STAGES[1]))
        monkeypatch.setattr(point, "STATE_TOKEN_CEILING", second)
        fitted = point.fit_state(built)
        assert fitted is not None
        assert fitted[1].name == point.STAGES[1].name

    def test_a_transcript_that_fits_nowhere_is_refused(self, monkeypatch):
        monkeypatch.setattr(point, "STATE_TOKEN_CEILING", 1)
        built = point.build_transcript(transcript(calls=40))
        assert point.fit_state(built) is None


class TestBatching:
    def test_questions_split_into_bounded_requests_in_order(self):
        asked = [Choice(f"call_{n}", point.PROMPT, options=list(point.OPTIONS)) for n in range(57)]
        groups = point.batches(asked, 25)
        assert [len(group) for group in groups] == [25, 25, 7]
        assert [q.id for q in groups[0]][:2] == ["call_0", "call_1"]

    def test_every_question_carries_the_three_options(self):
        built = point.build_transcript(transcript(calls=9))
        for question in point.questions_for(built.calls):
            assert question.options == [point.OPTION_DROP, point.OPTION_CALL, point.OPTION_BOTH]


class TestAnswerReading:
    def _answers(self, mapping):
        return {qid: Answer(id=qid, value=value, p=0.9) for qid, value in mapping.items()}

    def test_a_complete_batch_reads_back(self):
        asked = [Choice("call_1", point.PROMPT, options=list(point.OPTIONS))]
        read = point.read_decisions(self._answers({"call_1": point.OPTION_BOTH}), asked)
        assert read == {"call_1": (point.OPTION_BOTH, 0.9)}

    def test_a_batch_missing_one_answer_is_unusable_rather_than_partial(self):
        # All-or-nothing per batch: a mapping folded with a gap would be counted into
        # a fraction over calls nobody answered about.
        asked = [
            Choice("call_1", point.PROMPT, options=list(point.OPTIONS)),
            Choice("call_2", point.PROMPT, options=list(point.OPTIONS)),
        ]
        assert point.read_decisions(self._answers({"call_1": point.OPTION_DROP}), asked) is None

    def test_an_option_outside_the_domain_is_unusable(self):
        asked = [Choice("call_1", point.PROMPT, options=list(point.OPTIONS))]
        assert point.read_decisions(self._answers({"call_1": "keep-everything"}), asked) is None

    def test_no_answers_at_all_is_unusable(self):
        asked = [Choice("call_1", point.PROMPT, options=list(point.OPTIONS))]
        assert point.read_decisions(None, asked) is None


class TestTally:
    def test_each_option_lands_on_its_own_counter_and_its_own_chars(self):
        rows = [
            {"role": "user", "content": "u" * 10},
            tool_row(0, inp="i" * 100, out="o" * 1000),
            tool_row(1, inp="i" * 100, out="o" * 1000),
            tool_row(2, inp="i" * 100, out="o" * 1000),
        ]
        built = point.build_transcript(rows)
        # Row 0 is the user message, so rows 1-3 are the calls; with four rows every
        # one of them is inside the pinned tail, so pin nothing by hand and instead
        # assert against the pinning the walk actually produced.
        decisions = {
            call.question_id: (option, 0.5)
            for call, option in zip(
                [c for c in built.calls if not c.pinned],
                [point.OPTION_BOTH, point.OPTION_CALL, point.OPTION_DROP],
            )
        }
        counts = point.tally(built, decisions)
        assert counts["total_calls"] == 3
        assert counts["kept_both"] + counts["kept_call"] + counts["dropped"] == 3
        assert counts["chars_today"] == 10
        assert counts["chars_jev"] > counts["chars_today"]
        assert counts["chars_jev"] <= counts["chars_all"]

    def test_a_pinned_call_counts_as_kept_whole(self):
        built = point.build_transcript([tool_row(index, inp="i", out="o") for index in range(12)])
        decisions = {
            call.question_id: (point.OPTION_DROP, 0.5) for call in built.calls if not call.pinned
        }
        counts = point.tally(built, decisions)
        assert counts["pinned_calls"] == 7
        assert counts["kept_both"] == 7
        assert counts["dropped"] == 5

    def test_dropping_everything_leaves_the_two_arms_equal(self):
        rows = [{"role": "user", "content": "u" * 10}] + [
            tool_row(index, inp="i" * 5, out="o" * 5) for index in range(9)
        ]
        built = point.build_transcript(rows)
        for call in built.calls:
            call.pinned = False
        decisions = {call.question_id: (point.OPTION_DROP, 0.1) for call in built.calls}
        counts = point.tally(built, decisions)
        assert counts["chars_jev"] == counts["chars_today"] == 10


class TestRun:
    """The whole coroutine, with the gate's ``decide`` replaced by a stub.

    ``decide`` is the seam every refusal funnels through, so stubbing it is what lets
    these tests be about the POINT rather than about consent, sampling or transport.
    """

    def _enable(self, monkeypatch):
        monkeypatch.setattr(point.core, "is_enabled", lambda *_a, **_kw: True)
        monkeypatch.setattr(point, "wait_budget", lambda: 30.0)

    def _rows(self, monkeypatch, rows):
        monkeypatch.setattr(point, "read_rows", lambda _key: rows)

    def _appended(self, monkeypatch) -> list[dict]:
        written: list[dict] = []

        def _append(row):
            written.append(row)
            return True

        monkeypatch.setattr(point._log, "append", _append)
        return written

    def _answer_all(self, monkeypatch, option: str, *, seen: list | None = None):
        async def _decide(_p, state, questions, **_kw):
            if seen is not None:
                seen.append((state, [q.id for q in questions]))
            return {q.id: Answer(id=q.id, value=option, p=0.8) for q in questions}

        monkeypatch.setattr(point.core, "decide", _decide)

    def test_a_complete_run_writes_one_outcome_row_and_publishes_it(self, monkeypatch):
        self._enable(monkeypatch)
        self._rows(monkeypatch, transcript(calls=9))
        written = self._appended(monkeypatch)
        self._answer_all(monkeypatch, point.OPTION_CALL)
        record = asyncio.run(point.score_compaction("sess"))
        assert record is not None
        assert record["point"] == point.POINT
        assert record["error"] is None
        assert record["fitting_stage"] == point.STAGES[0].name
        assert len(written) == 1
        assert point.take_record("sess") == record

    def test_every_batch_carries_the_same_state(self, monkeypatch):
        monkeypatch.setattr(point, "QUESTIONS_PER_REQUEST", 2)
        self._enable(monkeypatch)
        self._rows(monkeypatch, transcript(calls=12))
        self._appended(monkeypatch)
        seen: list = []
        self._answer_all(monkeypatch, point.OPTION_BOTH, seen=seen)
        record = asyncio.run(point.score_compaction("sess"))
        assert record is not None
        assert record["requests"] == len(seen) > 1
        first = json.dumps(seen[0][0], sort_keys=True)
        assert all(json.dumps(state, sort_keys=True) == first for state, _ids in seen)
        # Every non-pinned call is asked about exactly once across the batches.
        asked = [qid for _state, ids in seen for qid in ids]
        assert len(asked) == len(set(asked))

    def test_a_partial_answer_is_recorded_and_never_published(self, monkeypatch):
        monkeypatch.setattr(point, "QUESTIONS_PER_REQUEST", 2)
        self._enable(monkeypatch)
        self._rows(monkeypatch, transcript(calls=12))
        written = self._appended(monkeypatch)
        calls = {"n": 0}

        async def _decide(_p, _state, questions, **_kw):
            calls["n"] += 1
            if calls["n"] == 2:
                return None
            return {q.id: Answer(id=q.id, value=point.OPTION_BOTH, p=0.7) for q in questions}

        monkeypatch.setattr(point.core, "decide", _decide)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert [row["error"] for row in written] == [point.ERROR_PARTIAL]
        assert point.take_record("sess") is None

    def test_no_answer_at_all_writes_nothing_of_its_own(self, monkeypatch):
        # The gate has already written a row per refused request carrying its own
        # error category, so a second row here would say nothing new.
        self._enable(monkeypatch)
        self._rows(monkeypatch, transcript(calls=9))
        written = self._appended(monkeypatch)

        async def _decide(*_a, **_kw):
            return None

        monkeypatch.setattr(point.core, "decide", _decide)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert written == []

    def test_a_state_that_fits_nowhere_is_recorded_and_nothing_is_sent(self, monkeypatch):
        monkeypatch.setattr(point, "STATE_TOKEN_CEILING", 1)
        self._enable(monkeypatch)
        self._rows(monkeypatch, transcript(calls=9))
        written = self._appended(monkeypatch)
        sent = {"n": 0}

        async def _decide(*_a, **_kw):
            sent["n"] += 1
            return None

        monkeypatch.setattr(point.core, "decide", _decide)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert sent["n"] == 0
        assert [row["error"] for row in written] == [point.ERROR_TOO_LARGE]

    def test_the_seam_being_off_reads_no_transcript_at_all(self, monkeypatch):
        # The keystone read is the cheap refusal and goes first: an unconsented
        # machine must not read a megabyte of transcript to learn it sends nothing.
        monkeypatch.setattr(point.core, "is_enabled", lambda *_a, **_kw: False)
        read = {"n": 0}

        def _read(_key):
            read["n"] += 1
            return transcript()

        monkeypatch.setattr(point, "read_rows", _read)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert read["n"] == 0

    def test_a_transcript_with_no_tool_calls_is_not_a_measurement(self, monkeypatch):
        self._enable(monkeypatch)
        self._rows(monkeypatch, [{"role": "user", "content": "hello"}])
        written = self._appended(monkeypatch)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert written == []

    def test_a_run_that_outlives_its_budget_records_the_timeout(self, monkeypatch):
        self._enable(monkeypatch)
        monkeypatch.setattr(point, "wait_budget", lambda: 0.01)
        self._rows(monkeypatch, transcript(calls=9))
        written = self._appended(monkeypatch)

        async def _slow(*_a, **_kw):
            await asyncio.sleep(5)

        monkeypatch.setattr(point.core, "decide", _slow)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert [row["error"] for row in written] == [point.ERROR_RUN_TIMEOUT]

    def test_a_refused_row_is_never_published(self, monkeypatch):
        self._enable(monkeypatch)
        self._rows(monkeypatch, transcript(calls=9))
        monkeypatch.setattr(point._log, "append", lambda _row: False)
        self._answer_all(monkeypatch, point.OPTION_BOTH)
        assert asyncio.run(point.score_compaction("sess")) is None
        assert point.take_record("sess") is None


class TestRecordStore:
    def test_a_record_is_claimed_once(self):
        point.publish_record("s", {"turn_id": "t"})
        assert point.take_record("s") == {"turn_id": "t"}
        assert point.take_record("s") is None

    def test_a_newer_record_replaces_the_one_nobody_read(self):
        point.publish_record("s", {"turn_id": "old"})
        point.publish_record("s", {"turn_id": "new"})
        assert point.take_record("s") == {"turn_id": "new"}

    def test_an_expired_record_is_not_handed_to_a_later_compaction(self, monkeypatch):
        monkeypatch.setattr(point, "RECORD_TTL_SECONDS", 0.0)
        point.publish_record("s", {"turn_id": "t"})
        assert point.take_record("s") is None

    def test_the_store_is_bounded(self, monkeypatch):
        monkeypatch.setattr(point, "MAX_PENDING_RECORDS", 3)
        for n in range(10):
            point.publish_record(f"s{n}", {"turn_id": str(n)})
        assert point.pending_count() == 3

    def test_a_keyless_record_is_dropped_rather_than_filed_under_nothing(self):
        point.publish_record("", {"turn_id": "t"})
        assert point.pending_count() == 0
        assert point.take_record("") is None


class TestConsentScope:
    """The third scope, held against the gate rather than against the point.

    A point can only be inert for the right reason if the REFUSAL is the gate's, so
    these drive ``gate`` directly.
    """

    def test_the_point_is_registered(self):
        assert point.POINT in gate_mod.DECISION_POINT_NAMES
        assert point.POINT in gate_mod.POINTS_NEEDING_COMPACTION

    def test_an_absent_scope_refuses_the_point(self):
        assert gate_mod._scope_consented(point.POINT, {}) is False

    def test_the_tool_argument_scope_does_not_grant_this_one(self):
        # The narrower yes must never read as the wider one: tool_args was reviewed as
        # the arguments of the one call about to run.
        state = {consent_mod.STATE_KEY_TOOL_ARGS: True}
        assert gate_mod._scope_consented(point.POINT, state) is False
        assert gate_mod._scope_consented("tool.risk", state) is True

    def test_only_a_literal_true_consents(self):
        for value in ("true", 1, "yes", None):
            assert (
                gate_mod._scope_consented(point.POINT, {consent_mod.STATE_KEY_COMPACTION: value})
                is False
            )
        assert (
            gate_mod._scope_consented(point.POINT, {consent_mod.STATE_KEY_COMPACTION: True}) is True
        )

    def test_a_point_needing_no_scope_is_untouched(self):
        assert gate_mod._scope_consented("skills.select", {}) is True
        assert gate_mod._scope_consented(None, {}) is True
