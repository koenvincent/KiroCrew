"""``compaction.keep`` -- at an AUTO compaction, which tool calls would Jev keep?

A SHADOW measurement, and nothing else. Jev decides nothing: the compaction that
triggered this point runs exactly as it does without the seam, on every backend,
and the answer becomes a row in the decision log plus one line on the compaction
card. :func:`score_compaction` has no apply path and no return value a caller
branches on -- that is what makes it safe to place on the compaction path at all.
What it buys is the number the live design cannot be argued about without: how
much of a real transcript an oracle would keep, against what today's recycle
replay keeps.

Why this point exists in shadow rather than live
-----------------------------------------------
Nothing in Kiro Crew can replace a harness's message list -- no backend offers a
set-messages call, and ``/compact`` summarizes inside the harness process. The one
place a keep-list COULD land is the transcript Crew re-injects after a recycle
(``context.build_session_replay``), and today that replay admits ``user`` /
``assistant`` / ``inject`` rows only (``context.RECALL_ROLES``): every tool call
and every tool result is already dropped. So a live keep-list would only ever
DISPLACE text inside the same 80 000-character budget, and whether that trade is
worth making is a question about data nobody has yet. This point produces that
data and changes no byte of the replay.

Where it runs, and where it deliberately does not
-------------------------------------------------
``session_compaction.CompactionCoordinator._compact_session``, at the top, as a
background task. That method is the ONE funnel both automatic entry points reach
-- the per-turn threshold trigger (``check_context_usage`` ->
``_trigger_compaction``) and the awaited between-turn one (``compact_if_needed``)
-- and it is reached BEFORE the Claude arm, before ``_recycle_unmanaged`` and
before ``_compact_in_place``, so every backend is covered and the scoring never
sits between the trigger and the compaction.

A MANUAL ``/compact`` is excluded, and by construction rather than by a flag: the
dashboard dispatches that command as an ordinary turn through
``provider.stream_command("/compact")`` and never calls ``_compact_session``, so
there is no manual path for this point to be on. A person who typed the command
is not being measured.

It cannot delay or alter the compaction
---------------------------------------
The task is created and NOT awaited. The compaction proceeds on its own
coroutine, and the only thing the deadline below decides is whether this task
gives up. When it does, it logs and publishes nothing, which is the same outcome
as "the seam is off": no card line. So the worst case this point can produce is a
missing observation.

What leaves the machine, and the third consent that authorizes it
-----------------------------------------------------------------
The whole slot transcript, at the fitting stage below: user and assistant text,
every tool-call input redacted and clipped, and every tool RESULT replaced by
``ok, N chars (omitted)`` -- results are never sent, only their sizes. That is a
category neither of the keystone's existing scopes covers: ``tool_args`` was
reviewed as the arguments of the one call about to run, not as everything the
session has run, in a request one to two orders of magnitude larger. So this point
needs a THIRD scope, ``consent.STATE_KEY_COMPACTION``, which absent reads as NOT
consented (``gate.POINTS_NEEDING_COMPACTION``, refused by
``gate._scope_consented``). An install that consented before this scope existed --
including one that granted ``tool_args`` -- is INERT here rather than retroactively
signed up.

Fitting, and the ~8% that never fits
------------------------------------
A real transcript is far larger than an oracle state: 87% of it is tool traffic,
and the p50 session at the 70% compaction threshold carries 198 tool calls over
510 000 characters. :data:`STAGES` walks the spike's measured ladder -- inputs at
1000, then 200, then 60 characters, then the conversation text cut to head+tail
halves, then calls rendered on one line -- and the FIRST stage under
:data:`STATE_TOKEN_CEILING` is the one sent. About one session in twelve does not
fit even at the last stage; that one writes an :data:`ERROR_TOO_LARGE` row and
sends nothing, so the population that cannot be measured is visible instead of
silently absent.

Batching, because the questions outnumber the state
---------------------------------------------------
One ``Choice`` per non-pinned call is hundreds of questions on a real transcript,
and one request carrying all of them would exceed the provider's own ceiling. They
are split into requests of :data:`QUESTIONS_PER_REQUEST`, each carrying the SAME
state, run at most :data:`MAX_CONCURRENT_REQUESTS` at a time. Every request is an
ordinary ``decide`` call, so each one is scrubbed, bounded and logged by the gate
on its own.

A PARTIAL answer publishes nothing. The card line is a fraction -- "23 of 61" --
and a fraction over a denominator some of whose members were never asked about is
a false number, not an incomplete one. The outcome row is still written, with
:data:`ERROR_PARTIAL`, so the refusal rate is measurable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kiro_crew import decisions as core
from kiro_crew.decisions import log as _log
from kiro_crew.decisions.types import Answer, Choice, Question

logger = logging.getLogger(__name__)

POINT = "compaction.keep"

#: The three outcomes of the reference design, which are this question's whole
#: closed domain: drop the call entirely, keep the call without its result, or keep
#: both. The gate refuses an answer outside a question's declared options, so a
#: provider inventing a fourth reads as an invalid result rather than as a branch
#: nobody wrote.
OPTION_DROP = "drop"
OPTION_CALL = "call"
OPTION_BOTH = "both"
OPTIONS: tuple[str, ...] = (OPTION_DROP, OPTION_CALL, OPTION_BOTH)

#: Prefix of a question's id, completed by the call's own index in the state, so an
#: answer names one call and a reader can find it in the request that was sent.
QUESTION_PREFIX = "call_"

#: The rubric, sent as every question's prompt. It names the CONSEQUENCE of each
#: option rather than the option's own word, because the words alone ("call",
#: "both") say nothing about what a reader loses.
PROMPT = (
    "This conversation is about to be compacted, and everything not kept is "
    "forgotten. For this one tool call, what is worth keeping so the assistant can "
    f"carry on? Answer {OPTION_DROP} if neither the call nor its result matters "
    f"any more. Answer {OPTION_CALL} if knowing that the call was made, and with "
    f"what, still matters but its output does not. Answer {OPTION_BOTH} if the "
    "output itself is still needed."
)

#: Estimated tokens one request's STATE may reach. The provider's own request
#: ceiling is larger; this is the state's share of it, leaving room for the
#: questions batched beside it. An estimate, not a count: see
#: :data:`CHARS_PER_TOKEN`.
STATE_TOKEN_CEILING = 25_000

#: Characters per token, the same rough divisor ``skills.select`` estimates
#: ``tokens_saved`` with. A real tokenizer is not worth importing for a fitting
#: decision whose next stage is a step change, and a wrong estimate costs one
#: stage, never a refusal.
CHARS_PER_TOKEN = 4

#: Transcript rows the state is built from. Everything else a transcript carries --
#: ``chunk`` (unflushed streaming text the durable rows already hold),
#: ``permission``, ``done``, the system notices -- says nothing about which tool
#: calls matter.
#:
#: ``thinking`` is NOT here, the same exclusion ``message.steer`` makes: private
#: reasoning is the largest and least reviewed text a turn produces, and the
#: question is answerable without it.
STATE_ROLES = frozenset({"user", "assistant", "tool"})

#: The one role a tool call arrives under, whose ``meta`` carries the join key, the
#: input and the result (``chat_runner._tool_call_meta``).
TOOL_ROLE = "tool"

#: Rows pinned into the state without a question: the FIRST row of the transcript
#: -- the request the whole conversation answers -- and the newest
#: :data:`PINNED_TAIL_ROWS`, which are what the agent is working on right now.
#: Pinning them is the reference's own rule, and it keeps the question off the
#: calls whose answer is not in doubt.
PINNED_TAIL_ROWS = 6

#: Rows the walk will even look at, newest last. The fitting stages bound what is
#: SENT; this bounds what is READ, so a session with tens of thousands of rows
#: cannot turn one compaction into an unbounded scan.
MAX_ROWS = 4000

#: Tool calls one compaction may ask about. Past this the state is not the problem
#: -- the request count is -- and a measurement that costs forty provider round
#: trips is not worth the compaction it is observing.
MAX_CALLS = 1000

#: Longest tool name sent or recorded. A name is a title the harness chose, so this
#: is a bound rather than a truncation anyone should see.
MAX_TOOL_CHARS = 120

#: Questions one request carries. The state is at most
#: :data:`STATE_TOKEN_CEILING`; each question adds the shared rubric plus three
#: short options, so this many keeps a request inside the provider's own ceiling
#: with room to spare. The gate's own row records at most eight answers
#: (``log._MAX_ANSWERS``) whatever this is, which is why the tallies live on the
#: outcome row rather than being recounted from the request rows.
QUESTIONS_PER_REQUEST = 25

#: Requests in flight at once. The whole run is a background observation, so this
#: is low deliberately: a shadow measurement must not be the reason a provider
#: rate-limits the point that is deciding something.
MAX_CONCURRENT_REQUESTS = 4

#: Scheduling slack on top of the provider budget, and the floor and ceiling the
#: wait is clamped into. Held equal to the values every other point in this package
#: waits by, so one shape covers all of them rather than each inventing its own.
#: The ceiling is what bounds the WHOLE run here: a transcript needing ten requests
#: gets the same ten seconds one request does, and a run that outlives it publishes
#: nothing.
WAIT_MARGIN_SECS = 0.5
MIN_WAIT_SECS = 0.25
MAX_WAIT_SECS = 10.0

#: How long the outcome row's write may hold this task, on top of the provider
#: budget. Named and valued as ``gate._LOG_BUDGET_SECS`` is, because it is the same
#: write on the same loop.
LOG_BUDGET_SECS = 0.05

#: Row ``error`` when no fitting stage brought the state under the ceiling. Written
#: instead of a silent return, because the share of sessions that cannot be measured
#: is one of the two numbers this whole point exists to produce.
ERROR_TOO_LARGE = "state-too-large"

#: Row ``error`` when at least one batch answered and at least one did not. The
#: tallies are still written -- they say how much was learned -- but no card line is
#: published, because a fraction whose denominator includes calls nobody was asked
#: about is a false number rather than an incomplete one.
ERROR_PARTIAL = "partial-answers"

#: Row ``error`` when the whole run outlived :func:`wait_budget`. The gate writes
#: its own row for each request it was still inside; this one says the RUN stopped,
#: which is the fact an operator raises ``timeout_ms`` on.
ERROR_RUN_TIMEOUT = "run-timeout"


@dataclass(frozen=True, slots=True)
class Stage:
    """One rung of the fitting ladder.

    *input_chars* clips each tool input; *text_share* keeps that fraction of each
    conversation message as its head and tail halves (1.0 keeps it verbatim);
    *one_line* collapses a tool input's whitespace so a call occupies one line.
    """

    name: str
    input_chars: int
    text_share: float
    one_line: bool = False


#: The fitting ladder, measured in the P3 spike over 206 real transcripts: the
#: share of sessions whose state lands under :data:`STATE_TOKEN_CEILING` runs 25%,
#: 67%, 80%, 88%, 92% down these five rungs. The FIRST one that fits is sent, so an
#: ordinary session pays the mildest stage and only a large one is cut hard.
#:
#: The order is part of the contract: inputs are clipped before conversation text is
#: touched at all, because tool inputs are 33% of a transcript's characters and the
#: text is 6%. Cutting the small half first would buy almost nothing and lose the
#: part the question is answered against.
STAGES: tuple[Stage, ...] = (
    Stage("inputs_1000", 1000, 1.0),
    Stage("inputs_200", 200, 1.0),
    Stage("inputs_60", 60, 1.0),
    Stage("text_half", 60, 0.5),
    Stage("calls_one_line", 60, 0.25, one_line=True),
)


def wait_budget() -> float:
    """How long the whole run may take, clamped into a sane window. Never raises."""
    try:
        budget = float(core.timeout_secs()) + WAIT_MARGIN_SECS
    except Exception:
        logger.debug("compaction.keep: provider budget unreadable", exc_info=True)
        return MIN_WAIT_SECS
    if not math.isfinite(budget):
        return MIN_WAIT_SECS
    return min(max(budget, MIN_WAIT_SECS), MAX_WAIT_SECS)


def redacted(text: object, limit: int | None) -> str:
    """*text* with credentials and exfiltration URLs replaced, then clipped to *limit*.

    ``limit=None`` redacts without clipping, which is what a conversation message
    gets: the brief is that user and assistant text travels verbatim, and the stage's
    own ``text_share`` is what cuts it afterwards. A NUMBER would have to be a guess
    at how much a redactor lengthens the text -- a placeholder is longer than most
    secrets -- so a message whose secret was replaced would lose real characters off
    its end to a bound that was only ever meant to be "as long as it was".

    The canonical redactors, not the gate's scanner, and the difference is the same
    one ``tool_risk.scrubbed`` states: the gate REFUSES a request carrying a
    credential, which is right for text a caller chose to send, while a credential
    in a tool argument is ORDINARY -- an ``aws`` command, a curl header -- so
    refusing on it would mean the point never fires on the transcripts most worth
    measuring. Replacing it keeps the state describable and leaves nothing to leak,
    and the gate's own scan still runs over the result.

    Clipped AFTER redaction, never before: a cut placed inside a secret leaves a
    fragment neither redactor matches, and the fragment is what would go on the
    wire. The clip is a HEAD, unlike ``message_steer.redacted``'s tail, because the
    identifying part of a tool input is its beginning -- the command, the path --
    not its end.

    Never raises: a redactor that fails yields the empty string rather than
    unredacted text, which is the only safe direction for text about to leave the
    machine.
    """
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    if not raw or (limit is not None and limit <= 0):
        return ""
    try:
        from kiro_crew.security import redact_credentials, redact_exfiltration_urls

        cleaned, _warnings = redact_credentials(raw)
        cleaned, _warnings = redact_exfiltration_urls(cleaned)
    except Exception:
        logger.debug("compaction.keep: redaction failed; dropping the field", exc_info=True)
        return ""
    return cleaned if limit is None else cleaned[:limit]


def head_tail(text: str, share: float) -> str:
    """*share* of *text*, as its head and tail halves joined by an ellipsis.

    A middle cut rather than a tail clip: the opening of a message says what was
    asked and the closing says what it settled on, and the part a reader of the
    whole conversation can least afford to lose is not in between. ``share >= 1``
    returns the text unchanged, so a stage that does not cut costs no allocation
    path of its own.
    """
    if share >= 1.0 or not text:
        return text
    keep = max(0, int(len(text) * share))
    if keep >= len(text):
        return text
    if keep == 0:
        return ""
    half = max(1, keep // 2)
    return text[:half] + " … " + text[-half:]


def result_placeholder(chars: int) -> str:
    """What a tool result is replaced by: its SIZE, never its content.

    Results are 53% of a transcript's characters and the part most likely to carry
    something private, so none of one is ever sent. The size is what the question
    needs: "the output itself is still needed" is a judgement about a call, and how
    much output there was is the only property of it that helps.
    """
    return f"ok, {max(0, chars)} chars (omitted)"


def read_rows(session_key: str) -> list[dict[str, Any]]:
    """The slot transcript's rows for *session_key*, newest last. Blocking IO.

    ``ConversationLog.read_messages`` rather than ``recent`` or the replay builder:
    those two project each row down to ``role`` and ``content``, and the tool input
    and result this point measures live in ``meta``. This is the same JSONL file the
    dashboard reads, so the state describes what this machine actually kept.

    Bounded to the newest :data:`MAX_ROWS`. Never raises: an unreadable transcript
    yields no rows, which is a compaction that goes unmeasured.
    """
    if not session_key:
        return []
    try:
        from kiro_crew.history import ConversationLog

        rows = ConversationLog().read_messages(session_key)
    except Exception:
        logger.debug("compaction.keep: transcript unreadable", exc_info=True)
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows[-MAX_ROWS:] if isinstance(row, dict)]


def replay_roles() -> frozenset[str]:
    """The roles today's recycle replay keeps, read from the replay's own definition.

    Imported here rather than restated, so ``chars_today`` cannot drift from what
    ``context.build_session_replay`` would actually re-inject. Function-local
    because ``context`` is the whole prompt builder and this module is reached from
    the compaction path; a failed import falls back to the three roles that
    definition has always held, which keeps the comparison available rather than
    losing the row.
    """
    try:
        from kiro_crew.context import RECALL_ROLES

        return frozenset(RECALL_ROLES)
    except Exception:
        logger.debug("compaction.keep: replay roles unreadable; using the shipped set")
        return frozenset({"user", "assistant", "inject"})


@dataclass(slots=True)
class CallFact:
    """One tool call as this point measures it -- real sizes, not sent sizes.

    ``input_chars`` and ``result_chars`` are the lengths the TRANSCRIPT holds, not
    the lengths of the clipped state: a live keep-list would re-inject the
    transcript's own bytes, so those are the numbers a chars comparison has to be
    made of. The clipping is a property of the question, not of the saving.
    """

    index: int
    tool: str
    raw_input: str
    input_chars: int
    result_chars: int
    pinned: bool

    @property
    def question_id(self) -> str:
        """This call's question id, which is also its key in the state."""
        return f"{QUESTION_PREFIX}{self.index}"


@dataclass(slots=True)
class Transcript:
    """Everything one compaction's scoring is computed from.

    ``chars_all`` is every character the compaction is about to discard -- the
    conversation text plus every tool input and result -- and it is the denominator
    the card's percentage is taken over. ``chars_today`` is the subset today's
    recycle replay keeps (:func:`replay_roles`), which is the arm Jev is being
    compared against.
    """

    messages: list[dict[str, str]]
    calls: list[CallFact]
    chars_all: int
    chars_today: int


def build_transcript(rows: Sequence[Mapping[str, Any]]) -> Transcript:
    """Read *rows* into the facts the state and the row are both built from.

    A pure walk, so it is testable without a session and cannot alter one. Pins the
    first row and the newest :data:`PINNED_TAIL_ROWS`, bounds the calls at
    :data:`MAX_CALLS`, and never raises: an unreadable row is skipped.
    """
    usable = [row for row in rows if isinstance(row, Mapping)]
    kept_roles = replay_roles()
    pinned_from = max(0, len(usable) - PINNED_TAIL_ROWS)
    messages: list[dict[str, str]] = []
    calls: list[CallFact] = []
    chars_all = 0
    chars_today = 0
    for position, row in enumerate(usable):
        try:
            role = str(row.get("role", "") or "")
            content = row.get("content")
            content = content if isinstance(content, str) else ""
            if role in kept_roles:
                chars_today += len(content)
            if role not in STATE_ROLES:
                continue
            pinned = position == 0 or position >= pinned_from
            if role != TOOL_ROLE:
                chars_all += len(content)
                if content:
                    messages.append({"role": role, "text": content})
                continue
            if len(calls) >= MAX_CALLS:
                continue
            meta = row.get("meta")
            meta = meta if isinstance(meta, Mapping) else {}
            raw_input = meta.get("input")
            raw_input = raw_input if isinstance(raw_input, str) else ""
            raw_output = meta.get("output")
            raw_output = raw_output if isinstance(raw_output, str) else ""
            chars_all += len(raw_input) + len(raw_output)
            calls.append(
                CallFact(
                    index=len(calls),
                    tool=content,
                    raw_input=raw_input,
                    input_chars=len(raw_input),
                    result_chars=len(raw_output),
                    pinned=pinned,
                )
            )
        except Exception:
            logger.debug("compaction.keep: skipping an unreadable transcript row", exc_info=True)
    return Transcript(messages=messages, calls=calls, chars_all=chars_all, chars_today=chars_today)


def build_state(transcript: Transcript, stage: Stage) -> dict[str, Any]:
    """The state one batch of questions is asked about, rendered at *stage*.

    Every tool input passes :func:`redacted` -- redaction first, then the stage's
    clip -- and every result is :func:`result_placeholder`, so no result byte is in
    here at any stage. A pinned call carries ``pinned: true`` so the oracle can see
    that it is context rather than something it is being asked about.
    """
    calls: list[dict[str, Any]] = []
    for call in transcript.calls:
        text = redacted(call.raw_input, stage.input_chars)
        if stage.one_line:
            text = " ".join(text.split())
        entry: dict[str, Any] = {
            "id": call.question_id,
            "tool": redacted(call.tool, MAX_TOOL_CHARS),
            "input": text,
            "result": result_placeholder(call.result_chars),
        }
        if call.pinned:
            entry["pinned"] = True
        calls.append(entry)
    messages = [
        {
            "role": row["role"],
            "text": head_tail(redacted(row["text"], None), stage.text_share),
        }
        for row in transcript.messages
    ]
    return {"messages": [row for row in messages if row["text"]], "calls": calls}


def estimated_tokens(state: Mapping[str, Any]) -> int:
    """Roughly how many tokens *state* is, rendered the way the wire renders it.

    ``json.dumps`` rather than ``len`` of the parts, for the reason
    ``gate._scan_text`` dumps: the fitting decision has to be made about the bytes
    that will actually be sent, keys and quoting included.
    """
    try:
        rendered = json.dumps(state, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(state)
    return len(rendered) // CHARS_PER_TOKEN


def fit_state(transcript: Transcript) -> tuple[dict[str, Any], Stage] | None:
    """The first stage whose state is under the ceiling, or ``None``.

    ``None`` is the ~8% of sessions the spike measured as never fitting. The caller
    writes :data:`ERROR_TOO_LARGE` for it and sends nothing: a state above the
    provider's ceiling is refused on arrival, and guessing at a sixth stage nobody
    measured would put an unmeasured shape on the wire.
    """
    for stage in STAGES:
        state = build_state(transcript, stage)
        if estimated_tokens(state) <= STATE_TOKEN_CEILING:
            return state, stage
    return None


def questions_for(calls: Sequence[CallFact]) -> list[Question]:
    """One ``Choice`` per NON-PINNED call, in transcript order.

    A pinned call gets none: it is kept whatever an answer would have said, so
    asking would spend a question on a decision already made.
    """
    return [
        Choice(call.question_id, PROMPT, options=list(OPTIONS)) for call in calls if not call.pinned
    ]


def batches(questions: Sequence[Question], size: int | None = None) -> list[list[Question]]:
    """*questions* split into requests of at most *size*, in order.

    Order is kept so a reader holding the request rows can line them up against the
    state's own call order; nothing depends on it, which is why the split is a plain
    slice rather than anything cleverer.

    *size* defaults to :data:`QUESTIONS_PER_REQUEST` resolved AT CALL TIME rather than
    as a default argument value: a default is evaluated once at import, so the bound
    would be frozen at whatever the constant held then -- invisible to a test that
    sets it, and to any future caller that lowers it.
    """
    step = max(1, QUESTIONS_PER_REQUEST if size is None else size)
    return [list(questions[i : i + step]) for i in range(0, len(questions), step)]


def read_decisions(answers: Any, asked: Sequence[Question]) -> dict[str, tuple[str, float]] | None:
    """``{question_id: (option, p)}`` for one batch, or ``None`` when unusable.

    The gate has already held each value against its own declared options, so this
    is the second reading rather than the only one. It is ALL-OR-NOTHING for the
    batch: a mapping missing one of the calls it was asked about cannot be folded
    into a fraction over all of them, and silently dropping the gap is what would
    make the card's number wrong rather than absent.
    """
    if not isinstance(answers, dict):
        return None
    out: dict[str, tuple[str, float]] = {}
    for question in asked:
        answer = answers.get(question.id)
        if not isinstance(answer, Answer):
            return None
        value = answer.value
        if not isinstance(value, str) or value not in OPTIONS:
            return None
        if isinstance(answer.p, bool) or not isinstance(answer.p, (int, float)):
            return None
        out[question.id] = (value, float(answer.p))
    return out


def tally(transcript: Transcript, decisions: Mapping[str, tuple[str, float]]) -> dict[str, Any]:
    """The counts and the two character arms, off one complete set of *decisions*.

    A pinned call counts as kept-both, because that is what pinning does; it is
    also counted on its own, so ``kept_both`` is never mistaken for "this many were
    judged worth keeping whole". ``chars_jev`` is built from the TRANSCRIPT's own
    lengths, not the clipped state's: what a live keep-list would re-inject is the
    transcript's bytes, so those are the bytes a comparison has to be made of.
    """
    pinned = kept_both = kept_call = dropped = 0
    chars_jev = transcript.chars_today
    for call in transcript.calls:
        if call.pinned:
            pinned += 1
            kept_both += 1
            chars_jev += call.input_chars + call.result_chars
            continue
        option, _p = decisions[call.question_id]
        if option == OPTION_BOTH:
            kept_both += 1
            chars_jev += call.input_chars + call.result_chars
        elif option == OPTION_CALL:
            kept_call += 1
            chars_jev += call.input_chars
        else:
            dropped += 1
    return {
        "total_calls": len(transcript.calls),
        "pinned_calls": pinned,
        "kept_both": kept_both,
        "kept_call": kept_call,
        "dropped": dropped,
        "chars_all": transcript.chars_all,
        "chars_today": transcript.chars_today,
        "chars_jev": chars_jev,
    }


def build_outcome(
    *,
    turn_id: str,
    counts: Mapping[str, Any],
    requests: int,
    fitting_stage: str,
) -> dict[str, Any]:
    """The fields the outcome row and the card line share.

    ``latency_ms`` is deliberately NOT here: it is a core row field
    (:func:`~kiro_crew.decisions.log.build_row`), so the row carries it at top level
    and an ``extra`` naming it would be dropped.

    ``point`` IS here, unlike ``skills.select``'s outcome, because this record is
    stamped on the same ``meta.decisions_strip`` field two other points use and the
    frontend dispatches on it: a record with no point reads as the oldest shape.
    """
    return {
        "turn_id": turn_id,
        "point": POINT,
        "requests": requests,
        "fitting_stage": fitting_stage,
        **dict(counts),
    }


# ── The hand-off to the compaction card ──
#
# The record has to reach the notice row ``dashboard/state.py`` appends when the
# compaction finishes. ``decisions.outcomes`` is the wrong vehicle: its one slot per
# session belongs to the next ASSISTANT reply, and claiming it here would take the
# strip a ``skills.select`` turn had already published and attach it to a compaction
# notice. So this module keeps its own store, on the same three bounds and for the
# same reasons -- one entry per session, a TTL, and a ceiling -- because the reader
# is allowed to never arrive: a compaction that finishes before the scoring does
# takes no record, which is the documented "no card line".

#: How long a published record stays claimable. Short, unlike ``outcomes``' ten
#: minutes: the consumer is the callback at the end of THIS compaction, seconds
#: away, and a record still sitting here a minute later belongs to a compaction
#: whose notice has already been drawn.
RECORD_TTL_SECONDS = 60.0

#: Most sessions holding an unclaimed record at once. A gateway compacts a handful
#: of sessions in a minute, so reaching this means records are being published and
#: never read, and the oldest are the least likely to ever be.
MAX_PENDING_RECORDS = 200

#: ``session_key -> (published_at_monotonic, record)``, oldest publish first.
_pending: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()

_pending_lock = threading.Lock()


def publish_record(session_key: str, record: dict[str, Any]) -> None:
    """Hand *record* to whatever draws this session's compaction notice. Never raises.

    A falsy key or a non-dict record is DROPPED rather than stored: the key is what
    the consumer matches on, so a record filed under ``""`` would be handed to the
    next keyless reader.
    """
    if not session_key or not isinstance(session_key, str) or not isinstance(record, dict):
        logger.debug("compaction.keep: record dropped (key=%r)", session_key)
        return
    now = time.monotonic()
    with _pending_lock:
        _drop_expired(now)
        _pending.pop(session_key, None)
        _pending[session_key] = (now, record)
        while len(_pending) > MAX_PENDING_RECORDS:
            _pending.popitem(last=False)


def take_record(session_key: str) -> dict[str, Any] | None:
    """Pop this session's pending record, or ``None`` when there is none.

    ``None`` is the overwhelmingly common answer -- the seam is off by default --
    so it is the cheap path: one lookup on a bounded mapping, no IO. The read is
    DESTRUCTIVE for the reason ``outcomes.consume`` is: the record describes ONE
    compaction, and leaving it in place would attach it to the next one.
    """
    if not session_key:
        return None
    now = time.monotonic()
    with _pending_lock:
        _drop_expired(now)
        entry = _pending.pop(session_key, None)
    if entry is None:
        return None
    published_at, record = entry
    return None if now - published_at > RECORD_TTL_SECONDS else record


def _drop_expired(now: float) -> None:
    """Evict entries older than the TTL. Caller holds :data:`_pending_lock`.

    Walks from the oldest and stops at the first live entry: the mapping is in
    publish order, so everything after it is younger.
    """
    for key in list(_pending.keys()):
        published_at, _record = _pending[key]
        if now - published_at <= RECORD_TTL_SECONDS:
            return
        del _pending[key]


def pending_count() -> int:
    """How many sessions hold an unclaimed record. For tests and diagnostics."""
    with _pending_lock:
        return len(_pending)


def reset_records() -> None:
    """Forget every pending record, so one test cannot inherit another's."""
    with _pending_lock:
        _pending.clear()


# ── The run ──


async def score_compaction(session_key: str) -> dict[str, Any] | None:
    """Score one compaction in the shadow. Returns the published record, or ``None``.

    The return value is for TESTS and for a caller that wants to log it. Nothing in
    the product branches on it: the compaction has already been decided by the time
    this is called, and this coroutine has no apply path. ``None`` covers the seam
    being off, an unsampled session, an unconsented scope, an empty or unreadable
    transcript, a state that never fits, a partial answer, a timeout and a refused
    row -- every one of which means "no card line".

    Never raises except :class:`asyncio.CancelledError`, which the gate propagates:
    cancellation is the gateway going away, not a measurement failure.
    """
    started = time.monotonic()
    turn_id = uuid.uuid4().hex[:16]
    try:
        # The keystone read and the transcript read are both filesystem IO, and both
        # are refusals in the overwhelming majority of installs, so the CHEAP one
        # goes first: an unconsented machine must not read a megabyte of transcript
        # to find out it is not sending anything.
        if not await asyncio.to_thread(core.is_enabled, POINT, session_key=session_key):
            return None
        return await asyncio.wait_for(
            _run(session_key, turn_id=turn_id, started=started), timeout=wait_budget()
        )
    except asyncio.TimeoutError:
        # The RUN stopped, which is not what the gate's own per-request rows say.
        # Recorded, because a compaction that goes unmeasured for want of time is
        # exactly the operator-actionable fact (``timeout_ms``) this point is here
        # to surface.
        logger.debug("compaction.keep: the run outlived its budget")
        await _record(
            session_key,
            latency_ms=_elapsed_ms(started),
            extra={"turn_id": turn_id, "point": POINT},
            error=ERROR_RUN_TIMEOUT,
        )
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("compaction.keep: leaving this compaction unmeasured", exc_info=True)
        return None


async def _run(session_key: str, *, turn_id: str, started: float) -> dict[str, Any] | None:
    """Read, fit, ask and record. Bounded by the caller's one wait."""
    rows = await asyncio.to_thread(read_rows, session_key)
    if not rows:
        return None
    transcript = await asyncio.to_thread(build_transcript, rows)
    if not transcript.calls:
        # Nothing to keep and nothing to drop: a conversation with no tool calls is
        # one this point has no observation about, and a row saying "0 of 0" would
        # be furniture in the log rather than a measurement.
        return None
    fitted = await asyncio.to_thread(fit_state, transcript)
    if fitted is None:
        await _record(
            session_key,
            latency_ms=_elapsed_ms(started),
            extra={
                "turn_id": turn_id,
                "point": POINT,
                "total_calls": len(transcript.calls),
                "chars_all": transcript.chars_all,
            },
            error=ERROR_TOO_LARGE,
        )
        return None
    state, stage = fitted
    asked = questions_for(transcript.calls)
    if not asked:
        # Every call is pinned, so there is nothing to ask and no answer to compare.
        return None
    groups = batches(asked)
    decisions, complete = await _ask_all(
        state, groups, session_key=session_key, turn_id=turn_id, stage=stage
    )
    if not decisions:
        # No batch answered. The gate has already written a row per refused request
        # carrying its own error category, so a second row here would say nothing
        # that is not already in the file.
        return None
    counts = (
        tally(transcript, decisions)
        if complete
        else {
            "total_calls": len(transcript.calls),
            "answered_calls": len(decisions),
            "chars_all": transcript.chars_all,
            "chars_today": transcript.chars_today,
        }
    )
    outcome = build_outcome(
        turn_id=turn_id,
        counts=counts,
        requests=len(groups),
        fitting_stage=stage.name,
    )
    row = await _record(
        session_key,
        latency_ms=_elapsed_ms(started),
        extra=outcome,
        error=None if complete else ERROR_PARTIAL,
    )
    if row is None or not complete:
        # A partial answer is recorded and NOT published: the line says "N of M
        # tool calls", and M includes calls nobody was asked about. A record whose
        # durable row was refused is not published either, for the reason every
        # point in this package states -- the thumbs on the line post that turn id,
        # and there would be no row for a verdict to be about.
        return None
    publish_record(session_key, row)
    return row


async def _ask_all(
    state: Mapping[str, Any],
    groups: Sequence[Sequence[Question]],
    *,
    session_key: str,
    turn_id: str,
    stage: Stage,
) -> tuple[dict[str, tuple[str, float]], bool]:
    """Every batch, at most :data:`MAX_CONCURRENT_REQUESTS` at a time.

    Returns ``(decisions, complete)``. ``complete`` is whether EVERY batch answered
    usably, which is what the caller needs to decide whether a fraction over all the
    calls is a true number.

    Each batch is an ordinary ``decide`` carrying the SAME state, so each one is
    scrubbed, bounded by ``timeout_secs`` and logged on its own -- the alternative,
    one giant request, exceeds the provider's ceiling and loses every answer to a
    single failure.
    """
    limit = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _one(index: int, batch: Sequence[Question]) -> dict[str, tuple[str, float]] | None:
        async with limit:
            extra: dict[str, Any] = {
                "turn_id": turn_id,
                "request": index,
                "requests": len(groups),
                "questions": len(batch),
                "fitting_stage": stage.name,
            }
            answers = await core.decide(
                POINT, dict(state), list(batch), session_key=session_key, extra=extra
            )
            return read_decisions(answers, batch)

    results = await asyncio.gather(
        *(_one(index, batch) for index, batch in enumerate(groups)),
        return_exceptions=True,
    )
    decisions: dict[str, tuple[str, float]] = {}
    complete = True
    for result in results:
        if isinstance(result, BaseException) or result is None:
            if isinstance(result, asyncio.CancelledError):
                raise result
            complete = False
            continue
        decisions.update(result)
    return decisions, complete


async def _record(
    session_key: str,
    *,
    latency_ms: int,
    extra: dict[str, Any],
    error: str | None,
) -> dict[str, Any] | None:
    """Write one row and return it, or ``None`` when it did not land. Never raises.

    OFF THE LOOP and bounded, the shape ``gate._write`` and
    ``tool_risk._record_outcome`` already use for the same write on the same loop:
    this coroutine runs on the gateway's event loop, so a synchronous append would
    put a lock wait and a filesystem write there.
    """
    try:
        row = _log.build_row(
            point=POINT,
            session_key=session_key,
            latency_ms=latency_ms,
            error=error,
            extra=extra,
        )
        written = await asyncio.wait_for(asyncio.to_thread(_log.append, row), LOG_BUDGET_SECS)
    except asyncio.TimeoutError:
        # The worker thread is NOT cancellable, so the append may still land --
        # acceptable for a write that cannot corrupt a line. What this arm refuses
        # is the RECORD, because the caller stopped waiting for its row.
        logger.debug("compaction.keep: the row outlived its write budget")
        return None
    except Exception:
        logger.debug("compaction.keep: could not record the row", exc_info=True)
        return None
    return row if written else None


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since *started* (a ``time.monotonic()`` reading)."""
    return int((time.monotonic() - started) * 1000)
