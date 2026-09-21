"""``task.split`` -- should this request be done inline, delegated, or split up?

Today the main agent alone decides how much of a request it keeps. It can answer
inline, hand the whole thing to one sub-agent (``spawn_run``), or cut it into
independent pieces and run several at once (``spawn_sub_agents``); the dashboard
draws a card per sub-agent either way (``chat_runner._native_subagent_sync``,
``handlers/messaging.api_spawn``). Nothing asks the question before the turn
starts, so a request that wanted three parallel workers is often answered by one
long serial reply, and a one-line question is sometimes delegated for no reason.

An ADVISORY, and only that
--------------------------
The answer becomes ONE system-side line prepended to the turn's context -- the
same pure-prepend channel a regenerate hint and hook output travel -- and the
agent still decides. Nothing here spawns anything, cancels anything, or changes a
tool's arguments: :func:`hint_line` returns text, and the caller's only use of it
is a string concatenation. That is the property that makes the point admissible
at all, and the reason the record calls its own arm a SUGGESTION and the agent's
arm the outcome.

Both arms, one line
-------------------
The oracle's arm is the answer. The BASELINE arm is what the agent actually did,
which is why it is recorded when the turn FINALIZES rather than when the question
is asked: it is counted from the turn's own spawn calls
(:func:`agent_choice_for`), so the row says whether the suggestion matched
behaviour without anybody reporting on themselves. That comparison is the whole
point of the row -- an advisory nobody can score is an advisory nobody can
withdraw.

Everything is a refusal back to today's turn
--------------------------------------------
:func:`suggest` returns ``None`` for: the seam off, the session unsampled, the
answer outside the three options, the transport failed, the budget expired. ``None``
means "prepend nothing", which is byte-identical to the turn this build runs today,
and :func:`record_outcome` is then never called, so a refusal leaves no row on the
reply either. A caller needs no try/except and no feature check of its own.

The record does not ride ``decisions.outcomes``
-----------------------------------------------
That registry holds ONE outcome per session for whatever finalizes the next
assistant message, and ``consume`` POPS. ``skills.select`` already claims it
during prompt assembly of the same turn, so publishing here would take that
turn's strip and replace it. So this point hands its row straight back to the
caller -- the shape ``tool.risk`` uses -- and the caller stamps it into the
``meta`` of the assistant row it is appending, under its own key
(``meta.decisions_split``).

Runs on the event loop
----------------------
Both halves are awaited by ``_run_chat`` itself, so there is no cross-thread
hand-off: ``gate.decide`` bounds the provider call by ``timeout_secs`` and this
module bounds the whole call by :func:`wait_budget`. The row's append is pushed to
a thread, because a synchronous write here would put a lock wait and a filesystem
write on the loop that serves every other session.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from typing import Any, Mapping, Sequence

from kiro_crew import decisions as core
from kiro_crew.decisions import log as _log
from kiro_crew.decisions.types import Answer, Choice, Question

logger = logging.getLogger(__name__)

POINT = "task.split"

#: The question's id, and therefore the key the answer arrives under.
QUESTION_ID = "shape"

#: The three shapes a turn can take, in increasing fan-out. The ORDER is part of
#: the contract: a reader comparing two rows needs to know which way "more
#: parallel" runs without consulting prose.
CHOICE_SINGLE = "single"
CHOICE_DELEGATE = "delegate"
CHOICE_SPLIT = "split"
CHOICES: tuple[str, ...] = (CHOICE_SINGLE, CHOICE_DELEGATE, CHOICE_SPLIT)

#: The rubric, sent as the question's prompt. One sentence per option, because the
#: options are the answer domain and a domain nobody defined is a domain every
#: provider reads differently. It names the EFFECT on the work rather than the
#: tools, so the answer does not turn on whether a provider has heard of
#: ``spawn_run``.
PROMPT = (
    "How should the assistant take on this request? "
    f"Answer {CHOICE_SINGLE} when one worker should do it from start to finish, "
    "because the steps depend on each other or the whole job is small. "
    f"Answer {CHOICE_DELEGATE} when it is one self-contained job worth handing to "
    "a single helper working on its own. "
    f"Answer {CHOICE_SPLIT} when it holds two or more parts that do not need each "
    "other's results, so several helpers could work at the same time."
)

#: Characters of the request sent with the question -- the SAME bound the other
#: points apply, because it is the same kind of text answering a question about
#: the same turn, and two different excerpt sizes would mean the consent text
#: describes one of them.
MAX_MESSAGE_CHARS = 2000

#: How many prior transcript rows the caller's history source may even look at.
#: The char budget is what bounds egress; this bounds the READ, so a long
#: conversation cannot turn one turn start into a whole-file scan. Held equal to
#: ``skills_select.MAX_HISTORY_MESSAGES`` by test.
MAX_HISTORY_MESSAGES = 20

#: Scheduling slack on top of the provider budget, and the floor and ceiling the
#: wait is clamped into. The ceiling is the real protection: this budget is spent
#: before the turn's first token, so a hand-edited ``timeout_ms`` must not hold a
#: reply there. Held equal to the values every other point in this package waits
#: by (``test_decisions_task_split.py``) so no point invents its own ceiling.
WAIT_MARGIN_SECS = 0.5
MIN_WAIT_SECS = 0.25
MAX_WAIT_SECS = 10.0

#: How long the outcome row's write may hold the caller, on top of the provider
#: budget. One ``O_APPEND`` of a few hundred bytes, so this exists only so a
#: stalled filesystem cannot make an observation cost the reply. Named and valued
#: as ``gate._LOG_BUDGET_SECS`` is, because it is the same write on the same loop.
LOG_BUDGET_SECS = 0.05

#: The MCP server a spawn call must be served by for the count to credit it, and
#: the two tools that count. Both halves come from the trusted ``_meta.kiro``
#: identity the ACP frame carries, never from a title: a shell command or a
#: third-party server exposing a same-named tool must not be able to move the
#: baseline arm, because that arm is what the suggestion is scored against.
CORE_MCP_SERVER = "kirocrew-core"
SPAWN_TOOLS = frozenset({"spawn_run", "spawn_sub_agents"})

#: Separator between a server-qualified prefix and the bare tool name. Transports
#: do not agree on one spelling -- kiro-cli reports ``<server>___<name>`` while the
#: canonical MCP form is ``mcp__<server>__<name>`` -- so the bare name is taken
#: after the LAST run of two or more underscores. Mirrors
#: ``session_directive._MCP_SEPARATOR_RE`` for the same reason it exists there: a
#: run of >= 2 is required, so ``do_spawn_run`` cannot smuggle a name in.
_MCP_SEPARATOR_RE = re.compile(r"_{2,}")


def questions() -> list[Question]:
    """The one question, with the three shapes as its whole domain.

    ONE ``Choice``, because the answer is consumed as one line of advice: a second
    question would be a second thing to reconcile inside a single hint.
    """
    return [Choice(QUESTION_ID, PROMPT, options=list(CHOICES))]


def is_spawn_call(mcp_server_name: str, tool_name: str) -> bool:
    """Whether a recorded tool CALL is one of the two spawn tools.

    Both arguments MUST come from the out-of-band ``_meta.kiro`` channel
    (``mcpServerName`` / ``toolName``), never the model-authored title. A shell
    tool has no MCP server name and a canonical tool name like ``execute_bash``, so
    it is not a spawn call; neither is a third-party server that merely exposes a
    tool named ``spawn_run``. Absent identity fails CLOSED, which here means "not
    counted": the count is the arm the suggestion is scored against, so a value the
    model can write must not be able to move it.
    """
    if mcp_server_name != CORE_MCP_SERVER:
        return False
    raw = tool_name or ""
    if raw in SPAWN_TOOLS:
        return True
    parts = _MCP_SEPARATOR_RE.split(raw)
    return len(parts) > 1 and parts[-1] in SPAWN_TOOLS


def _spawn_count(spawn_calls: object) -> int:
    """*spawn_calls* as a non-negative whole count, or 0. Never raises.

    A ``bool`` is NOT a count, for the reason ``gate._probability`` excludes one
    from being a probability: ``isinstance(True, int)`` holds, so an accidental
    flag would otherwise read as one spawn and report a delegation nobody made.
    """
    if isinstance(spawn_calls, bool) or not isinstance(spawn_calls, int):
        return 0
    return spawn_calls if spawn_calls > 0 else 0


def agent_choice_for(spawn_calls: int) -> str:
    """Which of :data:`CHOICES` the agent's own behaviour amounts to.

    Counted over spawn CALLS, which is what the turn's event stream reports: no
    call is :data:`CHOICE_SINGLE`, one is :data:`CHOICE_DELEGATE`, and two or more
    is :data:`CHOICE_SPLIT`. An unusable count reads as none rather than raising
    (:func:`_spawn_count`) -- this runs while a reply is being persisted.

    The simplification is deliberate and is the reason the record names the number
    beside the word: a single batch ``spawn_sub_agents`` call starting three
    helpers counts ONCE and therefore reads as ``delegate``. Reading the batch's
    own size would mean parsing tool arguments at the finalizer, which is a second
    egress-shaped path for a number that only has to be comparable with itself.
    """
    count = _spawn_count(spawn_calls)
    if count <= 0:
        return CHOICE_SINGLE
    return CHOICE_DELEGATE if count == 1 else CHOICE_SPLIT


def wait_budget() -> float:
    """How long one call may take, clamped into a sane window. Never raises."""
    try:
        budget = float(core.timeout_secs()) + WAIT_MARGIN_SECS
    except Exception:
        logger.debug("task.split: provider budget unreadable", exc_info=True)
        return MIN_WAIT_SECS
    if not math.isfinite(budget):
        return MIN_WAIT_SECS
    return min(max(budget, MIN_WAIT_SECS), MAX_WAIT_SECS)


def history_budget() -> int:
    """Characters of prior conversation this decision may carry, or 0.

    The consented ceiling (``gate.history_budget_chars``: the smaller of what
    ``config.json`` asks for and what the keystone recorded the owner reviewing).
    0 whenever either side is unreadable, which is the shipped default and sends
    the request alone.

    Filesystem IO (the keystone read), so a caller on the event loop hands it to a
    thread.
    """
    try:
        return max(0, int(core.history_budget_chars()))
    except Exception:
        logger.debug("task.split: history ceiling unreadable; sending no prior turns")
        return 0


def message_excerpt(text: str) -> str:
    """The part of the request that actually leaves the machine, after the cap.

    One function so the count on the row and the string in the request cannot
    disagree: :func:`build_state` sends this and :func:`message_chars` measures the
    same call.
    """
    return (text or "")[:MAX_MESSAGE_CHARS]


def message_chars(text: str) -> int:
    """Characters of *text* that were sent, which is the excerpt's own length."""
    return len(message_excerpt(text))


def build_state(
    text: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    *,
    history_budget_chars: int | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The state sent to the oracle: this request, and the prior turns.

    History is built by ``skills_select.build_history`` rather than by a second
    walk here: it is the same prior conversation spent out of the same consented
    ceiling, with the same two roles and the same newest-first order, and two
    nearly-identical walks would be two things to keep equal. ``history`` is
    OMITTED when there is none, so the request at the shipped default of 0 carries
    the request alone.

    *trace* receives ``history_chars`` and ``truncated`` from that builder, so the
    row can state the egress it actually paid for.
    """
    from kiro_crew.decisions.points.skills_select import build_history

    state: dict[str, Any] = {"message": message_excerpt(text)}
    rows = build_history(history, text, history_budget_chars=history_budget_chars, trace=trace)
    if rows:
        state["history"] = [dict(entry) for entry in rows]
    return state


def read_choice(answers: Any) -> str:
    """The shape the answer names, or ``""``. Identity is exact.

    The gate has already held the value against the declared options, so this is
    the second check rather than the only one -- and it is here because the caller
    renders it into prose the agent reads: a near-miss spelling would become a
    hint naming a shape nobody offered.
    """
    if not isinstance(answers, dict):
        return ""
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, Answer):
        return ""
    value = answer.value
    return value if isinstance(value, str) and value in CHOICES else ""


def probability_of(answers: Any) -> float | None:
    """The answer's probability, or ``None``. Only read after :func:`read_choice`."""
    if not isinstance(answers, dict):
        return None
    answer = answers.get(QUESTION_ID)
    return answer.p if isinstance(answer, Answer) else None


#: How the hint names each shape to the agent. One clause each, in the agent's own
#: vocabulary, so the line is actionable without the agent having to map a bare
#: word back onto a tool. English only and NOT localised: this text is prompt
#: input for a model, not interface copy for a person, and a translated hint would
#: make the advice a function of the reader's UI language.
_HINT_TEXT = {
    CHOICE_SINGLE: "handle this request yourself in this turn, without sub-agents",
    CHOICE_DELEGATE: "hand this request to a single sub-agent",
    CHOICE_SPLIT: "split this request into independent sub-tasks and run them in parallel",
}


def hint_line(decided: Mapping[str, Any]) -> str:
    """The one advisory line the caller prepends, or ``""``.

    Deliberately phrased as a SUGGESTION and marked as Jev's, because the agent
    still decides and a line that read as an instruction would make an advisory a
    silent gate. The probability is included for the same reason the strip prints
    it: an agent told how sure the suggestion is can discount a weak one, and a
    number nobody can see is a number nobody can weigh.

    ``""`` for an unreadable choice, so a caller that skipped :func:`read_choice`
    still prepends nothing rather than a half-sentence.
    """
    choice = str(decided.get("choice") or "")
    text = _HINT_TEXT.get(choice)
    if not text:
        return ""
    p = decided.get("p")
    if isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p):
        return f"Jev suggests: {text} ({float(p):.2f}). You decide."
    return f"Jev suggests: {text}. You decide."


async def suggest(
    text: str,
    *,
    session_key: str | None = None,
    history: Sequence[Mapping[str, Any]] | None = None,
    config: Any | None = None,
) -> dict[str, Any] | None:
    """Jev's suggested shape for this request, or ``None`` to prepend nothing.

    Returns ``{turn_id, choice, p, latency_ms}``. ``choice`` is one of
    :data:`CHOICES`; no answer at all is ``None`` rather than a choice of
    :data:`CHOICE_SINGLE`, so a caller can tell "Jev said do it inline" from
    "nothing was asked or nothing came back" -- the first is advice with a receipt,
    the second is the turn this build runs today.

    *history* is prior transcript rows (``{role, content}``, newest LAST), already
    bounded by the caller's read. Passed as a value rather than a callable because
    this runs on the loop and the caller's read is its own to schedule; at the
    shipped ceiling of 0 nothing from it is sent.

    Never raises except :class:`asyncio.CancelledError`, which ``decide``
    propagates: cancellation is the turn going away, not a decision failure.
    """
    turn_id = uuid.uuid4().hex[:16]
    # Bound before the try so the latency is always measurable, including when the
    # ceiling read and the history build were what took the time.
    started = time.monotonic()
    try:
        # The ceiling FIRST, off the loop: it reads the keystone as well as the
        # config, and at the shipped default of 0 there is nothing for the history
        # build to contribute.
        budget = await asyncio.to_thread(history_budget)
        trace: dict[str, Any] = {}
        state = build_state(
            text,
            history if budget > 0 else (),
            history_budget_chars=budget,
            trace=trace,
        )
        extra: dict[str, Any] = {
            "turn_id": turn_id,
            "message_chars": message_chars(text),
            "history_chars": trace.get("history_chars", 0),
        }
        answers = await asyncio.wait_for(
            core.decide(
                POINT, state, questions(), session_key=session_key, config=config, extra=extra
            ),
            timeout=wait_budget(),
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        # The gate writes its own row for a provider timeout; this arm is the
        # caller-side budget expiring around it, which means the gate is still
        # inside its own and will record whatever it finds.
        logger.debug("task.split: the call outlived the caller's budget")
        return None
    except Exception:
        # This sits before the turn's first token, so the seam may cost an
        # observation and must never cost the reply.
        logger.debug("task.split: prepending no hint", exc_info=True)
        return None
    choice = read_choice(answers)
    if not choice:
        return None
    return {
        "turn_id": turn_id,
        "choice": choice,
        "p": probability_of(answers),
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


def build_outcome(decided: Mapping[str, Any], spawn_calls: int) -> dict[str, Any]:
    """The fields the outcome row and the strip line share.

    ``agent_choice`` is derived from the COUNT rather than reported, and ``agree``
    is derived from the two words rather than asserted, so no field here can
    disagree with the one beside it. ``spawn_calls`` is kept as well as the word it
    produced, because the word is a bucket and a reader folding day-files needs the
    number the bucket came from.

    ``latency_ms`` is deliberately NOT here: it is a core row field
    (:func:`~kiro_crew.decisions.log.build_row`), so the row carries it at top
    level and an ``extra`` naming it would be dropped.
    """
    calls = _spawn_count(spawn_calls)
    jev_choice = str(decided.get("choice") or "")
    agent_choice = agent_choice_for(calls)
    return {
        "turn_id": decided.get("turn_id"),
        "jev_choice": jev_choice,
        "agent_choice": agent_choice,
        "spawn_calls": calls,
        "agree": jev_choice == agent_choice,
        "p": decided.get("p"),
    }


async def record_outcome(
    *,
    session_key: str | None,
    decided: Mapping[str, Any],
    spawn_calls: int,
) -> dict[str, Any] | None:
    """Write one outcome row for the finished turn and RETURN it. Never raises.

    Called when the turn FINALIZES, which is the earliest moment the baseline arm
    exists: the agent's own shape is its spawn calls, and those are only all in
    once the turn stops making them.

    ``None`` means no row was written -- the build raised, ``append`` refused it
    (the day-file ceiling, a sealed directory), or the write outlived
    :data:`LOG_BUDGET_SECS`. The caller stamps nothing in that case, the rule the
    other points already follow: a line whose durable row was refused names a
    ``turn_id`` no verdict could be filed against, and the thumbs on it POST that
    id.

    Returns the ROW as it was written, so what the transcript shows and what the
    log holds cannot drift into two descriptions of one turn.

    OFF THE LOOP: this is awaited from ``_run_chat``, so a synchronous append would
    put a lock wait and a filesystem write on the gateway's event loop.
    """
    try:
        row = _log.build_row(
            point=POINT,
            session_key=session_key,
            latency_ms=int(decided.get("latency_ms") or 0),
            extra=build_outcome(decided, spawn_calls),
        )
        written = await asyncio.wait_for(asyncio.to_thread(_log.append, row), LOG_BUDGET_SECS)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        # The worker thread is NOT cancellable, so the append may still land
        # afterwards -- acceptable for a write that cannot corrupt a line. What
        # this arm refuses is the LINE, because the caller stopped waiting for the
        # row a verdict would be filed against.
        logger.debug("task.split: outcome row outlived its write budget; stamping nothing")
        return None
    except Exception:
        logger.debug("task.split: could not record the outcome row", exc_info=True)
        return None
    if not written:
        logger.debug("task.split: outcome row was not written; stamping nothing")
        return None
    return row
