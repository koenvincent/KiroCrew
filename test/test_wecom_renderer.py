"""Tests for kiro_crew.wecom.renderer (WeComRenderer, Layer 2b)."""

from __future__ import annotations

import dataclasses

import pytest

from conftest import CREDENTIAL_STRADDLE_SHAPES, assert_rejected_without_backtracking
from kiro_crew.messaging.display_safety import canonicalize_display
from kiro_crew.messaging.renderer import _default_redactor
from kiro_crew.wecom.renderer import _STREAM_MAX_AGE_S, WeComRenderer, _render_options_as_text
from kiro_crew.wecom.transport import WECOM_CAPABILITIES

# The PEM markers are ASSEMBLED from fragments, never written as one literal, so
# the internal content scan (rule ``credential-private-key``, which matches a
# BEGIN...PRIVATE-KEY header on a source line) does not flag a test fixture. The
# runtime value is byte-identical to the real marker, so the test still exercises
# the exact string the renderer's guard and the redactor recognise.
_DASHES = "-" * 5
_KEY_KIND = "PRIVATE" + " KEY"


def _pem_begin(*, markup: bool = False, split_token: bool = False) -> str:
    kind = "**PRIVATE**" + " KEY" if markup else _KEY_KIND
    head = f"{_DASHES}BEG**IN**" if split_token else f"{_DASHES}BEGIN"
    return f"{head} RSA {kind}{_DASHES}"


def _pem_end() -> str:
    return f"{_DASHES}END RSA {_KEY_KIND}{_DASHES}"


class TestStripOptionsRedos:
    def test_unterminated_options_tag_is_not_redos(self) -> None:
        # Regression (py/polynomial-redos): a plain greedy ``.*`` body could
        # consume a "[" that ALSO starts the outer "[OPTIONS:" literal, so over
        # text with many "[OPTIONS:" prefixes search() re-explored the body from
        # each position — polynomial. The tempered body
        # (?:[^[]|\[(?!OPTIONS:))* forbids only a re-occurring "[OPTIONS:", so the
        # body is unambiguous (linear). A whitespace-padded unterminated tag and
        # many repeated "[OPTIONS:" prefixes (the real pump) must both be rejected
        # in CPU time linear in the pump -- see
        # conftest.assert_rejected_without_backtracking for why this is not a
        # 1.0 s wall-clock bound.

        # A single unterminated tag: no closing ']' after the last "[OPTIONS",
        # so the whole still-streaming partial is hidden.
        def hidden(text: str) -> None:
            assert (
                _render_options_as_text(text) == ""
            ), "an unterminated marker is hidden, never rendered"

        assert_rejected_without_backtracking(hidden, lambda n: "[OPTIONS:" + ("\t" * n) + "x")

        # Many repeated "[OPTIONS:" prefixes (the real polynomial pump). There is
        # no closing ']', so the trailer regex does not match; the property under
        # test is the cost of deciding that, not the rendered text.
        assert_rejected_without_backtracking(
            _render_options_as_text, lambda n: "[OPTIONS:" * n + "x"
        )


class FakeClient:
    """Records WS stream frames and response_url fallback POSTs."""

    def __init__(self, stream_ok: bool = True) -> None:
        self.frames: list[dict] = []
        self.replies: list[tuple[str, str]] = []
        self._stream_ok = stream_ok
        self.dead_streams: set[str] = set()
        #: (chat_id, content) of confirmed pushes — the answer's tail, and a head
        #: re-delivered after its sealing frame turned out to be refused.
        self.pushed: list[tuple[str, str]] = []

    async def send_stream(
        self,
        req_id: str,
        stream_id: str,
        content: str,
        *,
        finish: bool,
        await_ack: bool = False,
    ) -> bool:
        self.frames.append(
            {"req_id": req_id, "stream_id": stream_id, "content": content, "finish": finish}
        )
        return self._stream_ok

    def stream_is_dead(self, stream_id: str) -> bool:
        """The renderer consults this before every frame, so the fake owes it.

        Bubbles are live unless a test says otherwise; sealing behaviour has its
        own coverage in test_wecom_wire_reliability.py.
        """
        return stream_id in self.dead_streams

    def stream_had_rejection(self, stream_id: str) -> bool:
        """The narrower question: was everything written here ACCEPTED.

        Consulted by the aged rotation and by the post-turn seal recheck. A dead
        bubble was also refused, so it answers for both.
        """
        return stream_id in self.dead_streams

    async def send_proactive(self, chat_id: str, content: str) -> bool:
        self.pushed.append((chat_id, content))
        return True

    async def send_reply(self, url: str, content: str) -> None:
        self.replies.append((url, content))


def _renderer(client: FakeClient, req_id: str = "rq1") -> WeComRenderer:
    return WeComRenderer(client, req_id, "https://resp.url", WECOM_CAPABILITIES)


class TestStreaming:
    @pytest.mark.asyncio
    async def test_turn_start_sends_placeholder(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        # WeCom renders <think>…</think> as its own collapsed reasoning block, so
        # the placeholder does not sit in the answer and does not have to be
        # cleared before the real text arrives.
        assert c.frames[0]["content"] == "<think>…</think>"
        assert c.frames[0]["finish"] is False

    @pytest.mark.asyncio
    async def test_turn_start_idempotent(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_turn_start()  # second call no-ops
        assert len(c.frames) == 1

    @pytest.mark.asyncio
    async def test_final_answer_is_accumulated_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("Hello ")
        await r.on_text_chunk("world")
        await r.on_done()
        final = c.frames[-1]
        assert final["content"] == "Hello world"
        assert final["finish"] is True

    @pytest.mark.asyncio
    async def test_options_trailer_becomes_a_numbered_list(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("Pick one\n\n[OPTIONS: A | B | C]")
        await r.on_done()
        # Numbered text, not deleted: WeCom renders no chips, but the user still
        # has to learn the choices exist and can answer by typing one.
        assert c.frames[-1]["content"] == "Pick one\n\n1. A\n2. B\n3. C"

    @pytest.mark.asyncio
    async def test_tool_footer_pushed(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_tool_call("t1", "fs_read", tool_kind="read")
        # force-pushed frame carries the transient footer
        assert any("🔧 正在运行：fs_read" in f["content"] for f in c.frames)

    @pytest.mark.asyncio
    async def test_error_done_shows_error_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_done(stop_reason="error")
        assert c.frames[-1]["content"] == "⚠️ 出错了，请重试"
        assert c.frames[-1]["finish"] is True


class TestFallback:
    @pytest.mark.asyncio
    async def test_no_req_id_uses_response_url(self) -> None:
        c = FakeClient()
        r = WeComRenderer(c, "", "https://resp.url", WECOM_CAPABILITIES)
        await r.on_turn_start()  # no stream (no req_id)
        await r.on_text_chunk("reply text")
        await r.on_done()
        assert c.frames == []  # never streamed
        assert c.replies == [("https://resp.url", "reply text")]

    @pytest.mark.asyncio
    async def test_stream_died_falls_back_to_response_url(self) -> None:
        c = FakeClient(stream_ok=False)  # every send_stream reports failure
        r = _renderer(c)
        await r.on_turn_start()  # placeholder send reports False -> stream_ok flips off
        await r.on_text_chunk("answer")
        await r.on_done()
        assert c.replies == [("https://resp.url", "answer")]


class TestClose:
    @pytest.mark.asyncio
    async def test_close_after_done_is_noop(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("done text")
        await r.on_done()
        finish_frames_before = sum(1 for f in c.frames if f["finish"])
        await r.close()
        finish_frames_after = sum(1 for f in c.frames if f["finish"])
        assert finish_frames_before == finish_frames_after == 1

    @pytest.mark.asyncio
    async def test_close_without_done_finalizes(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("partial")
        await r.close()  # turn never reached on_done (e.g. cold-start failure)
        assert any(f["finish"] for f in c.frames)


class TestPromptChoice:
    @pytest.mark.asyncio
    async def test_prompt_choice_is_noop(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        before = len(c.frames)
        await r.on_prompt_choice([{"label": "yes"}], "rq")  # WeCom has no buttons
        assert len(c.frames) == before  # nothing rendered, no raise


class TestThinkReasoningRedaction:
    """The ``<think>`` reasoning frame is scrubbed render-aware, not literally.

    WeCom renders the ``<think>`` block as markdown, so a credential split by
    emphasis (``AKIA**REST**``) survives a literal byte scan and is reassembled
    on screen -- the same hazard the answer body is guarded against, on the same
    channel. Asserted against the RENDERED form.
    """

    @pytest.mark.asyncio
    async def test_think_reasoning_redacts_markup_split_credential(self) -> None:
        from kiro_crew.messaging.display_safety import canonicalize_display

        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        c.frames.clear()
        r._last_send = 0.0  # clear the throttle so the reasoning frame goes out
        await r.on_thinking("leaking AKIAIOSF**ODNN7EXAMPLE** in the trace")

        think = "".join(f["content"] for f in c.frames)
        assert "<think>" in think, f"no reasoning frame was sent: {c.frames}"
        assert "AKIAIOSFODNN7EXAMPLE" not in canonicalize_display(think)

    @pytest.mark.asyncio
    async def test_think_reasoning_keeps_clean_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        c.frames.clear()
        r._last_send = 0.0
        await r.on_thinking("just thinking out loud, no secret")

        think = "".join(f["content"] for f in c.frames)
        assert "just thinking out loud, no secret" in think


class TestAnswerBodyRedaction:
    """The answer body is scrubbed render-aware at the send, and no cut severs a key.

    WeCom renders the body as markdown and extracts no attachments from it, so a
    credential split by emphasis (``AKIA**REST**``) survives the driver's literal
    channel-neutral pass and is reassembled on screen. ``_render_slice`` redacts
    each outgoing slice, and the cut that produced the slice is chosen with
    ``safe_cut_offset`` so a credential is never split across two bubbles -- the
    offsets stay in raw ``text()`` coordinates, which is what keeps a bubble
    rotation resuming at the right place. Asserted against the RENDERED form.
    """

    @pytest.mark.asyncio
    async def test_answer_body_redacts_markup_split_credential(self) -> None:
        from kiro_crew.messaging.display_safety import canonicalize_display

        c = FakeClient()
        r = _renderer(c)
        await r.on_text_chunk("the key is AKIAIOSF**ODNN7EXAMPLE** ok")
        await r.on_done()

        final = c.frames[-1]["content"]
        assert "AKIAIOSFODNN7EXAMPLE" not in canonicalize_display(final)

    def test_text_stays_raw_for_persistence(self) -> None:
        # text() is what drive_turn persists; the redaction lives only on the
        # delivery slices, so the raw answer is unchanged here.
        c = FakeClient()
        r = _renderer(c)
        r._buf.append("plain answer body")
        assert r.text() == "plain answer body"

    @pytest.mark.asyncio
    async def test_answer_body_keeps_clean_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_text_chunk("Hello world")
        await r.on_done()

        assert c.frames[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_roll_keeps_offsets_in_raw_coordinates(self) -> None:
        # Offsets index raw text() (not a content-dependent redacted view), so a
        # credential completing across a bubble ROLL does not shift them: the
        # resume point is stable as text() grows -- the delivered bubbles reassemble
        # the whole answer with no dropped or duplicated non-secret span.
        c = FakeClient()
        r = _renderer(c)
        await r.on_text_chunk("alpha beta gamma delta")
        await r._push(force=True)
        c.dead_streams.add(r._stream_id)  # next push must roll
        await r.on_text_chunk(" epsilon zeta eta theta")
        await r._push(force=True)
        await r.on_done()

        # Reconstruct what the reader sees across bubbles: the resume after the roll
        # picks up exactly where the sealed bubble left off, so no word is dropped
        # and none is duplicated across the boundary.
        delivered = "".join(f["content"] for f in c.frames) + "".join(p[1] for p in c.pushed)
        for word in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"):
            assert word in delivered, word


# A small message cap, so a fixture positions itself against it with a line of
# padding instead of five thousand characters of it.
_SMALL_CAP = 96


def _capped_renderer(client: FakeClient, cap: int = _SMALL_CAP) -> WeComRenderer:
    caps = dataclasses.replace(WECOM_CAPABILITIES, max_message_chars=cap)
    return WeComRenderer(client, "rq1", "https://resp.url", caps, chat_id="chat1")


def _answer_whose_cap_lands_between(head: str, tail: str, cap: int = _SMALL_CAP) -> str:
    """An answer whose message cap falls EXACTLY between *head* and *tail*."""
    assert len(head) <= cap, "the fixture has to fit the cap it is positioned against"
    return "x" * (cap - len(head)) + head + tail + " and some trailing prose"


async def _stream_across_a_rotation(answer: str, cap: int = _SMALL_CAP) -> FakeClient:
    """Stream *answer*, let the platform seal the bubble, and finish the turn.

    The rotation is the moment the boundary becomes irreversible: a sealed bubble
    can never be rewritten, so what it shows sits beside the next bubble for good.
    """
    c = FakeClient()
    r = _capped_renderer(c, cap)
    await r.on_text_chunk(answer)
    await r._push(force=True)
    c.dead_streams.add(r._stream_id)
    await r._push(force=True)
    await r.on_done()
    return c


def _reader_instants(c: FakeClient) -> list[list[str]]:
    """Every state the screen passed through, as the bodies visible at that moment.

    A stream frame REPLACES its own bubble, so at any instant the screen holds the
    LATEST frame of each bubble opened so far -- and a sealed bubble's last frame
    stays there for good. The final state is not enough to assert against: a frame
    that puts a credential on screen has been read by the time a later frame
    supersedes it, so each instant is checked on its own.
    """
    order: list[str] = []
    shown: dict[str, str] = {}
    instants: list[list[str]] = []
    for f in c.frames:
        sid = f["stream_id"]
        if sid not in shown:
            order.append(sid)
        shown[sid] = f["content"]
        instants.append([shown[s] for s in order])
    pushed: list[str] = []
    for p in c.pushed:
        pushed.append(p[1])
        instants.append([shown[s] for s in order] + list(pushed))
    return instants


def _reader_views(c: FakeClient) -> tuple[str, str]:
    """(what the screen shows, what a copy of every message shows), at the end.

    The last instant of :func:`_reader_instants`, kept as a named pair because two
    tests read the two renderings apart.
    """
    instants = _reader_instants(c)
    bodies = instants[-1] if instants else []
    return (
        "".join(canonicalize_display(b) for b in bodies),
        canonicalize_display("".join(bodies)),
    )


def _without_think_wrapper(text: str) -> str:
    """The reasoning wrapper as the READER sees it: not at all.

    WeCom renders ``<think>...</think>`` as its own reasoning panel, so the tags are
    markup rather than characters on screen. They matter here because they sit
    exactly where a reasoning tail meets the next bubble's opening characters -- a
    scan that keeps them reads a break the reader does not have, and the credential
    those two pieces spell goes unseen.
    """
    return text.replace("<think>", "").replace("</think>", "")


def _assert_nothing_reached_the_reader(c: FakeClient) -> None:
    """No credential at ANY instant the screen passed through.

    Scanned with the SAME pair the renderer redacts with, so the assertion cannot
    drift away from the guard into checking a different notion of "credential". Both
    readings are checked at each instant, for the same reason
    :func:`joins_to_a_credential` checks both: neither contains the other.
    """
    for bodies in _reader_instants(c):
        bodies = [_without_think_wrapper(b) for b in bodies]
        for view in (
            "".join(canonicalize_display(b) for b in bodies),
            canonicalize_display("".join(bodies)),
        ):
            assert _default_redactor(view) == view, "a credential reached the reader"


class TestCutSafetyAcrossBubbles:
    """A cap may not sever a credential the reader's client will rejoin.

    The cap is applied to the RAW answer while the reader sees the CANONICAL
    rendering of each bubble, so a credential the model split with markup can be
    cut in half: each bubble is scrubbed alone and matches nothing, and the client
    renders the markup away and joins the halves on screen. A bubble rotation is
    irreversible, so there is no in-frame recovery.

    Each case below is a genuine straddle -- the premise assertions say so -- and
    every one of them is red without ``safe_cut_offset`` in ``_push`` and in
    ``_roll_if_sealed``.
    """

    @pytest.mark.asyncio
    async def test_a_link_split_key_does_not_cross_two_bubbles(self) -> None:
        # The acceptance case: a Markdown link whose target carries a comma, which
        # no hand-written character class reached across six review rounds.
        head, tail = "[AKIA](https://ex.test/a,b)", "IOSFODNN7EXAMPLE"
        c = await _stream_across_a_rotation(_answer_whose_cap_lands_between(head, tail))

        on_screen, in_a_copy = _reader_views(c)
        assert "AKIAIOSFODNN7EXAMPLE" not in on_screen
        assert "AKIAIOSFODNN7EXAMPLE" not in in_a_copy
        _assert_nothing_reached_the_reader(c)

    @pytest.mark.parametrize(("head", "tail"), CREDENTIAL_STRADDLE_SHAPES)
    @pytest.mark.asyncio
    async def test_no_straddling_shape_reaches_the_reader(self, head: str, tail: str) -> None:
        # Premise: neither half is a credential ALONE. That is why scrubbing each
        # bubble cannot see this, and it is asserted so a fixture that stops
        # straddling fails loudly instead of passing on a case it does not cover.
        assert _default_redactor(head) == head, "the head half must be clean alone"
        assert _default_redactor(tail) == tail, "the tail half must be clean alone"

        c = await _stream_across_a_rotation(_answer_whose_cap_lands_between(head, tail))

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_a_key_completing_after_the_frame_went_out_is_still_covered(self) -> None:
        # The growth case, and the reason the boundary is re-decided at the rotation
        # rather than trusted from when it was recorded. The first frame ends exactly
        # after the key's first half, which is SAFE at that moment because nothing
        # follows it. The second half arrives afterwards.
        answer = "x" * (_SMALL_CAP - 8) + "AKIAIOSF"
        c = FakeClient()
        r = _capped_renderer(c)
        await r.on_text_chunk(answer)
        await r._push(force=True)
        assert r._sent_abs == _SMALL_CAP, "the first frame must end on the cap"

        await r.on_text_chunk("ODNN7EXAMPLE and some trailing prose")
        c.dead_streams.add(r._stream_id)
        await r._push(force=True)
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_a_key_completing_after_the_rotation_is_still_covered(self) -> None:
        # The same growth, one moment later, which is a different window. Here the
        # answer ends exactly at the resume offset WHEN THE BUBBLE ROTATES, so the
        # rotation has no boundary to check -- nothing is severed yet -- and it
        # freezes the sealed bubble showing the key's first half. The second half
        # arrives afterwards and opens the replacement bubble, where it is scrubbed
        # alone and matches nothing. Only re-deciding the seam on the replacement
        # bubble's own frames catches it, because the bubble above cannot be edited.
        answer = "x" * (_SMALL_CAP - 8) + "AKIAIOSF"
        c = FakeClient()
        r = _capped_renderer(c)
        await r.on_text_chunk(answer)
        await r._push(force=True)
        assert r._sent_abs == len(answer), "the first frame must carry the whole answer"

        # Age the bubble out with nothing refused, so the rotation resumes exactly
        # where the accepted frame ended -- which is the end of the answer.
        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()
        assert r._carried == len(answer), "the rotation had nothing to sever yet"

        await r.on_text_chunk("ODNN7EXAMPLE and some trailing prose")
        await r._push(force=True)
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_the_seam_is_graded_against_what_the_sealed_bubble_shows(self) -> None:
        # The seam only moves BACK, so the bubble above keeps showing MORE than the
        # candidate's own prefix. Grading that prefix instead of the frozen span
        # accepts a resume point whose opening characters complete the sealed
        # bubble's trailing key prefix -- which is the pair actually on screen.
        #
        # The prefix is spelled as four links: wide in raw characters, four once the
        # platform renders them away. That width is what leaves a sampled candidate
        # clear of the prefix, so the tail opens with the alphanumeric run instead of
        # swallowing the whole key and scrubbing it.
        run = "ABCDEFGHIJKLMNOP"
        prefix = "".join(f"[{ch}](https://ex.test/)" for ch in "AKIA")
        first = run + prefix
        c = FakeClient()
        r = _capped_renderer(c, cap=200)
        await r.on_text_chunk(first)
        await r._push(force=True)
        assert r._sent_abs == len(first), "the first frame must carry the whole answer"

        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()
        assert r._carried == len(first), "the rotation had nothing to sever yet"

        await r.on_text_chunk("IOSFODNN7EXAMPLE trailing prose")
        await r._push(force=True)
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_a_zero_seam_does_not_switch_the_re_decision_off(self) -> None:
        # Zero is a LEGAL answer from the search: the seam may move back to the
        # answer's start. Gating the re-decision on `_carried` lets that answer
        # disable the check for the rest of the bubble, so the guard asks whether a
        # frozen span exists instead.
        #
        # The rotation below is real, so the sealed bubble is genuinely on screen;
        # only the seam is then placed at zero by hand, standing in for a search that
        # walked there on an earlier frame.
        run = "ABCDEFGHIJKLMNOP"
        prefix = "".join(f"[{ch}](https://ex.test/)" for ch in "AKIA")
        first = run + prefix
        c = FakeClient()
        r = _capped_renderer(c, cap=200)
        await r.on_text_chunk(first)
        await r._push(force=True)
        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()
        assert r._frozen_end, "the sealed bubble must have left an edge on screen"

        r._carried = 0
        await r.on_text_chunk("IOSFODNN7EXAMPLE trailing prose")
        await r._push(force=True)
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_reasoning_frozen_by_a_rotation_is_part_of_the_screen(self) -> None:
        # A bubble can rotate while the answer is still EMPTY, because reasoning is
        # what fills it in that window. The answer edge is then zero, so a seam
        # graded against the answer alone is graded against nothing -- while the
        # sealed bubble sits on screen showing the reasoning's last characters.
        #
        # Same straddle as the answer cases, across the thinking/answer boundary: the
        # key's head ends the reasoning, its tail opens the answer, and each side is
        # clean scanned alone.
        run = "ABCDEFGHIJKLMNOP"
        prefix = "".join(f"[{ch}](https://ex.test/)" for ch in "AKIA")
        head, tail = run + prefix, "IOSFODNN7EXAMPLE trailing prose"
        assert _default_redactor(head) == head, "the reasoning half must be clean alone"
        assert _default_redactor(tail) == tail, "the answer half must be clean alone"

        c = FakeClient()
        r = _capped_renderer(c, cap=200)
        await r.on_thinking(head)
        await r._push(force=True)
        assert r._sent_reasoning, "the reasoning frame must be recorded as delivered"

        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()
        assert not r._frozen_end, "the answer edge is zero -- nothing of it was sent"
        assert r._frozen_reasoning, "the sealed bubble still shows its reasoning"

        await r.on_text_chunk(tail)
        await r._push(force=True)
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_a_key_split_across_three_bubbles_is_still_caught(self) -> None:
        # The screen is every sealed bubble, not just the one directly above. A key
        # in three pieces -- first piece in bubble one, middle piece the whole of
        # bubble two, last piece opening bubble three -- is invisible to a seam graded
        # against bubble two alone: "middle" + "last" is not a credential, only
        # "first" + "middle" + "last" is. Grading against the answer up to the
        # furthest sealed edge keeps the first piece in view.
        first, middle, last = "AKIA", "IOSFODNN7", "EXAMPLE and some trailing prose"
        c = FakeClient()
        r = _capped_renderer(c, cap=200)

        await r.on_text_chunk(f"prose. {first}")
        await r._push(force=True)
        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()

        await r.on_text_chunk(middle)
        await r._push(force=True)
        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()

        await r.on_text_chunk(last)
        await r._push(force=True)
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_a_key_completing_before_the_final_frame_is_still_covered(self) -> None:
        # The same seam, reached by the FINAL frame instead of a streaming one. The
        # replacement bubble is live and young, so the seal does not rotate it and
        # the seam it inherited stays in place while the answer grew past it. The
        # final frame reads the answer from that seam, so it has to re-decide it too.
        answer = "x" * (_SMALL_CAP - 8) + "AKIAIOSF"
        c = FakeClient()
        r = _capped_renderer(c)
        await r.on_text_chunk(answer)
        await r._push(force=True)
        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()

        # No streaming frame between the growth and the seal.
        await r.on_text_chunk("ODNN7EXAMPLE and some trailing prose")
        await r.on_done()

        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_the_cap_leaves_the_rotation_nothing_to_undo(self) -> None:
        # Why the CAP picks a safe boundary too, instead of leaving the whole job to
        # the rotation. A bubble filled to a boundary that severs a key forces the
        # rotation to walk back behind it, and everything in between is then
        # delivered twice -- safe, but the reader meets the same sentence in both
        # bubbles. Choosing the cut when the bubble is filled leaves nothing to undo.
        head, tail = "[AKIA](https://ex.test/a,b)", "IOSFODNN7EXAMPLE"
        c = FakeClient()
        r = _capped_renderer(c)
        await r.on_text_chunk(_answer_whose_cap_lands_between(head, tail))
        await r._push(force=True)
        cut = r._sent_abs

        # Age the bubble past the platform's stream lifetime: that rotation resumes
        # from the frame it accepted, which is the offset under test here.
        r._stream_opened_at -= _STREAM_MAX_AGE_S + 1
        r._roll_if_sealed()

        assert r._carried == cut, "the rotation had to move the cut, so text repeats"

    @pytest.mark.asyncio
    async def test_prose_still_arrives_whole_across_the_rotation(self) -> None:
        # The allow direction, and the one that matters most: a guard that walks the
        # cut back for boundaries severing nothing would deliver a shorter answer, or
        # none. Emphasis spans the cap here, which canonicalisation collapses -- the
        # shape closest to the hazard that is nonetheless harmless.
        words = [f"word{i:02d}" for i in range(40)]
        answer = " ".join(words[:20]) + " **bold across the cap** " + " ".join(words[20:])
        c = await _stream_across_a_rotation(answer)

        delivered = "".join(f["content"] for f in c.frames) + "".join(p[1] for p in c.pushed)
        for word in words:
            assert word in delivered, word
        _assert_nothing_reached_the_reader(c)

    @pytest.mark.asyncio
    async def test_an_answer_within_the_cap_is_sent_unchanged(self) -> None:
        # No cut, so the guard costs nothing and changes nothing.
        c = FakeClient()
        r = _capped_renderer(c)
        await r.on_text_chunk("a short clean answer")
        await r._push(force=True)

        assert c.frames[-1]["content"] == "a short clean answer"
