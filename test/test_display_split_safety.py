"""``joins_to_a_credential`` / ``safe_split_offset`` -- the cut-safety primitive.

A message cap cuts RAW text while the reader sees the CANONICAL rendering of each
piece, so a credential the model split with markup can be severed by the cut:
each piece is scrubbed on its own and matches nothing, and the reader's client
renders the markup away and rejoins the halves. These pin the primitive that
decides where a cut may fall.
"""

from __future__ import annotations

import pytest

from conftest import CREDENTIAL_STRADDLE_SHAPES
from kiro_crew.messaging.display_safety import (
    canonicalize_display,
    joins_to_a_credential,
    redact_for_display,
    safe_resume_offset,
    safe_split_offset,
)
from kiro_crew.messaging.renderer import _default_redactor


class TestJoinsToACredential:
    @pytest.mark.parametrize(("head", "tail"), CREDENTIAL_STRADDLE_SHAPES)
    def test_a_severed_key_is_reported(self, head: str, tail: str) -> None:
        # Premise first: neither half is a credential ALONE, which is exactly why
        # scrubbing each piece cannot see this and the CUT is what has to be right.
        # Asserted so a fixture that stops straddling fails loudly instead of
        # passing on a case it does not exercise.
        assert _default_redactor(head) == head, "the head half must be clean alone"
        assert _default_redactor(tail) == tail, "the tail half must be clean alone"

        assert joins_to_a_credential(head, tail, _default_redactor)

    @pytest.mark.parametrize(
        ("head", "tail"),
        [
            pytest.param("plain prose ending here", " and continuing there", id="prose"),
            pytest.param("emphasis **spanning", " the cut** is harmless", id="emphasis-span"),
            pytest.param("a [link](https://ex.test/a,b) then", " more prose", id="whole-link"),
            pytest.param("", "AKIAIOSFODNN7EXAMPLE is scrubbed here", id="key-wholly-in-tail"),
            pytest.param("AKIAIOSFODNN7EXAMPLE is scrubbed here", "", id="key-wholly-in-head"),
        ],
    )
    def test_a_harmless_cut_is_allowed(self, head: str, tail: str) -> None:
        # The allow direction. A key that lies wholly inside one side is redacted by
        # that side's own pass, so it must NOT be reported here -- reporting it would
        # walk the cut back for a boundary that severs nothing, and a guard that
        # refuses everything delivers nothing.
        assert not joins_to_a_credential(head, tail, _default_redactor)

    def test_a_cut_that_closes_a_link_is_reported(self) -> None:
        # Caught ONLY by canonicalising the concatenation: each half alone is an
        # unfinished link, and only together do they form a link that collapses to
        # its label, putting the two halves of the key side by side.
        head = "[AKIA](https://ex.test/a,b"
        tail = ")IOSFODNN7EXAMPLE"
        assert joins_to_a_credential(head, tail, _default_redactor)

    def test_a_cut_inside_a_link_target_is_reported(self) -> None:
        # Caught ONLY by canonicalising each side and then joining. Completing the
        # link makes the concatenation collapse to the label, so the key inside the
        # URL disappears from that reading -- while on screen each half is an
        # unfinished link whose URL text stays visible, and the reader reads through.
        head = "[l](https://ex.test/x/AKIAIOSF"
        tail = "ODNN7EXAMPLE)"
        assert joins_to_a_credential(head, tail, _default_redactor)

    @pytest.mark.parametrize(
        ("head", "tail", "seen_by_the_join"),
        [
            pytest.param(
                "[AKIA](https://ex.test/a,b", ")IOSFODNN7EXAMPLE", True, id="cut-closes-a-link"
            ),
            pytest.param(
                "[l](https://ex.test/x/AKIAIOSF", "ODNN7EXAMPLE)", False, id="cut-inside-a-url"
            ),
        ],
    )
    def test_neither_reading_of_a_join_contains_the_other(
        self, head: str, tail: str, seen_by_the_join: bool
    ) -> None:
        # Why BOTH readings are scanned rather than one. Canonicalising the
        # concatenation is the wider reading for delimiter runs, which concatenation
        # can only extend; canonicalising each side first is wider wherever
        # canonicalising DROPS text, which is what a link does to its target. Each
        # shape here is found by exactly one reading, so dropping either reading
        # ships that shape. Pinned so the day one reading starts covering the other,
        # CI says so instead of the guard quietly narrowing.
        head_safe = redact_for_display(head, _default_redactor)[0]
        tail_safe = redact_for_display(tail, _default_redactor)[0]
        joined = canonicalize_display(head_safe + tail_safe)
        on_screen = canonicalize_display(head_safe) + canonicalize_display(tail_safe)

        assert (_default_redactor(joined) != joined) is seen_by_the_join
        assert (_default_redactor(on_screen) != on_screen) is not seen_by_the_join


class TestSafeSplitOffset:
    def test_prose_cuts_at_the_limit(self) -> None:
        text = "just some ordinary prose with nothing secret in it at all"
        assert safe_split_offset(text, 20, _default_redactor) == 20

    def test_text_within_the_limit_is_not_cut(self) -> None:
        text = "short"
        assert safe_split_offset(text, 999, _default_redactor) == len(text)

    def test_a_non_positive_limit_yields_nothing(self) -> None:
        assert safe_split_offset("anything", 0, _default_redactor) == 0

    def test_the_offset_moves_back_off_a_severed_key(self) -> None:
        head, tail = "[AKIA](https://ex.test/a,b)", "IOSFODNN7EXAMPLE"
        pad = "x" * 40
        text = pad + head + tail + " tail prose"
        limit = len(pad) + len(head)

        offset = safe_split_offset(text, limit, _default_redactor)

        assert 0 < offset <= len(pad), "the cut must land before the key begins"
        assert not joins_to_a_credential(text[:offset], text[offset:], _default_redactor)

    def test_the_search_is_logarithmic_not_linear(self) -> None:
        # The cost bound is the reason the candidates step back exponentially: this
        # runs on attacker-influenced text on every outgoing frame. Counting the
        # redaction passes is what pins it -- a linear walk would take ~2000 here.
        calls = 0

        def counting_redactor(text: str) -> str:
            nonlocal calls
            calls += 1
            return _default_redactor(text)

        key = "AKIAIOSFODNN7EXAMPLE"
        text = "x" * 2000 + key[:8] + key[8:] + " tail prose"
        limit = 2000 + 8

        offset = safe_split_offset(text, limit, counting_redactor)

        assert offset <= 2000
        # Four candidates (the limit, then 1, 2, 4, 8 back) at a handful of passes
        # each. The ceiling is deliberately loose: the property is the ORDER, and a
        # linear walk cannot fit under it.
        assert calls < 60, calls


class TestSafeResumeOffset:
    """A head already on screen cannot be shortened, so it must be what is graded.

    :func:`safe_split_offset` moves the head WITH the cut, which is right when both
    pieces are being created now. After a bubble is sealed the head is fixed, and
    grading ``text[:candidate]`` instead judges a head the reader never saw.
    """

    @staticmethod
    def _frozen_and_answer() -> tuple[str, str]:
        """A frozen head whose canonical form ends in a key prefix, behind a run.

        The prefix is spelled as four links, so it is long in RAW characters while
        canonicalising to four. That matters: the candidate ladder steps back 1, 2,
        4, 8 ... from the seam, and every step at or before a SHORT prefix leaves
        the whole key inside the tail, where the tail's own redaction removes it.
        Only a prefix wide enough in raw characters leaves a step clear of it, and
        that step's tail opens with the alphanumeric run instead.
        """
        run = "ABCDEFGHIJKLMNOP"
        prefix = "".join(f"[{c}](https://ex.test/)" for c in "AKIA")
        frozen = run + prefix
        return frozen, frozen + "IOSFODNN7EXAMPLE trailing prose"

    def test_the_frozen_head_is_graded_not_a_shorter_hypothetical_one(self) -> None:
        frozen, answer = self._frozen_and_answer()

        # Premises. Each delivery is clean scanned ALONE -- which is why nothing
        # downstream can catch this -- and the frozen head does end in a key prefix
        # once the platform has rendered its markup away.
        assert _default_redactor(frozen) == frozen, "the frozen head must be clean alone"
        assert canonicalize_display(_default_redactor(frozen)).endswith("AKIA")

        # The discriminator: the partition rule ACCEPTS an offset whose tail opens
        # with the run, because it grades that offset's own prefix as the head. Pin
        # it, so this fixture cannot quietly stop distinguishing the two rules.
        partition_offset = safe_split_offset(answer, len(frozen), _default_redactor)
        assert joins_to_a_credential(
            frozen, answer[partition_offset:], _default_redactor
        ), "fixture no longer discriminates: the partition rule's offset is already safe"

        resume_offset = safe_resume_offset(frozen, answer, len(frozen), _default_redactor)
        assert resume_offset != partition_offset
        assert not joins_to_a_credential(frozen, answer[resume_offset:], _default_redactor)

    def test_withholding_is_the_answer_when_no_candidate_is_safe(self) -> None:
        # A frozen head that completes a key against EVERY sampled resume point,
        # including 0. Delivering nothing is safe and always available; the head
        # cannot be edited, so there is no other remedy.
        frozen = "AKIA"
        answer = "IOSFODNN7EXAMPLE"

        offset = safe_resume_offset(frozen, answer, len(answer), _default_redactor)

        assert offset == len(answer), "no safe resume means deliver nothing yet"
        assert answer[offset:] == ""

    def test_a_clean_resume_stays_at_the_seam(self) -> None:
        # The allow direction: nothing is withheld and nothing is repeated when the
        # join is harmless. A guard that moved the seam on every call would make
        # every long answer repeat itself.
        frozen = "plain prose ending here"
        answer = frozen + " and continuing there"

        assert safe_resume_offset(frozen, answer, len(frozen), _default_redactor) == len(frozen)
