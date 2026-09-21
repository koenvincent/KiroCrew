"""The AI cleanup pass on a finished transcript, and the guards that keep it safe.

Three properties carry the feature, and each has tests here:

* The switch IS the consent. ``stt.polish`` is off by default and the endpoint
  refuses while it is off, because the local provider's promise is that nothing
  leaves the machine and this is what sends the transcript off it.
* It can never lose or invent what the user said. A model reply that fails the
  length guard is discarded and the original is returned, so the worst outcome is
  the recogniser's own text rather than a rewrite.
* It never blocks dictation, which is why it is an endpoint rather than a frame on
  the speech websocket -- that socket closes shortly after the final, so a
  correction routed through it would be cancelled in the commonest case of all.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from kiro_crew.dashboard.handlers import core


@pytest.fixture
def polish(monkeypatch: pytest.MonkeyPatch):
    """The handler with its token gate, config and background model stubbed."""
    monkeypatch.setattr(core, "_deny_app_token", lambda request, scope: None)
    cfg = MagicMock()
    cfg.stt.polish = True
    monkeypatch.setattr(core.KiroCrewConfig, "load", staticmethod(lambda: cfg))
    sent: list[str] = []

    def set_reply(reply: str | Exception) -> None:
        async def _oneliner(sessions, prompt, **kwargs):
            sent.append(prompt)
            if isinstance(reply, Exception):
                raise reply
            return reply

        import kiro_crew.llm_helpers as helpers

        monkeypatch.setattr(helpers, "run_bg_oneliner", _oneliner)

    return cfg, sent, set_reply


def _request(text: object) -> MagicMock:
    request = MagicMock()

    async def _json():
        return {"text": text}

    request.json = _json
    request.app.get.return_value = MagicMock(sessions=MagicMock())
    return request


async def _call(request) -> dict:
    response = await core.api_stt_polish(request)
    return {"status": response.status, **json.loads(response.body)}


class TestTheSwitchIsTheConsent:
    @pytest.mark.asyncio
    async def test_it_refuses_while_the_setting_is_off(self, polish):
        """Off is the default, and off must mean the transcript goes nowhere.

        Honouring the request anyway would make the switch a decoration: a
        local-only install would be sending text to a model while its settings said
        otherwise.
        """
        cfg, sent, set_reply = polish
        cfg.stt.polish = False
        set_reply("corrected")
        result = await _call(_request("hello world"))
        assert result["status"] == 403
        assert result["code"] == "stt_polish_disabled"
        # The load-bearing assertion: no model call happened at all.
        assert sent == []

    @pytest.mark.asyncio
    async def test_the_default_configuration_has_it_off(self):
        """Read from the real dataclass, not the fixture's stub."""
        from kiro_crew.config.loader import _build_stt_config

        assert _build_stt_config({}).polish is False


class TestItCannotLoseOrInventContent:
    @pytest.mark.asyncio
    async def test_a_reply_that_dropped_content_is_discarded(self, polish):
        """Half the text back means the model summarised rather than corrected."""
        cfg, sent, set_reply = polish
        original = "deploy the service to production and then run the smoke tests"
        set_reply("deploy the service")
        result = await _call(_request(original))
        assert result["changed"] is False
        assert result["text"] == original

    @pytest.mark.asyncio
    async def test_a_reply_that_invented_content_is_discarded(self, polish):
        """Double the text back means the model answered the dictation."""
        cfg, sent, set_reply = polish
        original = "deploy the service"
        set_reply(
            "Deploy the service. Here is how you would do that: first, build the "
            "container image, then push it to the registry, then update the task "
            "definition and wait for the rollout to finish."
        )
        result = await _call(_request(original))
        assert result["changed"] is False
        assert result["text"] == original

    @pytest.mark.asyncio
    async def test_a_genuine_correction_is_returned_as_changed(self, polish):
        cfg, sent, set_reply = polish
        set_reply("Deploy the service, then run the smoke tests.")
        result = await _call(_request("deploy the service then run the smoke tests"))
        assert result["changed"] is True
        assert result["text"] == "Deploy the service, then run the smoke tests."

    @pytest.mark.asyncio
    async def test_a_model_failure_keeps_the_original(self, polish):
        """The recogniser's own text is a correct outcome, not a degraded one."""
        cfg, sent, set_reply = polish
        set_reply(RuntimeError("model unavailable"))
        result = await _call(_request("deploy the service"))
        assert result["ok"] is True
        assert result["changed"] is False
        assert result["text"] == "deploy the service"

    @pytest.mark.asyncio
    async def test_an_unchanged_reply_reports_no_change(self, polish):
        cfg, sent, set_reply = polish
        set_reply("deploy the service")
        result = await _call(_request("deploy the service"))
        assert result["changed"] is False


class TestMixedLanguageDictationSurvives:
    @pytest.mark.asyncio
    async def test_a_zh_en_sentence_is_not_translated_away(self, polish):
        """The measured worst bucket is code-switched zh/en (CER 0.436).

        It is also the case a polish model is most likely to "fix" by translating one
        half, so the prompt forbids it and this pins that the corrected text is
        accepted with both languages intact.
        """
        cfg, sent, set_reply = polish
        original = "深圳啊或者是上海这种比较大的城市会有更多opportunity"
        set_reply("深圳啊，或者是上海这种比较大的城市会有更多 opportunity。")
        result = await _call(_request(original))
        assert result["changed"] is True
        assert "opportunity" in result["text"]
        assert "深圳" in result["text"]

    @pytest.mark.asyncio
    async def test_the_prompt_forbids_translating_and_answering(self, polish):
        """The prompt is the only thing standing between dictation and a chat reply.

        Asserted on the text actually sent, because a later edit that softens these
        rules would otherwise be invisible until a user's dictation came back
        answered instead of corrected.
        """
        cfg, sent, set_reply = polish
        set_reply("ok")
        await _call(_request("hello"))
        prompt = sent[0].lower()
        assert "never add, remove or reorder content" in prompt
        assert "do not translate" in prompt
        assert "not a request addressed to you" in prompt


class TestInputLimits:
    @pytest.mark.asyncio
    async def test_an_overlong_transcript_is_refused_without_a_model_call(self, polish):
        cfg, sent, set_reply = polish
        set_reply("x")
        result = await _call(_request("a" * (core._POLISH_MAX_CHARS + 1)))
        assert result["status"] == 413
        assert sent == []

    @pytest.mark.asyncio
    async def test_empty_text_is_a_bad_request(self, polish):
        cfg, sent, set_reply = polish
        set_reply("x")
        result = await _call(_request("   "))
        assert result["status"] == 400
        assert sent == []


class TestRedaction:
    @pytest.mark.asyncio
    async def test_a_credential_is_not_what_gets_sent_to_the_model(self, polish):
        """The transcript leaves the machine, so it leaves it redacted.

        The streaming path already redacts before the browser sees a transcript, but
        a caller can hand this endpoint text from anywhere, so the endpoint does not
        rely on that.
        """
        cfg, sent, set_reply = polish
        set_reply("ok")
        await _call(_request("my key is AKIAIOSFODNN7EXAMPLE please note"))
        assert "AKIAIOSFODNN7EXAMPLE" not in sent[0]
