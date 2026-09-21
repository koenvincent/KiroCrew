"""The THIRD keystone scope: ``compaction``, round-tripped through writer and route.

The scope exists so that neither the main switch nor ``tool_args`` can be read as
permission to send a whole transcript. So the tests that matter are the NEGATIVE ones:
what a record written before this key existed means, what an omitted field does, and
what a disable clears.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kiro_crew.decisions import consent

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


@pytest.fixture
def keystone(tmp_path, monkeypatch):
    """A redirected keystone path, so nothing touches the real home."""
    path = tmp_path / "decisions_consent.json"
    monkeypatch.setattr(consent, "consent_path", lambda: path)
    from kiro_crew.config import loader as config_loader

    monkeypatch.setattr(config_loader, "decisions_consent_path", lambda: path, raising=False)
    return path


class TestReader:
    def test_absent_reads_as_not_consented(self, keystone):
        assert consent.consented_compaction() is False

    def test_a_record_predating_the_key_reads_as_not_consented(self, keystone):
        # The whole reason the key exists: an owner who consented to the message
        # excerpt, and even to tool arguments, has not consented to this.
        keystone.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "endpoint": DEFAULT_ENDPOINT,
                    "history_budget_chars": 4000,
                    "tool_args": True,
                }
            ),
            encoding="utf-8",
        )
        assert consent.is_enabled() is True
        assert consent.consented_tool_args() is True
        assert consent.consented_compaction() is False

    @pytest.mark.parametrize("value", ["true", 1, "yes", [], {}, None, 0])
    def test_only_a_literal_true_consents(self, keystone, value):
        keystone.write_text(json.dumps({"compaction": value}), encoding="utf-8")
        assert consent.consented_compaction() is False

    def test_a_corrupt_keystone_reads_as_not_consented(self, keystone):
        keystone.write_text("{not json", encoding="utf-8")
        assert consent.consented_compaction() is False


class TestWriter:
    def test_enabling_records_it(self, keystone):
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=True)
        assert consent.consented_compaction() is True

    def test_the_default_consents_to_none(self, keystone):
        # A caller that does not mention the scope must not grant it.
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT)
        assert consent.consented_compaction() is False

    def test_disabling_clears_it(self, keystone):
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=True)
        consent.save_enabled(False, endpoint=DEFAULT_ENDPOINT)
        assert consent.consented_compaction() is False
        # And a later re-enable cannot inherit it.
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=consent.KEEP_COMPACTION)
        assert consent.consented_compaction() is False

    def test_keep_preserves_a_granted_scope(self, keystone):
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=True)
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=consent.KEEP_COMPACTION)
        assert consent.consented_compaction() is True

    def test_keep_does_not_invent_one(self, keystone):
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=consent.KEEP_COMPACTION)
        assert consent.consented_compaction() is False

    def test_the_three_scopes_move_independently(self, keystone):
        consent.save_enabled(
            True,
            endpoint=DEFAULT_ENDPOINT,
            history_budget_chars=2000,
            tool_args=True,
            compaction=True,
        )
        # Revoking the widest one leaves the other two exactly as they were.
        consent.save_enabled(
            True,
            endpoint=DEFAULT_ENDPOINT,
            history_budget_chars=consent.KEEP_HISTORY_BUDGET,
            tool_args=consent.KEEP_TOOL_ARGS,
            compaction=False,
        )
        assert consent.consented_compaction() is False
        assert consent.consented_tool_args() is True
        assert consent.consented_history_budget() == 2000

    def test_a_non_boolean_is_refused_rather_than_coerced(self, keystone):
        with pytest.raises(ValueError):
            consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction="true")


class TestRoute:
    """``PUT``/``GET /api/decisions/consent``, driven without a socket.

    ``make_mocked_request`` rather than a TestServer: the handler is a coroutine and a
    server per test is what pushes a CI shard past its wall-clock cap.
    """

    def _request(self, body: dict | None):
        from aiohttp.test_utils import make_mocked_request

        request = make_mocked_request(
            "PUT" if body is not None else "GET", "/api/decisions/consent"
        )
        request["user"] = "owner"
        request["app"] = ""
        if body is not None:

            async def _json():
                return body

            request.json = _json  # type: ignore[method-assign]
        return request

    @pytest.fixture(autouse=True)
    def _owner(self, monkeypatch, keystone):
        from kiro_crew.dashboard.handlers import decisions as handler

        monkeypatch.setattr(handler, "is_owner_dashboard_request", lambda _r: True)
        monkeypatch.setattr(handler, "_audit", lambda *_a, **_kw: asyncio.sleep(0))
        from kiro_crew.decisions import capability

        monkeypatch.setattr(capability, "is_decisions_denied", lambda *_a, **_kw: False)
        return handler

    def _body(self, response) -> dict:
        return json.loads(response.body.decode("utf-8"))

    def test_the_get_reports_the_scope(self, _owner):
        response = asyncio.run(_owner.api_decisions_consent_get(self._request(None)))
        assert self._body(response)["compaction"] is False

    def test_the_put_grants_it(self, _owner):
        response = asyncio.run(
            _owner.api_decisions_consent_put(
                self._request({"enabled": True, "endpoint": DEFAULT_ENDPOINT, "compaction": True})
            )
        )
        assert response.status == 200
        assert self._body(response)["compaction"] is True
        assert consent.consented_compaction() is True

    def test_an_omitted_field_preserves_the_recorded_scope(self, _owner):
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=True)
        response = asyncio.run(
            _owner.api_decisions_consent_put(
                self._request({"enabled": True, "endpoint": DEFAULT_ENDPOINT})
            )
        )
        assert self._body(response)["compaction"] is True

    def test_an_explicit_false_revokes_it(self, _owner):
        consent.save_enabled(True, endpoint=DEFAULT_ENDPOINT, compaction=True)
        response = asyncio.run(
            _owner.api_decisions_consent_put(
                self._request({"enabled": True, "endpoint": DEFAULT_ENDPOINT, "compaction": False})
            )
        )
        assert self._body(response)["compaction"] is False

    def test_a_truthy_stand_in_is_a_400(self, _owner):
        response = asyncio.run(
            _owner.api_decisions_consent_put(
                self._request({"enabled": True, "endpoint": DEFAULT_ENDPOINT, "compaction": "true"})
            )
        )
        assert response.status == 400
        assert consent.consented_compaction() is False

    def test_granting_this_scope_does_not_grant_tool_arguments(self, _owner):
        response = asyncio.run(
            _owner.api_decisions_consent_put(
                self._request({"enabled": True, "endpoint": DEFAULT_ENDPOINT, "compaction": True})
            )
        )
        body = self._body(response)
        assert body["compaction"] is True
        assert body["tool_args"] is False
