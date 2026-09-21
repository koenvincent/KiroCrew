"""Agent template management: the roster the Agent templates tab reads and writes.

A template is the harness-level agent definition (``~/.kiro/agents/<name>.json``)
a chat or a crewmate runs. The crew editor's Template pane edits a crew's
PRIVATE copy of one (blueprint semantics, see ``api_agent_fork``); this module is
the other half -- managing the shared templates themselves:

* ``GET /api/agents/templates`` -- every installed global template with the two
  things a management page needs beside the spec's own fields: whether it can be
  edited here (``read_only`` names why not), and what still points at it
  (``used_by``: crews, the default agent, schedules, chat folders, webhooks,
  private copies).
* ``POST /api/agents/templates`` -- create a template, blank or as a copy of an
  installed one. Nothing is enrolled as a crewmate.
* ``DELETE /api/agents/detail/{name}`` -- delete a template the user owns, refused
  while anything still references it (``409 template_referenced`` lists what).

What is NOT editable here, and why each is a rule rather than a gap:

* a **package** template (``<Package>-<name>.json``): the package rewrites the
  file on its next install, so an edit would be silently reverted -- duplicate
  it to own a copy;
* a **runtime** template (``kirocrew*.json`` in ``OWNED_KIRO_AGENT_FILES``): the
  runtime refreshes these; the same reversal applies;
* a **markdown** spec: one hand-authored document that a JSON round-trip would
  lose fields from (the detail PATCH already refuses it);
* a **private copy** (``private_to`` set): it belongs to one crew and is edited
  from that crew's Template pane, where reset/publish keep its lineage straight.

What the reference guard counts, and the one holder it deliberately does not:

* it counts every CONFIGURATION that names a template -- a crew binding, the
  default agent, a schedule's ``agent_id``, a chat folder's ``default_agent``, a
  webhook token's ``agent``, a private copy's ``forked_from`` -- because each
  silently redirects FUTURE work (a new session, the next cron fire, the next
  webhook call) onto whatever the missing name falls back to;
* it does NOT count an open chat slot that picked the template (``agent_kind:
  "template"``). A slot is a conversation the user has on screen, not a
  configuration that outlives it: its header names the template, and a delete
  guard that passed or refused depending on which tabs happen to be open in
  which window would be one nobody could reason about from the roster. The
  slot's next session start degrades the way any slot naming an unknown agent
  does (kiro-cli falls back to the default spec), visibly, in that chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from aiohttp import web

from kiro_crew import agent_state
from kiro_crew.agent import agents_spec_lock, kiro_agents_dir_path
from kiro_crew.agent_capabilities import CapabilityError, require_unmanaged_template
from kiro_crew.agent_discovery import (
    AgentInfo,
    _global_agent_info,
    _read_agent_spec,
    clear_list_agents_cache,
    list_agents,
)
from kiro_crew.agent_files import OWNED_KIRO_AGENT_FILES
from kiro_crew.agent_spec_format import is_markdown_spec
from kiro_crew.config.loader import KiroCrewConfig, update_config_locked
from kiro_crew.config.paths import config_dir
from kiro_crew.cron import _CRONS_FILE, _read_job_records
from kiro_crew.dashboard.state import DashboardState
from kiro_crew.executors import discovery_executor
from kiro_crew.webhooks import token_store

logger = logging.getLogger(__name__)

#: Longest prompt / description the dashboard editor writes. kiro-cli imposes no
#: cap, but a spec is read on every session start, and a multi-megabyte prompt
#: from a paste accident is a defect, not a feature.
MAX_TEMPLATE_PROMPT_CHARS = 200_000
MAX_TEMPLATE_DESCRIPTION_CHARS = 2_000
#: Tool references per list. Generous -- a wildcard covers a whole server -- but
#: finite, like ``MAX_AGENT_SKILLS`` on the skills list.
MAX_TEMPLATE_TOOLS = 200
#: Longest single tool reference. A real one is ``@server/tool`` or a wildcard,
#: tens of characters; the cap keeps the list's total size bounded by count.
MAX_TEMPLATE_TOOL_CHARS = 256

#: ``(folder_id, folder_name, default_agent)`` for one chat folder, read from
#: the folder store on the loop (the roster snapshots it; the delete guard
#: reads it under the store lock, see ``api_agent_template_delete``).
FolderPin = tuple[str, str, str]

#: The spec keys the templates tab may write through the detail PATCH, beyond the
#: ``model`` and ``skills`` the crew pane already writes. ``resources`` and
#: ``mcpServers`` are deliberately absent: skills are a computed view OVER
#: ``resources`` (writing both would race), and an MCP server is a capability
#: grant that has its own admission path (the MCP page and its quarantine).
TEMPLATE_DEFINITION_KEYS = frozenset({"description", "prompt", "tools", "allowedTools"})

_READ_ONLY_PACKAGE = "package"
_READ_ONLY_RUNTIME = "runtime"
_READ_ONLY_MARKDOWN = "markdown"
_READ_ONLY_PRIVATE_COPY = "private_copy"


def template_read_only_reason(info: AgentInfo) -> str | None:
    """Why *info* cannot be edited or deleted from the templates tab, or ``None``.

    Order matters only for the label: a markdown package spec is reported as a
    package spec, because duplicating it is the remedy either way.
    """
    if info.source == "package":
        return _READ_ONLY_PACKAGE
    if info.kirocrew_owned or info.filename in OWNED_KIRO_AGENT_FILES:
        return _READ_ONLY_RUNTIME
    if is_markdown_spec(info.filename):
        return _READ_ONLY_MARKDOWN
    if info.private_to:
        return _READ_ONLY_PRIVATE_COPY
    return None


def _cron_agent_ids() -> list[tuple[str, str, str]]:
    """``(job_id, job_name, agent_id)`` for every stored job naming an agent.

    A pure file read, like ``count_enabled_from_disk``: the scheduler's in-memory
    snapshot belongs to the event loop and is not touched from here.
    """
    records, _loadable = _read_job_records(config_dir() / _CRONS_FILE)
    out: list[tuple[str, str, str]] = []
    for job in records:
        agent_id = job.get("agent_id")
        if isinstance(agent_id, str) and agent_id:
            out.append(
                (str(job.get("id", "")), str(job.get("name") or job.get("id", "")), agent_id)
            )
    return out


def _webhook_agent_ids() -> list[tuple[str, str, str]]:
    """``(token_id, label, agent)`` for every webhook token pinned to an agent.

    A pinned token's calls are refused (``409 destination_agent_unavailable``)
    once the agent is gone, so the hook stops working rather than falling back;
    still a holder the guard must count -- a delete that breaks an integration
    is the same silent breakage one shelf over.
    """
    out: list[tuple[str, str, str]] = []
    for entry in token_store().list_entries():
        agent = entry.get("agent")
        if isinstance(agent, str) and agent:
            out.append(
                (str(entry.get("id", "")), str(entry.get("label") or entry.get("id", "")), agent)
            )
    return out


def template_references(
    aliases: set[str],
    cfg: KiroCrewConfig,
    forks: dict[str, dict],
    crons: list[tuple[str, str, str]],
    folders: list[FolderPin],
    webhooks: list[tuple[str, str, str]],
) -> list[dict[str, str]]:
    """Everything that still resolves one of *aliases* (a template's name and stem).

    Each row is ``{"kind", "id", "label"}``, ``kind`` one of ``crew``, ``default``,
    ``schedule``, ``folder``, ``webhook``, ``private_copy``. The list doubles as
    the delete guard's evidence and the roster's ``used_by``. A chat folder
    counts because its ``default_agent`` is what every new session filed there
    starts on; a template deleted underneath it would fall through to the
    default agent silently, the exact failure the guard exists to refuse.
    """
    refs: list[dict[str, str]] = []
    for crew, crew_cfg in sorted(cfg.agents.items()):
        if crew_cfg.kiro_agent in aliases:
            refs.append({"kind": "crew", "id": crew, "label": crew})
    if aliases & {cfg.agent.default_agent, cfg.default_agent}:
        refs.append({"kind": "default", "id": "default", "label": "default agent"})
    for job_id, job_name, agent_id in crons:
        if agent_id in aliases:
            refs.append({"kind": "schedule", "id": job_id, "label": job_name})
    for folder_id, folder_name, default_agent in folders:
        if default_agent in aliases:
            refs.append({"kind": "folder", "id": folder_id, "label": folder_name})
    for token_id, label, agent in webhooks:
        if agent in aliases:
            refs.append({"kind": "webhook", "id": token_id, "label": label})
    for copy_name, info in sorted(forks.items()):
        if info.get("forked_from") in aliases:
            refs.append(
                {"kind": "private_copy", "id": copy_name, "label": str(info.get("private_to", ""))}
            )
    return refs


def _folder_pins_of(folders: list[dict[str, Any]]) -> list[FolderPin]:
    """The agent pins among *folders* (the store's committed list)."""
    return [
        (str(f.get("id", "")), str(f.get("name", "")), str(f.get("default_agent") or ""))
        for f in folders
        if f.get("default_agent")
    ]


async def _folder_pins(state: DashboardState) -> list[FolderPin]:
    """Snapshot the folder store's agent pins on the loop, for the executor."""
    return await state.read_folders(_folder_pins_of)


def _template_rows(folders: list[FolderPin]) -> list[dict[str, Any]]:
    """Thread-side: the roster with editability and references attached."""
    infos = list(list_agents(agents_dir=kiro_agents_dir_path()))
    cfg = KiroCrewConfig.load()
    forks = agent_state.all_fork_info()
    crons = _cron_agent_ids()
    webhooks = _webhook_agent_ids()
    rows: list[dict[str, Any]] = []
    for info in infos:
        aliases = {info.name, Path(info.filename).stem}
        row = info.to_dict()
        row["read_only"] = template_read_only_reason(info)
        row["used_by"] = template_references(aliases, cfg, forks, crons, folders, webhooks)
        rows.append(row)
    rows.sort(key=lambda r: (r["source"] != "builtin" or r["kirocrew_owned"], r["name"].lower()))
    return rows


async def api_agent_templates(request: web.Request) -> web.Response:
    """GET /api/agents/templates -- the management roster of installed templates.

    Global scope only, like ``/api/agents/installed``: a project template is a
    file in that checkout and is edited there, and a management action taken
    here persists into the global agents directory.
    """
    state: DashboardState = request.app["state"]
    try:
        folders = await _folder_pins(state)
        rows = await asyncio.get_running_loop().run_in_executor(
            discovery_executor(), _template_rows, folders
        )
    except Exception:
        logger.warning("Agent templates roster could not be loaded", exc_info=True)
        return web.json_response(
            {
                "error": "Agent templates could not be loaded. Retry.",
                "code": "templates_unavailable",
            },
            status=503,
        )
    return web.json_response({"templates": rows})


def _find_info(name: str) -> AgentInfo | None:
    for info in list_agents(agents_dir=kiro_agents_dir_path()):
        if info.name == name or Path(info.filename).stem == name:
            return info
    return None


def _blank_spec(name: str, description: str) -> dict[str, Any]:
    """The smallest spec kiro-cli runs: read-only tools, no skills, model auto."""
    return {
        "name": name,
        "description": description,
        "prompt": "",
        "tools": ["fs_read", "grep", "glob"],
        "allowedTools": ["fs_read", "grep", "glob"],
    }


async def api_agent_template_create(request: web.Request) -> web.Response:
    """POST /api/agents/templates -- create a template, blank or copied.

    Body: ``{"name", "description"?, "from"?}``. ``from`` names an installed
    template to copy (any source -- copying a package template is exactly how a
    user comes to own an editable version of it). The copy carries no fork
    lineage: it is a real, shareable template, not a crew's private copy.
    """
    from kiro_crew.dashboard.handlers.agents import (
        _TEMPLATE_NAME_RE,
        _AmbiguousTemplateName,
        _get_config_lock,
        _is_reserved_basename,
        _load_template_specs,
        _require_owner,
        _reserved_binding_names,
        _spec_stem_on_disk,
        _write_spec_file,
    )

    denied = await _require_owner(request, "agent_templates.create")
    if denied is not None:
        return denied
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"error": "invalid JSON", "code": "invalid_json"}, status=400)
    if not isinstance(body, dict):
        return web.json_response(
            {"error": "body must be a JSON object", "code": "invalid_body"}, status=400
        )
    raw_name = body.get("name")
    if not isinstance(raw_name, str) or not _TEMPLATE_NAME_RE.match(raw_name.strip()):
        return web.json_response(
            {
                "error": "name must be 1-63 letters, digits, dots, dashes or underscores",
                "code": "invalid_template_name",
            },
            status=400,
        )
    name: str = raw_name.strip()
    if _is_reserved_basename(name) or f"{name.lower()}.json" in {
        f.lower() for f in OWNED_KIRO_AGENT_FILES
    }:
        return web.json_response(
            {"error": f"'{name}' is reserved", "code": "template_name_reserved"}, status=400
        )
    description = body.get("description", "")
    if not isinstance(description, str) or len(description) > MAX_TEMPLATE_DESCRIPTION_CHARS:
        return web.json_response(
            {"error": "description must be a short string", "code": "invalid_description"},
            status=400,
        )
    source_name = body.get("from")
    if source_name is not None and (not isinstance(source_name, str) or not source_name.strip()):
        return web.json_response(
            {"error": "from must name an installed template", "code": "invalid_source"}, status=400
        )

    state: DashboardState = request.app["state"]
    async with _get_config_lock():
        agents_dir = kiro_agents_dir_path()
        source: dict[str, Any] | None = None
        # Resolving either name can find two files declaring it (``atlas.json``
        # beside ``SomePkg-atlas.json``); that is the user's to untangle, so it
        # is a 409 naming the problem, not a 500.
        try:
            if source_name:
                source, _resolved, taken, _path = await asyncio.to_thread(
                    _load_template_specs,
                    agents_dir,
                    source_name.strip(),
                    "api_agent_template_create",
                )
                if source is None:
                    return web.json_response(
                        {
                            "error": f"Template '{source_name}' not found",
                            "code": "template_not_found",
                        },
                        status=404,
                    )
            else:
                _none, _resolved, taken, _path = await asyncio.to_thread(
                    _load_template_specs, agents_dir, name, "api_agent_template_create"
                )
        except _AmbiguousTemplateName as exc:
            return web.json_response(
                {
                    "error": f"'{exc}' matches more than one template file; rename one first.",
                    "code": "ambiguous_template_name",
                },
                status=409,
            )
        # A member of this name would make the new template unreachable by bare
        # name (name-first resolution answers the member) -- refused, like publish.
        if name.lower() in taken:
            return web.json_response(
                {"error": f"A template named '{name}' already exists", "code": "name_taken"},
                status=409,
            )

        def _create() -> Path:
            created: list[Path] = []

            def _check_then_write(cfg_data: dict) -> None:
                if name.lower() in _reserved_binding_names(cfg_data):
                    raise _NameBound()
                if name in (cfg_data.get("agents") or {}):
                    raise _NameBound()
                with agents_spec_lock(agents_dir):
                    if _spec_stem_on_disk(agents_dir, name):
                        raise FileExistsError(name)
                    data = dict(source) if source else _blank_spec(name, description)
                    data["name"] = name
                    if description or not source:
                        data["description"] = description
                    dest = agents_dir / f"{name}.json"
                    agent_state.prune(name)
                    agent_state.lift_and_strip_bookkeeping(data, name)
                    _write_spec_file(dest, data)
                    created.append(dest)

            update_config_locked(mutate=_check_then_write)
            return created[0]

        try:
            dest = await asyncio.to_thread(_create)
        except _NameBound:
            return web.json_response(
                {"error": f"A crew is bound to the name '{name}'", "code": "name_bound"},
                status=409,
            )
        except FileExistsError:
            return web.json_response(
                {"error": f"A template named '{name}' already exists", "code": "name_taken"},
                status=409,
            )
        except Exception:
            logger.exception("template create failed for %r", name)
            return web.json_response(
                {"error": "Could not write the template", "code": "template_write_failed"},
                status=500,
            )
    clear_list_agents_cache()
    state.push_refresh("agents")
    return web.json_response({"ok": True, "name": name, "filename": dest.name}, status=201)


class _NameBound(Exception):
    """A crew binding already resolves the requested template name."""


async def api_agent_template_delete(request: web.Request) -> web.Response:
    """DELETE /api/agents/detail/{name} -- delete a template the user owns.

    Refused with ``409 template_read_only`` for a package, runtime, markdown or
    private-copy spec, and with ``409 template_referenced`` (carrying the list)
    while a crew, the default agent, a schedule, a chat folder, a webhook or a
    private copy still names it: deleting underneath them would leave sessions
    that cannot start, or that silently start on the default agent.

    The check and the unlink are one critical section against the stores this
    handler can serialize with: the config lock covers crew bindings, the
    default agent and the fork sidecar (their writers hold it too), and the
    final reference check runs INSIDE the folder store's lock together with the
    unlink, so a folder pinned between the check and the unlink is impossible
    -- the pin waits for the lock and then sees the template gone, which the
    folder editor already reports as an orphan. Schedules and webhook tokens
    are read at check time; their writers hold neither lock.
    """
    from kiro_crew.dashboard.handlers.agents import _get_config_lock, _require_owner

    name = request.match_info["name"]
    denied = await _require_owner(request, "agent_templates.delete")
    if denied is not None:
        return denied
    state: DashboardState = request.app["state"]
    async with _get_config_lock():
        info = await asyncio.get_running_loop().run_in_executor(
            discovery_executor(), _find_info, name
        )
        if info is None:
            return web.json_response(
                {"error": f"Template '{name}' not found", "code": "template_not_found"}, status=404
            )
        found: AgentInfo = info
        reason = template_read_only_reason(found)
        if reason is not None:
            return web.json_response(
                {
                    "error": f"Template '{name}' is read-only ({reason})",
                    "code": "template_read_only",
                    "reason": reason,
                },
                status=409,
            )
        try:
            for identity in dict.fromkeys((Path(found.filename).stem, found.name)):
                await asyncio.to_thread(require_unmanaged_template, identity)
        except CapabilityError as exc:
            return web.json_response({"error": exc.code, "code": exc.code}, status=exc.status)

        aliases = {name, found.name, Path(found.filename).stem}

        def _load_stores() -> tuple[KiroCrewConfig, dict[str, dict], list, list]:
            # The file-backed stores, read off the loop. The config lock held
            # above serializes the first two with their writers.
            return (
                KiroCrewConfig.load(),
                agent_state.all_fork_info(),
                _cron_agent_ids(),
                _webhook_agent_ids(),
            )

        def _guard_then_unlink(folders: list[dict[str, Any]]) -> list[dict[str, str]]:
            # Runs under the folder store's lock (``read_folders``), on the loop:
            # the folder pins read here are the committed ones and no pin can
            # land before the unlink below. The work is a set lookup and one
            # unlink of a small file, not a scan.
            refs = template_references(
                aliases, cfg, forks, crons, _folder_pins_of(folders), webhooks
            )
            if refs:
                return refs
            agents_dir = kiro_agents_dir_path()
            path = agents_dir / found.filename
            with agents_spec_lock(agents_dir):
                path.unlink()
                with contextlib.suppress(Exception):
                    agent_state.prune(found.name)
            return []

        try:
            cfg, forks, crons, webhooks = await asyncio.to_thread(_load_stores)
            refs = await state.read_folders(_guard_then_unlink)
        except FileNotFoundError:
            return web.json_response(
                {"error": f"Template '{name}' not found", "code": "template_not_found"}, status=404
            )
        except Exception:
            logger.exception("template delete failed for %r", name)
            return web.json_response(
                {"error": "Could not delete the template", "code": "template_delete_failed"},
                status=500,
            )
        if refs:
            return web.json_response(
                {
                    "error": f"Cannot delete '{name}': still referenced",
                    "code": "template_referenced",
                    "references": refs,
                },
                status=409,
            )
    clear_list_agents_cache()
    state.push_refresh("agents")
    return web.json_response({"ok": True})


def validate_definition_patch(patch_body: dict[str, Any]) -> str | None:
    """Shape-check the definition keys of a detail PATCH; the error text or None."""
    for key in ("description", "prompt"):
        if key in patch_body:
            value = patch_body[key]
            cap = MAX_TEMPLATE_PROMPT_CHARS if key == "prompt" else MAX_TEMPLATE_DESCRIPTION_CHARS
            if not isinstance(value, str):
                return f"{key} must be a string"
            if len(value) > cap:
                return f"{key} is longer than {cap} characters"
    for key in ("tools", "allowedTools"):
        if key in patch_body:
            value = patch_body[key]
            if not isinstance(value, list) or not all(isinstance(t, str) and t for t in value):
                return f"{key} must be a list of non-empty strings"
            if len(value) > MAX_TEMPLATE_TOOLS:
                return f"at most {MAX_TEMPLATE_TOOLS} entries in {key}"
            if any(len(t) > MAX_TEMPLATE_TOOL_CHARS for t in value):
                return f"each {key} entry must be at most {MAX_TEMPLATE_TOOL_CHARS} characters"
    return None


def apply_definition_patch(data: dict[str, Any], patch_body: dict[str, Any]) -> None:
    """Write the validated definition keys into *data* (the spec about to be saved)."""
    for key in TEMPLATE_DEFINITION_KEYS:
        if key in patch_body:
            data[key] = patch_body[key]


def read_only_reason_for_path(path: Path) -> str | None:
    """The templates-tab read-only rule for THE spec file the PATCH is about to write.

    Classified from the file itself -- its name and its declared ``name`` -- the
    way discovery classifies every row, never by looking the declared name up in
    the deduplicated roster: two files can declare one name (``atlas.json``
    beside ``SomePkg-atlas.json``), and the roster keeps only the package twin,
    so a lookup would answer for the wrong file and let the package copy through
    as if it were the plain one.
    """
    if path.name in OWNED_KIRO_AGENT_FILES:
        return _READ_ONLY_RUNTIME
    if is_markdown_spec(path):
        return _READ_ONLY_MARKDOWN
    spec = _read_agent_spec(path, operation="api_agent_detail", source="dashboard")
    info = _global_agent_info(path, spec if isinstance(spec, dict) else {})
    fork = agent_state.all_fork_info().get(info.name)
    if fork:
        info.private_to = str(fork.get("private_to", ""))
    return template_read_only_reason(info)
