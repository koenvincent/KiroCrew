import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileCode2, MessageSquare, UserPlus, Copy, Trash2, Lock, Plus, X, Check, MoreHorizontal } from 'lucide-react'
import { useAppDispatch } from '../../store'
import { createSlot } from '../../store/chatSlice'
import { api, ApiError } from '../../api/client'
import { Card, CardTitle, Btn, Badge, SearchInput, EmptyState, PanelSectionHeader } from '../../components/ui'
import Modal from '../../components/Modal'
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from '../../components/ui/dropdown-menu'
import ErrorNotice from '../../components/ErrorNotice'
import ListDetailBack from '../../components/ListDetailBack'
import SimpleSelect from '../../components/SimpleSelect'
import AgentSkillsEditor from '../../components/AgentSkillsEditor'
import { useSidePanelLeaveGuard } from '../../components/SidePanelLayout'
import { useListDetailView } from '../../hooks/useListDetailView'
import { useAvailableModels } from '../../hooks/useAvailableModels'
import { parseErrorCode } from '../../utils/errorReport'
import { errMessage } from '../../utils/thunkError'
import { templateSourceKind } from '../../lib/templateSource'
import { i18nT } from '../../i18n/t'

/** One row of `GET /api/agents/templates`: the discovery record plus the two
 *  facts a management page needs — whether it may be edited here, and what
 *  still points at it. */
export interface TemplateRow {
  name: string
  filename: string
  description: string
  model: string
  skills: string[]
  mcp_servers: string[]
  source: string
  package: string
  scope: string
  kirocrew_owned: boolean
  forked_from: string
  private_to: string
  read_only: 'package' | 'runtime' | 'markdown' | 'private_copy' | null
  used_by: TemplateReference[]
}

export interface TemplateReference {
  kind: 'crew' | 'default' | 'schedule' | 'folder' | 'webhook' | 'private_copy'
  id: string
  label: string
}

/** `GET /api/agents/detail/{name}`: the spec passed through, plus the two
 *  computed views (`skills`, `unmanaged_skills`) over `resources`. */
interface TemplateDetail {
  name?: string
  description?: string
  model?: string
  prompt?: string
  tools?: unknown
  allowedTools?: unknown
  resources?: unknown
  mcpServers?: unknown
  skills?: string[]
  unmanaged_skills?: string[]
}

/** The editable definition. `allowed` mirrors `allowedTools`: the tools the
 *  agent may call without asking. */
interface Draft {
  description: string
  model: string
  prompt: string
  tools: string[]
  allowed: string[]
}

const PANE_SHELL_CLASS = 'flex gap-3 -mx-2 md:mx-0 h-[calc(100vh-260px)] supports-[height:100svh]:h-[calc(100svh-260px)] min-h-[420px]'

const strList = (v: unknown): string[] => Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []

const draftFrom = (d: TemplateDetail): Draft => ({
  description: typeof d.description === 'string' ? d.description : '',
  model: typeof d.model === 'string' ? d.model : '',
  prompt: typeof d.prompt === 'string' ? d.prompt : '',
  tools: strList(d.tools),
  allowed: strList(d.allowedTools),
})

const sameList = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i])
const sameDraft = (a: Draft, b: Draft) =>
  a.description === b.description && a.model === b.model && a.prompt === b.prompt
  && sameList(a.tools, b.tools) && sameList(a.allowed, b.allowed)

/** A template group in the list. `custom` is what the user owns and can edit
 *  here; the other three each say why a row is read-only. */
type GroupKey = 'custom' | 'private' | 'package' | 'builtin'
const GROUP_ORDER: GroupKey[] = ['custom', 'private', 'package', 'builtin']

const readOnlyHint = (reason: TemplateRow['read_only']): string => {
  switch (reason) {
    case 'package': return i18nT('pages.overview.agentTemplatesTab.read_only_package')
    case 'runtime': return i18nT('pages.overview.agentTemplatesTab.read_only_runtime')
    case 'markdown': return i18nT('pages.overview.agentTemplatesTab.read_only_markdown')
    case 'private_copy': return i18nT('pages.overview.agentTemplatesTab.read_only_private_copy')
    default: return ''
  }
}

const referenceKindLabel = (kind: TemplateReference['kind']): string => {
  switch (kind) {
    case 'crew': return i18nT('pages.overview.agentTemplatesTab.ref_crewmate')
    case 'default': return i18nT('pages.overview.agentTemplatesTab.ref_default')
    case 'schedule': return i18nT('pages.overview.agentTemplatesTab.ref_schedule')
    case 'folder': return i18nT('pages.overview.agentTemplatesTab.ref_folder')
    case 'webhook': return i18nT('pages.overview.agentTemplatesTab.ref_webhook')
    case 'private_copy': return i18nT('pages.overview.agentTemplatesTab.ref_private_copy')
  }
}

/** The `references` list of a `409 template_referenced` body. `ApiError.body`
 *  is the raw response text, so the structured field is read out of it here. */
const referencesIn = (body: string): TemplateReference[] => {
  try {
    const parsed = JSON.parse(body) as { references?: unknown }
    return Array.isArray(parsed.references)
      ? parsed.references.filter((r): r is TemplateReference => !!r && typeof r === 'object' && typeof (r as TemplateReference).kind === 'string')
      : []
  } catch {
    return []
  }
}

/** Where a reference lives, so the delete guard's rows can be followed. */
const referenceHref = (ref: TemplateReference): string | null => {
  switch (ref.kind) {
    case 'crew': return `/members?member=${encodeURIComponent(ref.id)}`
    case 'schedule': return '/schedule'
    case 'folder': return '/chat'
    case 'webhook': return '/webhooks'
    case 'default': return '/capabilities?tab=crews'
    default: return null
  }
}

function ChipList({ items, onRemove, onToggle, marked, addLabel, onAdd, readOnly }: {
  items: string[]
  /** Tools marked ✓ — auto-approved. Only the tools section passes this. */
  marked?: Set<string>
  onToggle?: (item: string) => void
  onRemove?: (item: string) => void
  addLabel?: string
  onAdd?: (item: string) => void
  readOnly?: boolean
}) {
  const [adding, setAdding] = useState(false)
  const [value, setValue] = useState('')
  const commit = () => {
    const v = value.trim()
    if (v && onAdd) onAdd(v)
    setValue(''); setAdding(false)
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(item => {
        const isMarked = marked?.has(item)
        return (
          <span key={item} className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-[11.5px] font-mono bg-bg-elevated ${isMarked ? 'border-accent/40' : 'border-border-strong'}`}>
            {onToggle && !readOnly ? (
              <button
                type="button"
                className={`inline-flex items-center gap-1 ${isMarked ? 'text-accent' : 'text-text'}`}
                title={isMarked ? i18nT('pages.overview.agentTemplatesTab.tool_auto_approved') : i18nT('pages.overview.agentTemplatesTab.tool_asks_first')}
                aria-pressed={!!isMarked}
                onClick={() => onToggle(item)}
              >
                {item}{isMarked && <Check className="lucide-inline" aria-hidden />}
              </button>
            ) : (
              <span className={isMarked ? 'text-accent inline-flex items-center gap-1' : ''}>{item}{isMarked && <Check className="lucide-inline" aria-hidden />}</span>
            )}
            {onRemove && !readOnly && (
              <button type="button" className="text-muted hover:text-danger" aria-label={i18nT('pages.overview.agentTemplatesTab.remove_item', { item })} title={i18nT('pages.overview.agentTemplatesTab.remove_item_hint')} onClick={() => onRemove(item)}>
                <X className="lucide-inline" aria-hidden />
              </button>
            )}
          </span>
        )
      })}
      {onAdd && !readOnly && (adding ? (
        <input
          autoFocus
          className="px-2 py-1 rounded-md border border-border bg-bg text-[11.5px] font-mono text-text w-44"
          value={value}
          placeholder={addLabel}
          aria-label={addLabel}
          onChange={e => setValue(e.target.value)}
          onBlur={commit}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit() } if (e.key === 'Escape') { setValue(''); setAdding(false) } }}
        />
      ) : (
        <button type="button" className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-dashed border-border-strong text-[11.5px] text-muted hover:text-text hover:border-border-strong" onClick={() => setAdding(true)}>
          <Plus className="lucide-inline" aria-hidden />{addLabel}
        </button>
      ))}
      {items.length === 0 && (readOnly || !onAdd) && (
        <span className="text-[12px] text-muted italic">{i18nT('pages.overview.agentTemplatesTab.none')}</span>
      )}
    </div>
  )
}

function Section({ title, hint, children }: { title: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="mt-5">
      <h3 className="flex items-center gap-2 mb-2 text-[11px] font-semibold tracking-wider uppercase text-muted">
        {title}
        {hint && <span className="font-normal tracking-normal normal-case text-muted-strong">{hint}</span>}
      </h3>
      {children}
    </section>
  )
}

export default function AgentTemplatesTab() {
  const queryClient = useQueryClient()
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const { isMobile, showList, showDetail, openDetail, closeDetail } = useListDetailView()
  const availableModels = useAvailableModels()
  const models = useMemo(() => (availableModels || []).map(m => m.name).filter(Boolean), [availableModels])

  const [filter, setFilter] = useState('')
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [baseline, setBaseline] = useState<Draft | null>(null)
  const [notice, setNotice] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)
  const [creating, setCreating] = useState(false)
  const [createForm, setCreateForm] = useState<{ name: string; description: string; from: string }>({ name: '', description: '', from: '' })
  const [blocked, setBlocked] = useState<{ name: string; references: TemplateReference[] } | null>(null)

  const { data, isLoading, error, refetch } = useQuery<{ templates: TemplateRow[] }>({
    queryKey: ['agent-templates'],
    queryFn: () => api.agentTemplates(),
    // Templates are files another window, a package install or an editor can
    // change underneath us; nothing pushes an invalidation for those.
    staleTime: 0,
  })
  const rows = useMemo(() => data?.templates ?? [], [data])
  const selected = useMemo(() => rows.find(r => r.name === selectedName) ?? null, [rows, selectedName])

  const detailQuery = useQuery<TemplateDetail>({
    queryKey: ['agent-templates', 'detail', selectedName],
    queryFn: () => api.agentDetail(selectedName!),
    enabled: !!selectedName,
    staleTime: 0,
  })
  const dirty = !!draft && !!baseline && !sameDraft(draft, baseline)
  // Seed the editor from THIS template's own detail response, never from the
  // previous one: the header already names the new template while the fetch
  // is in flight, and a draft seeded from the old body would save A's prompt
  // under B's name. Never over a dirty draft either: the skills editor saves
  // on its own and invalidates the detail query, and a refetch that reseeded
  // would silently drop the prompt the user is still typing. A row switch
  // clears the draft first, so a clean editor is the only one reseeded.
  useEffect(() => {
    if (dirty || !detailQuery.data || detailQuery.isFetching) return
    const d = draftFrom(detailQuery.data)
    setDraft(d); setBaseline(d)
  }, [detailQuery.data, detailQuery.isFetching, dirty])
  const editable = !!selected && selected.read_only === null
  // Enrolling creates a crewmate NAMED after the template, so that is the
  // name that decides whether it has already happened.
  const enrolled = !!selected && selected.used_by.some(u => u.kind === 'crew' && u.id === selected.name)
  const dirtyMessage = i18nT('pages.overview.agentTemplatesTab.discard_unsaved_changes')
  // The rail click belongs to the shell; it consults this guard before
  // unmounting the pane and taking an unsaved draft with it.
  useSidePanelLeaveGuard(useCallback(() => !dirty || confirm(dirtyMessage), [dirty, dirtyMessage]), dirty)

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['agent-templates'] })
  const writeError = (e: unknown) => {
    const code = e instanceof ApiError ? parseErrorCode(e.body) : undefined
    setNotice({
      kind: 'err',
      text: code === 'template_read_only'
        ? i18nT('pages.overview.agentTemplatesTab.err_read_only')
        : code === 'name_taken' || code === 'name_bound'
          ? i18nT('pages.overview.agentTemplatesTab.err_name_taken')
          : code === 'invalid_template_name'
            ? i18nT('pages.overview.agentTemplatesTab.name_rule')
            : errMessage(e) || i18nT('pages.overview.agentTemplatesTab.err_generic'),
    })
  }

  const save = useMutation({
    mutationFn: ({ name, d }: { name: string; d: Draft }) => api.agentPatch(name, {
      description: d.description,
      prompt: d.prompt,
      tools: d.tools,
      allowedTools: d.allowed,
      model: d.model,
    }),
    onSuccess: (_r, vars) => {
      // Adopt only if this template is still the one on screen.
      if (vars.name === selectedName) setBaseline(vars.d)
      setNotice({ kind: 'ok', text: i18nT('pages.overview.agentTemplatesTab.saved') })
      invalidate()
    },
    onError: writeError,
  })
  const create = useMutation({
    mutationFn: (f: { name: string; description: string; from: string }) =>
      api.agentTemplateCreate({ name: f.name.trim(), description: f.description.trim(), ...(f.from ? { from: f.from } : {}) }),
    onSuccess: (r: { name: string }) => {
      setCreating(false)
      setCreateForm({ name: '', description: '', from: '' })
      setNotice(null)
      setSelectedName(r.name)
      openDetail()
      invalidate()
    },
    onError: writeError,
  })
  const remove = useMutation({
    mutationFn: (name: string) => api.agentTemplateDelete(name),
    onSuccess: () => { setSelectedName(null); setNotice(null); invalidate() },
    onError: (e: unknown, name) => {
      if (e instanceof ApiError && parseErrorCode(e.body) === 'template_referenced') {
        setBlocked({ name, references: referencesIn(e.body) })
        return
      }
      writeError(e)
    },
  })
  const enroll = useMutation({
    mutationFn: (row: TemplateRow) => api.createKirocrewAgent({ name: row.name, kiro_agent: row.name, memory_store: 'default' }),
    onSuccess: (r: { error?: string }, row) => {
      if (r?.error) { setNotice({ kind: 'err', text: r.error }); return }
      setNotice({ kind: 'ok', text: i18nT('pages.overview.agentTemplatesTab.enrolled', { name: row.name }) })
      invalidate()
    },
    onError: writeError,
  })

  const chatWith = async (row: TemplateRow) => {
    try {
      // The template namespace, so a same-name crewmate is not what answers.
      await dispatch(createSlot({ agent: row.name, agent_kind: 'template' })).unwrap()
      navigate('/chat')
    } catch (e) {
      setNotice({ kind: 'err', text: errMessage(e) || i18nT('pages.overview.agentTemplatesTab.err_generic') })
    }
  }

  const select = (row: TemplateRow) => {
    if (row.name === selectedName) { openDetail(); return }
    if (dirty && !confirm(dirtyMessage)) return
    setNotice(null)
    setDraft(null); setBaseline(null)
    setSelectedName(row.name)
    openDetail()
  }

  const matches = useCallback((r: TemplateRow) =>
    !filter || `${r.name} ${r.description} ${r.package} ${r.source}`.toLowerCase().includes(filter.toLowerCase()), [filter])
  const grouped = useMemo(() => {
    const g: Record<GroupKey, TemplateRow[]> = { custom: [], private: [], package: [], builtin: [] }
    for (const r of rows) if (matches(r)) g[templateSourceKind(r)].push(r)
    return g
  }, [rows, matches])
  const allFiltered = useMemo(() => GROUP_ORDER.flatMap(k => grouped[k]), [grouped])

  // Keep a desktop detail pane populated; never yank a dirty editor.
  useEffect(() => {
    if (dirty) return
    if (allFiltered.length === 0) return
    if (!selectedName || !rows.some(r => r.name === selectedName)) setSelectedName(allFiltered[0].name)
  }, [allFiltered, rows, selectedName, dirty])

  const groupLabel = (k: GroupKey) => {
    switch (k) {
      case 'custom': return i18nT('pages.overview.agentTemplatesTab.group_mine')
      case 'private': return i18nT('pages.overview.agentTemplatesTab.group_private_copies')
      case 'package': return i18nT('pages.overview.agentTemplatesTab.group_packages')
      case 'builtin': return i18nT('pages.overview.agentTemplatesTab.group_builtin')
    }
  }

  const renderRow = (r: TemplateRow) => {
    const isSel = r.name === selectedName
    const crews = r.used_by.filter(u => u.kind === 'crew').length
    const copies = r.used_by.filter(u => u.kind === 'private_copy').length
    return (
      <div
        key={r.filename}
        role="option"
        aria-selected={isSel}
        tabIndex={0}
        onClick={() => select(r)}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(r) } }}
        className={`flex flex-col gap-0.5 px-3 py-2 rounded-md cursor-pointer mb-1 transition-colors ${isSel ? 'list-selected bg-accent-subtle' : 'hover:bg-bg-hover'}`}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <span className={`text-[13px] font-semibold font-mono truncate flex-1 ${isSel ? 'text-accent' : 'text-text'}`}>{r.name}</span>
          {r.read_only && <Lock className="lucide-inline text-muted shrink-0" aria-label={readOnlyHint(r.read_only)} />}
        </div>
        {r.description && <span className="text-[11px] text-muted truncate">{r.description}</span>}
        <div className="flex flex-wrap gap-1 mt-0.5">
          <Badge variant="muted" title={i18nT('pages.overview.agentTemplatesTab.model')}><span className="font-mono">{i18nT('pages.overview.agentTemplatesTab.model_badge', { model: r.model || 'auto' })}</span></Badge>
          {crews > 0 && <Badge variant="ok">{i18nT('pages.overview.agentTemplatesTab.crewmates_count', { count: crews })}</Badge>}
          {copies > 0 && <Badge variant="warn">{i18nT('pages.overview.agentTemplatesTab.private_copies_count', { count: copies })}</Badge>}
          {r.package && <Badge variant="aim">{r.package}</Badge>}
        </div>
      </div>
    )
  }

  const usedByLine = (r: TemplateRow) => {
    const crews = r.used_by.filter(u => u.kind === 'crew').map(u => u.label)
    const schedules = r.used_by.filter(u => u.kind === 'schedule').length
    const copies = r.used_by.filter(u => u.kind === 'private_copy').length
    const isDefault = r.used_by.some(u => u.kind === 'default')
    return (
      <p className="text-[12px] text-muted mt-3 flex flex-wrap gap-x-3 gap-y-1">
        <span>
          {crews.length
            ? i18nT('pages.overview.agentTemplatesTab.runs_as_crewmates', { count: crews.length, names: crews.join(', ') })
            : i18nT('pages.overview.agentTemplatesTab.runs_as_no_crewmate')}
        </span>
        {isDefault && <span>{i18nT('pages.overview.agentTemplatesTab.is_default_agent')}</span>}
        <span>{i18nT('pages.overview.agentTemplatesTab.schedules_count', { count: schedules })}</span>
        <span>{i18nT('pages.overview.agentTemplatesTab.private_copies_count', { count: copies })}</span>
      </p>
    )
  }

  const openCreate = (from = '') => { setNotice(null); setCreateForm({ name: '', description: '', from }); setCreating(true) }
  const nameProblem = (n: string) => n.trim() && !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$/.test(n.trim())
  const setD = (patch: Partial<Draft>) => setDraft(d => d ? { ...d, ...patch } : d)

  const detailDraft = draft
  const detailReadOnly = !editable
  // A spec is a user-editable file: any of these may be missing, an object or a
  // number, and only a string is a renderable React child.
  const str = (v: unknown): string => typeof v === 'string' ? v : ''
  const mcpServers = detailQuery.data && typeof detailQuery.data.mcpServers === 'object' && detailQuery.data.mcpServers && !Array.isArray(detailQuery.data.mcpServers)
    ? Object.entries(detailQuery.data.mcpServers as Record<string, unknown>).map(([name, cfg]) => {
      const c = cfg && typeof cfg === 'object' ? cfg as Record<string, unknown> : {}
      return [name, str(c.url) || str(c.command) || str(c.type)] as const
    })
    : []
  const resources = strList(detailQuery.data?.resources).filter(u => !u.startsWith('skill://'))

  return (<>
    <Card>
      <CardTitle>
        <FileCode2 className="lucide-inline" /> {i18nT('pages.overview.agentTemplatesTab.title')}
        <span className="ml-auto"><Btn primary onClick={() => openCreate()}><Plus className="lucide-inline" aria-hidden /> {i18nT('pages.overview.agentTemplatesTab.new_template')}</Btn></span>
      </CardTitle>
      <p className="text-muted text-[13px] mb-3 leading-relaxed">{i18nT('pages.overview.agentTemplatesTab.intro')}</p>
      {rows.length > 0 && (
        <div className="mb-3"><SearchInput placeholder={i18nT('pages.overview.agentTemplatesTab.filter')} value={filter} onChange={e => setFilter(e.target.value)} /></div>
      )}
      {isLoading && <p className="text-muted italic text-sm px-3 py-4">{i18nT('pages.overview.agentTemplatesTab.loading')}</p>}
      {error && <ErrorNotice message={i18nT('pages.overview.agentTemplatesTab.load_failed')} askAgent className="mb-2" />}
      {error && <Btn className="mb-3" onClick={() => void refetch()}>{i18nT('pages.overview.agentTemplatesTab.retry')}</Btn>}
      {!isLoading && !error && rows.length === 0 ? (
        <EmptyState
          icon={<FileCode2 className="lucide-inline" />}
          title={i18nT('pages.overview.agentTemplatesTab.empty_title')}
          subtitle={i18nT('pages.overview.agentTemplatesTab.empty_subtitle')}
          action={<Btn primary onClick={() => openCreate()}>{i18nT('pages.overview.agentTemplatesTab.new_template')}</Btn>}
        />
      ) : rows.length > 0 && (
        <div className={PANE_SHELL_CLASS}>
          {showList && (
            <div className={`${isMobile ? 'w-full' : 'w-[300px]'} shrink-0 overflow-y-auto scrollbar-overlay border border-border rounded-md p-2`} role="listbox" aria-label={i18nT('pages.overview.agentTemplatesTab.title')}>
              {GROUP_ORDER.map(k => grouped[k].length > 0 && (
                <div key={k} role="group" aria-label={groupLabel(k)}>
                  <PanelSectionHeader label={groupLabel(k)} count={grouped[k].length} className="px-2 pt-2 pb-1" />
                  {grouped[k].map(renderRow)}
                </div>
              ))}
              {allFiltered.length === 0 && <div className="text-muted/70 text-[12px] italic px-2 py-2">{i18nT('pages.overview.agentTemplatesTab.no_match', { query: filter })}</div>}
            </div>
          )}

          {showDetail && (
            <div className="flex-1 min-w-0 flex flex-col border border-border rounded-md bg-card overflow-hidden relative">
              {!selected ? (
                <div className="flex items-center justify-center h-full text-muted text-[13px]">{i18nT('pages.overview.agentTemplatesTab.select_one')}</div>
              ) : (
                <div className="flex flex-col h-full min-h-0">
                  {isMobile && <div className="px-4 pt-2.5 shrink-0"><ListDetailBack label={i18nT('pages.overview.agentTemplatesTab.title')} onBack={closeDetail} /></div>}
                  <div className="flex items-start justify-between gap-3 flex-wrap px-4 py-3 border-b border-border shrink-0">
                    <div className="min-w-0">
                      <div className="text-[17px] font-mono font-semibold text-text-strong truncate">{selected.name}</div>
                      <div className="text-[12px] text-muted mt-0.5">
                        {selected.read_only ? i18nT('pages.overview.agentTemplatesTab.read_only_short') : i18nT('pages.overview.agentTemplatesTab.your_template')}
                        {' · '}<code className="text-[11.5px]">~/.kiro/agents/{selected.filename}</code>
                      </div>
                    </div>
                    {/* Two controls in the row: the primary action and one overflow
                        menu holding the rest. The read-only remedy (Duplicate to
                        edit) sits in the read-only banner below, next to its reason. */}
                    <div className="flex flex-wrap gap-2 justify-end shrink-0">
                      <Btn onClick={() => void chatWith(selected)} disabled={dirty} title={dirty ? i18nT('pages.overview.agentTemplatesTab.chat_with_dirty_hint') : i18nT('pages.overview.agentTemplatesTab.chat_with_hint')}><MessageSquare className="lucide-inline" aria-hidden /> {i18nT('pages.overview.agentTemplatesTab.chat_with')}</Btn>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Btn className="!px-1.5" aria-label={i18nT('pages.overview.agentTemplatesTab.more_actions')} title={i18nT('pages.overview.agentTemplatesTab.more_actions')}>
                            <MoreHorizontal className="lucide-inline" aria-hidden />
                          </Btn>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="min-w-[260px]">
                          {!selected.private_to && (
                            <DropdownMenuItem disabled={enroll.isPending || enrolled} onSelect={() => enroll.mutate(selected)} className="items-start">
                              <UserPlus className="lucide-inline mt-0.5 shrink-0 text-muted" aria-hidden />
                              <span className="flex flex-col gap-0.5">
                                <span>{i18nT('pages.overview.agentTemplatesTab.enroll')}</span>
                                {/* What enrolling starts, so the click is not a leap: a
                                    crewmate with its own memory, and nothing running. */}
                                <span className="text-[11.5px] text-muted whitespace-normal">
                                  {enrolled ? i18nT('pages.overview.agentTemplatesTab.enroll_already') : i18nT('pages.overview.agentTemplatesTab.enroll_hint')}
                                </span>
                              </span>
                            </DropdownMenuItem>
                          )}
                          <DropdownMenuItem onSelect={() => openCreate(selected.name)}>
                            <Copy className="lucide-inline shrink-0 text-muted" aria-hidden />
                            <span>{editable ? i18nT('pages.overview.agentTemplatesTab.duplicate') : i18nT('pages.overview.agentTemplatesTab.duplicate_to_edit')}</span>
                          </DropdownMenuItem>
                          {editable && (
                            <DropdownMenuItem disabled={remove.isPending} className="text-danger focus:text-danger" onSelect={() => {
                              if (confirm(i18nT('pages.overview.agentTemplatesTab.delete_confirm', { name: selected.name }))) remove.mutate(selected.name)
                            }}>
                              <Trash2 className="lucide-inline shrink-0" aria-hidden />
                              <span>{i18nT('pages.overview.agentTemplatesTab.delete')}</span>
                            </DropdownMenuItem>
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </div>

                  <div className="flex-1 min-h-0 overflow-y-auto px-4 pb-24">
                    {usedByLine(selected)}
                    {selected.read_only ? (
                      <div className="mt-3 flex items-start gap-2 px-3 py-2 rounded-md border border-aim/35 bg-aim-subtle text-[12.5px] text-text">
                        <Lock className="lucide-inline mt-0.5 shrink-0" aria-hidden />
                        <span className="flex-1">{readOnlyHint(selected.read_only)} {selected.read_only !== 'private_copy' && i18nT('pages.overview.agentTemplatesTab.duplicate_hint')}</span>
                        {selected.read_only !== 'private_copy' && (
                          <Btn primary className="shrink-0" onClick={() => openCreate(selected.name)}><Copy className="lucide-inline" aria-hidden /> {i18nT('pages.overview.agentTemplatesTab.duplicate_to_edit')}</Btn>
                        )}
                      </div>
                    ) : (
                      <div className="mt-3 px-3 py-2 rounded-md border border-border bg-bg-elevated text-[12.5px] text-text">
                        {i18nT('pages.overview.agentTemplatesTab.editing_shared_hint')}
                      </div>
                    )}
                    {/* No hand-off: the notice sits over the template draft below, which
                        a refused save leaves unsaved; leaving with the agent would drop it. */}
                    {notice && (
                      notice.kind === 'err'
                        ? <ErrorNotice message={notice.text} className="mt-3" />
                        : <p className="mt-3 text-[12.5px] text-ok" role="status">{notice.text}</p>
                    )}

                    {/* The error branch comes first: a rejected detail read leaves the
                        draft null, and a draft-gated loading branch would mask it forever. */}
                    {detailQuery.error ? (
                      <ErrorNotice message={i18nT('pages.overview.agentTemplatesTab.detail_failed')} askAgent className="mt-4" />
                    ) : detailQuery.isLoading || !detailDraft ? (
                      <p className="text-muted text-[12px] italic mt-4">{i18nT('pages.overview.agentTemplatesTab.loading')}</p>
                    ) : (<>
                      <Section title={i18nT('pages.overview.agentTemplatesTab.definition')}>
                        <div className="grid grid-cols-[140px_1fr] gap-x-3 gap-y-2 items-start">
                          <label className="text-[12.5px] text-muted pt-1.5" htmlFor="tpl-description">{i18nT('pages.overview.agentTemplatesTab.description')}</label>
                          <input id="tpl-description" aria-label={i18nT('pages.overview.agentTemplatesTab.description')} className="w-full px-2.5 py-1.5 rounded-md border border-border bg-bg-elevated text-[12.5px] text-text disabled:opacity-70" value={detailDraft.description} disabled={detailReadOnly} onChange={e => setD({ description: e.target.value })} />
                          <span className="text-[12.5px] text-muted pt-1.5">{i18nT('pages.overview.agentTemplatesTab.model')}</span>
                          <div className="max-w-[320px]">
                            <SimpleSelect
                              aria-label={i18nT('pages.overview.agentTemplatesTab.model')}
                              options={detailDraft.model && !models.includes(detailDraft.model) ? ['', detailDraft.model, ...models] : ['', ...models]}
                              clearLabel={i18nT('pages.overview.agentTemplatesTab.model_auto')}
                              value={detailDraft.model}
                              onChange={v => setD({ model: v })}
                              disabled={detailReadOnly}
                            />
                          </div>
                        </div>
                      </Section>
                      <Section title={i18nT('pages.overview.agentTemplatesTab.prompt')} hint={i18nT('pages.overview.agentTemplatesTab.chars_count', { count: detailDraft.prompt.length })}>
                        <textarea
                          aria-label={i18nT('pages.overview.agentTemplatesTab.prompt')}
                          className="w-full min-h-[200px] px-3 py-2.5 rounded-md border border-border bg-bg text-[12px] leading-relaxed font-mono text-text disabled:opacity-70"
                          value={detailDraft.prompt}
                          disabled={detailReadOnly}
                          spellCheck={false}
                          onChange={e => setD({ prompt: e.target.value })}
                        />
                      </Section>
                      <Section title={i18nT('pages.overview.agentTemplatesTab.tools')} hint={i18nT('pages.overview.agentTemplatesTab.tools_hint')}>
                        <ChipList
                          items={detailDraft.tools}
                          marked={new Set(detailDraft.allowed)}
                          readOnly={detailReadOnly}
                          onToggle={t => setD({ allowed: detailDraft.allowed.includes(t) ? detailDraft.allowed.filter(x => x !== t) : [...detailDraft.allowed, t] })}
                          onRemove={t => setD({ tools: detailDraft.tools.filter(x => x !== t), allowed: detailDraft.allowed.filter(x => x !== t) })}
                          onAdd={t => { if (!detailDraft.tools.includes(t)) setD({ tools: [...detailDraft.tools, t] }) }}
                          addLabel={i18nT('pages.overview.agentTemplatesTab.add_tool')}
                        />
                      </Section>
                      {editable ? (
                        // The editor renders its own "Skills" heading; wrapping it in a
                        // Section would stack two.
                        <div className="mt-5">
                          <AgentSkillsEditor
                            agentName={selected.name}
                            skills={detailQuery.data?.skills ?? []}
                            unmanaged={detailQuery.data?.unmanaged_skills ?? []}
                            onChange={() => { void queryClient.invalidateQueries({ queryKey: ['agent-templates'] }) }}
                          />
                        </div>
                      ) : (
                        <Section title={i18nT('pages.overview.agentTemplatesTab.skills')}>
                          <ChipList items={detailQuery.data?.skills ?? []} readOnly />
                        </Section>
                      )}
                      <Section title={i18nT('pages.overview.agentTemplatesTab.resources')} hint={<><Lock className="lucide-inline" aria-hidden /> {i18nT('pages.overview.agentTemplatesTab.read_only_here')}</>}>
                        <ChipList items={resources} readOnly />
                      </Section>
                      <Section title={i18nT('pages.overview.agentTemplatesTab.mcp_servers')} hint={<><Lock className="lucide-inline" aria-hidden /> {i18nT('pages.overview.agentTemplatesTab.mcp_read_only_hint')}</>}>
                        {mcpServers.length === 0 ? (
                          <span className="text-[12px] text-muted italic">{i18nT('pages.overview.agentTemplatesTab.none')}</span>
                        ) : (
                          <table className="w-full text-[12px]">
                            <tbody>
                              {mcpServers.map(([name, where]) => (
                                <tr key={name} className="border-t border-border">
                                  <td className="py-1.5 pr-3 font-mono text-text">{name}</td>
                                  <td className="py-1.5 text-muted font-mono truncate">{where}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                      </Section>
                    </>)}
                  </div>

                  {dirty && editable && (
                    <div className="absolute left-4 right-4 bottom-4 flex items-center gap-3 px-4 py-2.5 rounded-lg border border-border-strong bg-bg-elevated shadow-lg">
                      <span className="flex flex-col gap-0.5 min-w-0">
                        <span className="text-[12.5px] text-text">
                          {i18nT('pages.overview.agentTemplatesTab.unsaved_changes')}
                          {' · '}
                          {i18nT('pages.overview.agentTemplatesTab.affects_crewmates', { count: selected.used_by.filter(u => u.kind === 'crew').length })}
                        </span>
                        {/* Save vs the page header's Apply & Restart: saving writes the
                            file and needs no restart, said here where both are visible. */}
                        <span className="text-[11.5px] text-muted">{i18nT('pages.overview.agentTemplatesTab.save_hint')}</span>
                      </span>
                      <span className="flex-1" />
                      <Btn disabled={save.isPending} onClick={() => { if (baseline) setDraft(baseline) }}>{i18nT('pages.overview.agentTemplatesTab.discard')}</Btn>
                      <Btn primary disabled={save.isPending} onClick={() => { if (draft) save.mutate({ name: selected.name, d: draft }) }}>{i18nT('pages.overview.agentTemplatesTab.save')}</Btn>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Card>

    <Modal open={creating} onClose={() => { if (!create.isPending) { setCreating(false); setNotice(null) } }} title={i18nT('pages.overview.agentTemplatesTab.new_template')} maxWidth={560} guardAccidentalDismiss footer={<>
      <Btn disabled={create.isPending} onClick={() => { setCreating(false); setNotice(null) }}>{i18nT('pages.overview.agentTemplatesTab.cancel')}</Btn>
      <Btn primary disabled={!createForm.name.trim() || !!nameProblem(createForm.name) || create.isPending} onClick={() => create.mutate(createForm)}>{i18nT('pages.overview.agentTemplatesTab.create_and_edit')}</Btn>
    </>}>
      <p className="text-[12.5px] text-muted mb-3">{i18nT('pages.overview.agentTemplatesTab.new_template_intro')}</p>
      <div className="grid grid-cols-2 gap-2 mb-3" role="radiogroup" aria-label={i18nT('pages.overview.agentTemplatesTab.start_from')}>
        {[{ key: '', title: i18nT('pages.overview.agentTemplatesTab.start_blank'), sub: i18nT('pages.overview.agentTemplatesTab.start_blank_sub') },
          { key: '__copy__', title: i18nT('pages.overview.agentTemplatesTab.start_copy'), sub: i18nT('pages.overview.agentTemplatesTab.start_copy_sub') }].map(opt => {
          const on = opt.key === '' ? !createForm.from : !!createForm.from
          return (
            <button key={opt.key} type="button" role="radio" aria-checked={on}
              className={`text-left p-3 rounded-md border bg-bg ${on ? 'border-accent ring-1 ring-accent' : 'border-border-strong'}`}
              onClick={() => setCreateForm(f => ({ ...f, from: opt.key === '' ? '' : (f.from || selected?.name || rows[0]?.name || '') }))}>
              <span className="block text-text-strong font-semibold text-[13px]">{opt.title}</span>
              <span className="text-muted text-[12px]">{opt.sub}</span>
            </button>
          )
        })}
      </div>
      {createForm.from && (
        <div className="mb-3">
          <label className="block text-[12px] text-muted mb-1">{i18nT('pages.overview.agentTemplatesTab.copy_of')}</label>
          <SimpleSelect aria-label={i18nT('pages.overview.agentTemplatesTab.copy_of')} options={rows.filter(r => !r.private_to).map(r => r.name)} value={createForm.from} onChange={v => setCreateForm(f => ({ ...f, from: v }))} />
        </div>
      )}
      <label className="block text-[12px] text-muted mb-1" htmlFor="tpl-new-name">{i18nT('pages.overview.agentTemplatesTab.name')}</label>
      <input id="tpl-new-name" aria-label={i18nT('pages.overview.agentTemplatesTab.name')} className="w-full px-2.5 py-1.5 rounded-md border border-border bg-bg-elevated text-[12.5px] font-mono text-text" value={createForm.name} onChange={e => setCreateForm(f => ({ ...f, name: e.target.value }))} autoFocus />
      <p className={`text-[11.5px] mt-1 ${nameProblem(createForm.name) ? 'text-danger' : 'text-muted'}`}>{i18nT('pages.overview.agentTemplatesTab.name_rule')}</p>
      <label className="block text-[12px] text-muted mb-1 mt-3" htmlFor="tpl-new-desc">{i18nT('pages.overview.agentTemplatesTab.description')}</label>
      <input id="tpl-new-desc" aria-label={i18nT('pages.overview.agentTemplatesTab.description')} className="w-full px-2.5 py-1.5 rounded-md border border-border bg-bg-elevated text-[12.5px] text-text" value={createForm.description} onChange={e => setCreateForm(f => ({ ...f, description: e.target.value }))} placeholder={i18nT('pages.overview.agentTemplatesTab.description_placeholder')} />
      {/* No hand-off: the name and description typed into this dialog are unsaved
          until the create succeeds. */}
      {notice?.kind === 'err' && <ErrorNotice message={notice.text} className="mt-3" />}
    </Modal>

    <Modal open={!!blocked} onClose={() => setBlocked(null)} title={i18nT('pages.overview.agentTemplatesTab.cannot_delete_title', { name: blocked?.name ?? '' })} maxWidth={520} footer={<Btn onClick={() => setBlocked(null)}>{i18nT('pages.overview.agentTemplatesTab.close')}</Btn>}>
      <p className="text-[12.5px] text-muted mb-3">{i18nT('pages.overview.agentTemplatesTab.cannot_delete_body', { count: blocked?.references.length ?? 0 })}</p>
      <div className="border border-border rounded-md overflow-hidden">
        {(blocked?.references ?? []).map(ref => {
          const href = referenceHref(ref)
          return (
            <div key={`${ref.kind}:${ref.id}`} className="flex items-center gap-3 px-3 py-2 border-t border-border first:border-t-0 text-[12.5px]">
              <span className="text-muted w-24 shrink-0 text-[11.5px]">{referenceKindLabel(ref.kind)}</span>
              <span className="font-mono text-text truncate">{ref.label || ref.id}</span>
              {href && <button type="button" className="ml-auto text-accent text-[12px]" onClick={() => { setBlocked(null); navigate(href) }}>{i18nT('pages.overview.agentTemplatesTab.open')} →</button>}
            </div>
          )
        })}
      </div>
    </Modal>
  </>)
}
