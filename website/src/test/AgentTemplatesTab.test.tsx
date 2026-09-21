/**
 * AgentTemplatesTab — the Agent templates tab under Agent Capabilities.
 *
 * Pins what a management page must not get wrong: the roster groups by origin
 * (Mine / Private copies / From packages / Built-in), a read-only row explains
 * why and offers "Duplicate to edit" instead of Delete, an owned row saves the
 * definition keys through the detail PATCH and refuses to leave a dirty draft
 * silently (including under a background refetch), a delete that the server
 * refuses as referenced opens the reference list instead of a bare error, create
 * sends `from` only for a duplicate, and "Chat with this template" creates a
 * slot in the TEMPLATE namespace. The secondary actions live in one overflow
 * menu (the row holds two controls), so the tests open it the way Radix lets
 * jsdom: keyboard activation of the trigger.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'

const mockApi = vi.hoisted(() => ({
  agentTemplates: vi.fn(),
  agentDetail: vi.fn(),
  agentPatch: vi.fn(),
  agentTemplateCreate: vi.fn(),
  agentTemplateDelete: vi.fn(),
  createKirocrewAgent: vi.fn(),
  skillsCatalog: vi.fn(),
  skills: vi.fn(),
}))
/** Mirrors the real `ApiError`: `body` is the RAW response text, so a test that
 *  hands it an object would pass where the real client fails. */
const StubApiError = vi.hoisted(() => class ApiError extends Error {
  status: number
  body: string
  constructor(status: number, message: string, body: unknown = '') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = typeof body === 'string' ? body : JSON.stringify(body)
  }
})
const mockDispatch = vi.hoisted(() => vi.fn())
const mockNavigate = vi.hoisted(() => vi.fn())
const mockCreateSlot = vi.hoisted(() => vi.fn((opts: unknown) => ({ type: 'chat/createSlot', payload: opts })))

vi.mock('../api/client', () => ({ api: mockApi, ApiError: StubApiError }))
vi.mock('../store', () => ({ useAppDispatch: () => mockDispatch, useAppSelector: () => [] }))
vi.mock('../store/chatSlice', () => ({ createSlot: mockCreateSlot }))
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => mockNavigate }
})
vi.mock('../hooks/useAvailableModels', () => ({ useAvailableModels: () => [{ name: 'claude-x' }, { name: 'gpt-y' }] }))
vi.mock('../components/AgentSkillsEditor', () => ({ default: () => <div data-testid="skills-editor" /> }))

import AgentTemplatesTab from '../pages/overview/AgentTemplatesTab'
import type { TemplateRow } from '../pages/overview/AgentTemplatesTab'

const row = (over: Partial<TemplateRow>): TemplateRow => ({
  name: 'x', filename: 'x.json', description: '', model: '', skills: [], mcp_servers: [],
  source: 'builtin', package: '', scope: 'global', kirocrew_owned: false, forked_from: '', private_to: '',
  read_only: null, used_by: [], ...over,
})
const MINE = row({ name: 'reviewer', filename: 'reviewer.json', description: 'Careful reviewer', model: 'claude-x', used_by: [{ kind: 'crew', id: 'pr-bot', label: 'pr-bot' }] })
const PKG = row({ name: 'atlas', filename: 'Pkg-atlas.json', source: 'package', package: 'Pkg', read_only: 'package' })
const RUNTIME = row({ name: 'kirocrew-worker', filename: 'kirocrew-worker.json', kirocrew_owned: true, read_only: 'runtime' })
const COPY = row({ name: 'pr-bot', filename: 'pr-bot.json', forked_from: 'reviewer', private_to: 'pr-bot', read_only: 'private_copy' })

function renderTab() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter><AgentTemplatesTab /></MemoryRouter>
    </QueryClientProvider>,
  )
}

const option = (name: string) => screen.getByRole('option', { name: new RegExp(`^${name}\\b`) })

/** Open the detail pane's overflow menu. Radix opens on keyboard activation, a
 *  path jsdom handles, unlike the PointerEvent-driven mouse open. */
const openMore = () => {
  fireEvent.keyDown(screen.getByRole('button', { name: 'More actions' }), { key: 'Enter' })
  return screen.findAllByRole('menuitem')
}

beforeEach(() => {
  Object.values(mockApi).forEach(m => m.mockReset())
  mockDispatch.mockReset(); mockNavigate.mockReset(); mockCreateSlot.mockClear()
  mockApi.agentTemplates.mockResolvedValue({ templates: [MINE, PKG, RUNTIME, COPY] })
  mockApi.agentDetail.mockImplementation(async (name: string) => ({
    name, description: name === 'reviewer' ? 'Careful reviewer' : 'shipped', model: 'claude-x',
    prompt: `prompt of ${name}`, tools: ['fs_read', '@docs/search'], allowedTools: ['@docs/search'],
    resources: ['file://AGENTS.md', 'skill://x'], mcpServers: { docs: { command: 'docs-mcp' } }, skills: ['adversarial-review'],
  }))
  mockApi.agentPatch.mockResolvedValue({ ok: true })
  mockApi.agentTemplateCreate.mockResolvedValue({ ok: true, name: 'pr-summarizer', filename: 'pr-summarizer.json' })
  mockApi.agentTemplateDelete.mockResolvedValue({ ok: true })
  mockApi.createKirocrewAgent.mockResolvedValue({ ok: true })
  mockDispatch.mockImplementation(() => ({ unwrap: () => Promise.resolve({ key: 'chat-1' }) }))
})

describe('AgentTemplatesTab roster', () => {
  it('groups rows by origin and marks read-only rows', async () => {
    renderTab()
    await screen.findByRole('option', { name: /reviewer/ })
    const groups = screen.getAllByRole('group').map(g => g.getAttribute('aria-label'))
    expect(groups).toEqual(['Mine', 'Private copies', 'From packages', 'Built-in'])
    expect(within(screen.getByRole('group', { name: 'Mine' })).getByRole('option', { name: /reviewer/ })).toBeInTheDocument()
    expect(within(screen.getByRole('group', { name: 'From packages' })).getByRole('option', { name: /atlas/ })).toBeInTheDocument()
    // The first row is auto-selected and its detail read fires.
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
  })

  it('offers Delete on an owned template and Duplicate to edit on a package one', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    // The row itself holds two controls: the primary action and the menu.
    const row = screen.getByRole('button', { name: 'More actions' }).parentElement!
    expect(within(row).getAllByRole('button')).toHaveLength(2)
    let items = await openMore()
    expect(items.map(i => i.textContent)).toEqual([
      expect.stringContaining('Enroll as crewmate'), 'Duplicate', 'Delete',
    ])
    // The enroll item says what enrolling starts. `pr-bot` runs this template
    // under its own name, so a crewmate NAMED reviewer can still be enrolled.
    expect(items[0]).toHaveTextContent('Adds a crewmate with its own memory')
    expect(items[0]).not.toHaveAttribute('data-disabled')
    fireEvent.keyDown(items[0], { key: 'Escape' })

    fireEvent.click(option('atlas'))
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('atlas'))
    items = await openMore()
    expect(items.map(i => i.textContent)).toEqual([
      expect.stringContaining('Adds a crewmate with its own memory'), 'Duplicate to edit',
    ])
    fireEvent.keyDown(items[0], { key: 'Escape' })
    // The read-only banner carries the reason once and the remedy beside it;
    // the header subtitle only says "Read-only".
    expect(screen.getAllByText(/Installed by a package/)).toHaveLength(1)
    expect(screen.getByText((_, el) => el?.tagName === 'DIV' && el.textContent === 'Read-only · ~/.kiro/agents/Pkg-atlas.json')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Duplicate to edit/ })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Prompt' })).toBeDisabled())
  })

  it('names a chat folder among the references, and an enrolled template says so', async () => {
    mockApi.agentTemplates.mockResolvedValue({
      templates: [row({ name: 'reviewer', filename: 'reviewer.json', used_by: [
        { kind: 'folder', id: 'f-1', label: 'Reviews' }, { kind: 'crew', id: 'reviewer', label: 'reviewer' },
      ] })],
    })
    mockApi.agentTemplateDelete.mockRejectedValue(new StubApiError(409, 'referenced', {
      code: 'template_referenced', references: [{ kind: 'folder', id: 'f-1', label: 'Reviews' }],
    }))
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const items = await openMore()
    expect(items[0]).toHaveTextContent('Already enrolled as a crewmate.')
    expect(items[0]).toHaveAttribute('data-disabled')
    fireEvent.click(items.find(i => i.textContent === 'Delete')!)
    const dialog = await screen.findByRole('dialog', { name: /Can’t delete reviewer yet/ })
    expect(within(dialog).getByText('Chat folder')).toBeInTheDocument()
    expect(within(dialog).getByText('Reviews')).toBeInTheDocument()
  })
})

describe('AgentTemplatesTab editing', () => {
  it('saves the definition keys through the detail PATCH and clears the dirty bar', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const prompt = await screen.findByRole('textbox', { name: 'Prompt' })
    await waitFor(() => expect(prompt).toHaveValue('prompt of reviewer'))
    fireEvent.change(prompt, { target: { value: 'Review carefully.' } })
    expect(screen.getByText(/Unsaved changes/)).toBeInTheDocument()
    // Toggle auto-approval on fs_read; the chip is a pressed button.
    fireEvent.click(screen.getByRole('button', { name: 'fs_read', pressed: false }))
    fireEvent.click(screen.getByRole('button', { name: 'Save template' }))
    await waitFor(() => expect(mockApi.agentPatch).toHaveBeenCalledWith('reviewer', {
      description: 'Careful reviewer',
      prompt: 'Review carefully.',
      tools: ['fs_read', '@docs/search'],
      allowedTools: ['@docs/search', 'fs_read'],
      model: 'claude-x',
    }))
    await waitFor(() => expect(screen.queryByText(/Unsaved changes/)).toBeNull())
    expect(screen.getByRole('status')).toHaveTextContent('Template saved.')
  })

  it('keeps a dirty draft through a background refetch of the detail', async () => {
    // The skills editor saves on its own and invalidates ['agent-templates'],
    // a prefix the detail key shares; the refetch must not reseed the editor
    // over what the user is still typing.
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={qc}><MemoryRouter><AgentTemplatesTab /></MemoryRouter></QueryClientProvider>)
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const prompt = await screen.findByRole('textbox', { name: 'Prompt' })
    await waitFor(() => expect(prompt).toHaveValue('prompt of reviewer'))
    fireEvent.change(prompt, { target: { value: 'still typing' } })
    mockApi.agentDetail.mockImplementation(async (name: string) => ({
      name, prompt: `prompt of ${name}`, tools: ['fs_read'], allowedTools: [], resources: ['skill://y'], skills: ['y'],
    }))
    await qc.invalidateQueries({ queryKey: ['agent-templates'] })
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledTimes(2))
    expect(screen.getByRole('textbox', { name: 'Prompt' })).toHaveValue('still typing')
    expect(screen.getByText(/Unsaved changes · affects 1 crewmate/)).toBeInTheDocument()
    expect(screen.getByText(/needs no restart/)).toBeInTheDocument()
  })

  it('asks before a row switch discards a dirty draft', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const prompt = await screen.findByRole('textbox', { name: 'Prompt' })
    await waitFor(() => expect(prompt).toHaveValue('prompt of reviewer'))
    fireEvent.change(prompt, { target: { value: 'changed' } })
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    fireEvent.click(option('atlas'))
    expect(confirmSpy).toHaveBeenCalled()
    expect(mockApi.agentDetail).not.toHaveBeenCalledWith('atlas')
    confirmSpy.mockRestore()
  })

  it('surfaces a read-only refusal from the server in the pane', async () => {
    mockApi.agentPatch.mockRejectedValue(new StubApiError(409, 'read only', { code: 'template_read_only' }))
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const prompt = await screen.findByRole('textbox', { name: 'Prompt' })
    await waitFor(() => expect(prompt).toHaveValue('prompt of reviewer'))
    fireEvent.change(prompt, { target: { value: 'x' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save template' }))
    await screen.findByText(/read-only here\. Duplicate it/)
  })
})

describe('AgentTemplatesTab detail robustness', () => {
  it('shows the detail error instead of Loading when the detail read is rejected', async () => {
    mockApi.agentDetail.mockRejectedValue(new StubApiError(500, 'boom'))
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    await screen.findByText('This template could not be read.')
    expect(screen.queryByText('Loading…')).toBeNull()
  })

  it('renders only string-valued MCP fields from a hand-edited spec', async () => {
    mockApi.agentDetail.mockImplementation(async (name: string) => ({
      name, prompt: 'p', tools: [], allowedTools: [], resources: [], skills: [],
      mcpServers: {
        docs: { command: { bin: 'docs-mcp' }, url: 42, type: 'stdio' },
        broken: 'not an object',
        remote: { url: 'https://mcp.example.test' },
      },
    }))
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const rows = await screen.findAllByRole('row')
    expect(rows.map(r => r.textContent)).toEqual(['docsstdio', 'broken', 'remotehttps://mcp.example.test'])
  })

  it('shows one Skills heading on an owned template', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    await screen.findByTestId('skills-editor')
    expect(screen.queryByRole('heading', { name: 'Skills' })).toBeNull()
  })
})

describe('AgentTemplatesTab actions', () => {
  it('starts a chat in the template namespace', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    fireEvent.click(screen.getByRole('button', { name: /Chat with this template/ }))
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith('/chat'))
    expect(mockCreateSlot).toHaveBeenCalledWith({ agent: 'reviewer', agent_kind: 'template' })
  })

  it('shows the reference list when the server refuses a delete as referenced', async () => {
    mockApi.agentTemplateDelete.mockRejectedValue(new StubApiError(409, 'referenced', {
      code: 'template_referenced',
      references: [{ kind: 'crew', id: 'pr-bot', label: 'pr-bot' }, { kind: 'schedule', id: 'job-1', label: 'nightly triage' }],
    }))
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    const items = await openMore()
    fireEvent.click(items.find(i => i.textContent === 'Delete')!)
    const dialog = await screen.findByRole('dialog', { name: /Can’t delete reviewer yet/ })
    expect(within(dialog).getByText('nightly triage')).toBeInTheDocument()
    expect(within(dialog).getByText('pr-bot')).toBeInTheDocument()
    // The roster is untouched: the row is still listed.
    expect(option('reviewer')).toBeInTheDocument()
  })

  it('creates a blank template, and a duplicate sends `from`', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    fireEvent.click(screen.getByRole('button', { name: /New template/ }))
    const dialog = await screen.findByRole('dialog', { name: 'New template' })
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Name' }), { target: { value: 'pr-summarizer' } })
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Description' }), { target: { value: 'Sums up PRs' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Create and edit' }))
    await waitFor(() => expect(mockApi.agentTemplateCreate).toHaveBeenCalledWith({ name: 'pr-summarizer', description: 'Sums up PRs' }))
    // Now the duplicate path, seeded from the selected template.
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    const items = await openMore()
    fireEvent.click(items.find(i => i.textContent === 'Duplicate')!)
    const dup = await screen.findByRole('dialog', { name: 'New template' })
    expect(within(dup).getByRole('radio', { name: /Duplicate an existing template/ })).toHaveAttribute('aria-checked', 'true')
    fireEvent.change(within(dup).getByRole('textbox', { name: 'Name' }), { target: { value: 'reviewer-2' } })
    fireEvent.click(within(dup).getByRole('button', { name: 'Create and edit' }))
    await waitFor(() => expect(mockApi.agentTemplateCreate).toHaveBeenLastCalledWith({ name: 'reviewer-2', description: '', from: 'reviewer' }))
  })

  it('refuses a name the server would refuse before sending it', async () => {
    renderTab()
    await waitFor(() => expect(mockApi.agentDetail).toHaveBeenCalledWith('reviewer'))
    fireEvent.click(screen.getByRole('button', { name: /New template/ }))
    const dialog = await screen.findByRole('dialog', { name: 'New template' })
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Name' }), { target: { value: 'has space' } })
    expect(within(dialog).getByRole('button', { name: 'Create and edit' })).toBeDisabled()
  })
})
