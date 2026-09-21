/**
 * Your crew in the cloud: the panel's own guards and its five states.
 *
 * The helpers are asserted DIRECTLY as well as through the rendered panel: each
 * one is a decision with a wrong answer that is invisible on screen (an
 * unreachable machine offered as reachable looks identical to a reachable one;
 * a stale attempt described as the current one reads exactly like the real
 * one), and a pure call cannot pass by finding a healthy sibling row.
 *
 * The render tests pin one claim per state, plus the claim that binds them: a
 * failed read and "no launch yet" must never render the same thing. Saying
 * "your crew runs only on this computer" when the truth is "we could not find
 * out" tells a reader their crew is not deployed, which is the one wrong answer
 * this panel can give that a reader would act on.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { renderWithProviders } from '../../test/helpers'
import DeployMyCrewDialog, {
  deployAge,
  deployHasShellTarget,
  deployRunsAsTask,
  deployIsReachable,
  deployProgress,
  deployView,
  newestLaunch,
} from './DeployMyCrew'
import type { LaunchJob, LaunchJobStatus, LaunchStep } from '../../api/client'

const cloudLaunches = vi.fn()
vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>()
  return { ...actual, api: { cloudLaunches: (...a: unknown[]) => cloudLaunches(...a) } }
})

const navigateSpy = vi.fn()
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return { ...actual, useNavigate: () => navigateSpy }
})

/** A `done` job carrying a machine, so a test that wants anything else has to
 *  say which field it changed. */
function job(over: Partial<LaunchJob> = {}): LaunchJob {
  return {
    id: 'j1',
    provider_id: 'aws_ec2',
    profile: 'dev',
    region: 'us-west-2',
    size_key: 'small',
    tag: 'crew-alpha',
    status: 'done',
    steps: [],
    instance_id: 'i-0abc123def4567890',
    created_at: 1_700_000_000,
    updated_at: 1_700_000_000,
    ...over,
  }
}

function step(state: LaunchStep['state']): LaunchStep {
  return { key: state, label: state, state }
}

const MEMBERS = [{ name: 'Ada' }, { name: 'Grace' }]

describe('deployIsReachable', () => {
  it('is true only for a finished launch that carries a machine', () => {
    expect(deployIsReachable(job())).toBe(true)
  })

  it('is false while the launch is still working', () => {
    for (const status of ['pending', 'running', 'awaiting_signin'] as LaunchJobStatus[]) {
      expect(deployIsReachable(job({ status })), status).toBe(false)
    }
  })

  // The case worth the test: a failed or cancelled attempt can still carry the
  // instance_id of the machine it got as far as creating. Offering that as
  // reachable sends the reader at something half-built or already gone, and
  // every field except `status` reads exactly like the healthy row.
  it('is false for a failed or cancelled launch even when it kept its machine', () => {
    for (const status of ['failed', 'cancelled'] as LaunchJobStatus[]) {
      const j = job({ status })
      expect(j.instance_id, 'fixture must keep the machine or this proves nothing').toBeTruthy()
      expect(deployIsReachable(j), status).toBe(false)
    }
  })

  it('is false for a finished launch with no machine recorded', () => {
    expect(deployIsReachable(job({ instance_id: undefined }))).toBe(false)
  })
})

describe('deployHasShellTarget', () => {
  it('is true for a finished EC2 launch that kept its machine', () => {
    expect(deployHasShellTarget(job())).toBe(true)
  })

  it('treats a record with no provisioner id as EC2, the only kind that predates the field', () => {
    expect(deployHasShellTarget(job({ provider_id: undefined as unknown as string }))).toBe(true)
  })

  // The Fargate lane writes the task ARN into instance_id. Session Manager
  // does not accept an ARN, so that launch is reachable yet has no address.
  it('is false for a finished Fargate launch even though it is reachable', () => {
    const j = job({ provider_id: 'aws_fargate', instance_id: 'arn:aws:ecs:us-west-2:1:task/c/t' })
    expect(deployIsReachable(j)).toBe(true)
    expect(deployHasShellTarget(j)).toBe(false)
  })

  it('is false whenever the launch is not reachable', () => {
    expect(deployHasShellTarget(job({ status: 'failed' }))).toBe(false)
    expect(deployHasShellTarget(job({ instance_id: undefined }))).toBe(false)
  })
})

describe('deployRunsAsTask', () => {
  it('is true for the Fargate lane', () => {
    expect(deployRunsAsTask(job({ provider_id: 'aws_fargate' }))).toBe(true)
  })

  it('is false for EC2, and for a record that predates the provisioner field', () => {
    expect(deployRunsAsTask(job())).toBe(false)
    expect(deployRunsAsTask(job({ provider_id: undefined as unknown as string }))).toBe(false)
  })

  // Keyed on the Fargate id, not on "anything but EC2": a provisioner this
  // panel does not know is drawn verbatim in Details and gets no sentence,
  // because "runs as a container task" could be false of it.
  it('is false for a provisioner the panel does not know', () => {
    expect(deployRunsAsTask(job({ provider_id: 'some_future_lane' }))).toBe(false)
  })

  // Not the negation of deployHasShellTarget: an EC2 launch with no machine has
  // no target either, but "runs as a container task" would be false of it.
  it('stays false for an EC2 launch that recorded no machine', () => {
    const j = job({ instance_id: undefined })
    expect(deployHasShellTarget(j)).toBe(false)
    expect(deployRunsAsTask(j)).toBe(false)
  })
})

describe('newestLaunch', () => {
  it('is undefined for no launches', () => {
    expect(newestLaunch([])).toBeUndefined()
  })

  // The API returns newest first today; the panel must not depend on it. A list
  // handed over oldest-first would otherwise make a stale attempt the story.
  it('picks by created_at, not by list order', () => {
    const older = job({ id: 'old', created_at: 100 })
    const newer = job({ id: 'new', created_at: 200 })
    expect(newestLaunch([older, newer])?.id).toBe('new')
    expect(newestLaunch([newer, older])?.id).toBe('new')
  })
})

describe('deployView', () => {
  it('is none with no launches', () => {
    expect(deployView([])).toEqual({ kind: 'none' })
  })

  it('is deploying for a pending or running launch', () => {
    for (const status of ['pending', 'running'] as LaunchJobStatus[]) {
      expect(deployView([job({ status })]).kind, status).toBe('deploying')
    }
  })

  // Split from the other moving statuses on purpose: waiting achieves nothing
  // here, because the launch is waiting on the user.
  it('is signin, not deploying, while the launch waits for the user', () => {
    expect(deployView([job({ status: 'awaiting_signin' })]).kind).toBe('signin')
  })

  it('is deployed for a finished launch and failed for a failed or cancelled one', () => {
    expect(deployView([job()]).kind).toBe('deployed')
    expect(deployView([job({ status: 'failed' })]).kind).toBe('failed')
    expect(deployView([job({ status: 'cancelled' })]).kind).toBe('failed')
  })

  it('describes the NEWEST launch even when an older one finished', () => {
    const done = job({ id: 'done', created_at: 100 })
    const failed = job({ id: 'failed', status: 'failed', created_at: 200 })
    const view = deployView([done, failed])
    expect(view.kind).toBe('failed')
    expect(view.kind === 'failed' && view.job.id).toBe('failed')
  })
})

describe('deployProgress', () => {
  it('counts finished steps plus the one under way, over the recorded total', () => {
    expect(deployProgress({ steps: [step('done'), step('active'), step('pending')] }))
      .toEqual({ current: 2, total: 3 })
  })

  it('falls back to the four steps every launch has when none were recorded', () => {
    expect(deployProgress({ steps: [] })).toEqual({ current: 1, total: 4 })
  })

  it('never reports a step past the total', () => {
    expect(deployProgress({ steps: [step('done'), step('done')] })).toEqual({ current: 2, total: 2 })
  })
})

describe('deployAge', () => {
  const NOW = 1_700_000_000_000

  it('renders the unknown glyph for a missing or future timestamp, never a garbage age', () => {
    expect(deployAge(0, NOW)).toBe('\u2013')
    expect(deployAge(Number.NaN, NOW)).toBe('\u2013')
    expect(deployAge(NOW / 1000 + 3600, NOW)).toBe('\u2013')
  })

  it('shows days and hours past a day, hours and minutes past an hour, minutes below', () => {
    const h = 3600
    expect(deployAge(NOW / 1000 - (2 * 24 * h + 3 * h), NOW)).toMatch(/2.*3/)
    expect(deployAge(NOW / 1000 - (5 * h + 10 * 60), NOW)).toMatch(/5.*10/)
    expect(deployAge(NOW / 1000 - 7 * 60, NOW)).toMatch(/7/)
  })

  it('drops a zero part rather than printing "2d 0h"', () => {
    expect(deployAge(NOW / 1000 - 2 * 24 * 3600, NOW)).not.toMatch(/0/)
  })
})

describe('the panel', () => {
  beforeEach(() => {
    cloudLaunches.mockReset()
    navigateSpy.mockReset()
  })
  afterEach(() => { vi.useRealTimers() })

  it('reports a read failure as unknown, and never as "runs only here"', async () => {
    cloudLaunches.mockRejectedValue(new Error('boom'))
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)

    await waitFor(() => expect(screen.getByTestId('deploy-error')).toBeTruthy())
    // The load-bearing half: the not-deployed sentence must be ABSENT, because a
    // reader who sees it concludes their crew is not deployed.
    expect(screen.queryByTestId('deploy-state-none')).toBeNull()
    // Through the shared error surface, so the agent hand-off is on it.
    expect(within(screen.getByTestId('deploy-error')).getByRole('alert')).toBeTruthy()
    expect(screen.getByTestId('deploy-retry')).toBeTruthy()
  })

  it('offers the deploy action when nothing is launched, and leads into Settings', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [] })
    const onClose = vi.fn()
    renderWithProviders(<DeployMyCrewDialog open onClose={onClose} members={MEMBERS} />)

    await waitFor(() => expect(screen.getByTestId('deploy-state-none')).toBeTruthy())
    expect(screen.queryByTestId('deploy-error')).toBeNull()
    expect(screen.queryByTestId('deploy-details')).toBeNull()
    // The button only opens Settings, and its label says so; the line under
    // it says the one thing the label cannot: nothing is created until the
    // steps there are confirmed.
    expect(screen.getByTestId('deploy-action-deploy').textContent).toContain('Settings')
    expect(screen.getByTestId('deploy-action-hint').textContent).toContain('Nothing is created')

    fireEvent.click(screen.getByTestId('deploy-action-deploy'))
    expect(navigateSpy).toHaveBeenCalledWith('/settings/instances')
    // Closed before the navigation, so the dialog is not what a reader comes
    // back to.
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows the step count while deploying and offers nothing to press', async () => {
    cloudLaunches.mockResolvedValue({
      jobs: [job({ status: 'running', steps: [step('done'), step('active'), step('pending'), step('pending')] })],
    })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)

    await waitFor(() => expect(screen.getByTestId('deploy-state-deploying')).toBeTruthy())
    expect(screen.getByTestId('deploy-progress').textContent).toContain('2')
    expect(screen.getByTestId('deploy-progress').textContent).toContain('4')
    expect(screen.queryByTestId('deploy-action-deploy')).toBeNull()
    expect(screen.queryByTestId('deploy-action-signin')).toBeNull()
    expect(screen.queryByTestId('deploy-address')).toBeNull()
    // The deploy runs in the gateway, so Close does not cancel it, and the
    // dialog says so rather than leaving the reader to babysit the window.
    expect(screen.getByTestId('deploy-close-hint').textContent).toContain('close')
  })

  it('re-reads the launches while one is moving, and not once it has settled', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    cloudLaunches.mockResolvedValue({ jobs: [job({ status: 'running' })] })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)
    await waitFor(() => expect(screen.getByTestId('deploy-state-deploying')).toBeTruthy())
    expect(cloudLaunches).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(4_100)
    expect(cloudLaunches).toHaveBeenCalledTimes(2)

    // The second read says it finished: no third read follows.
    cloudLaunches.mockResolvedValue({ jobs: [job()] })
    await vi.advanceTimersByTimeAsync(4_100)
    await waitFor(() => expect(screen.getByTestId('deploy-state-deployed')).toBeTruthy())
    const settled = cloudLaunches.mock.calls.length
    await vi.advanceTimersByTimeAsync(8_500)
    expect(cloudLaunches).toHaveBeenCalledTimes(settled)
  })

  it('sends the reader to finish the sign-in when the launch is waiting on them', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [job({ status: 'awaiting_signin' })] })
    const onClose = vi.fn()
    renderWithProviders(<DeployMyCrewDialog open onClose={onClose} members={MEMBERS} />)

    await waitFor(() => expect(screen.getByTestId('deploy-state-signin')).toBeTruthy())
    // Names the navigation and the destination, like the deploy button.
    expect(screen.getByTestId('deploy-action-signin').textContent).toContain('Settings')
    fireEvent.click(screen.getByTestId('deploy-action-signin'))
    expect(navigateSpy).toHaveBeenCalledWith('/settings/instances')
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows the two honest numbers and the copyable address once deployed', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    // jsdom exposes `navigator.clipboard` as a getter, so it is redefined rather
    // than assigned, and put back afterwards.
    const realClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard')
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    try {
      cloudLaunches.mockResolvedValue({ jobs: [job()] })
      renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)

      await waitFor(() => expect(screen.getByTestId('deploy-state-deployed')).toBeTruthy())
      expect(screen.getByTestId('deploy-stat-region').textContent).toBe('us-west-2')
      // The fixture's launch is years old, so the age is a real number, not the
      // unknown glyph.
      expect(screen.getByTestId('deploy-stat-since').textContent).not.toBe('\u2013')
      expect(screen.getByTestId('deploy-address-value').textContent).toBe('i-0abc123def4567890')
      // The row and the no-target sentence are exclusive.
      expect(screen.queryByTestId('deploy-no-target')).toBeNull()

      fireEvent.click(screen.getByTestId('deploy-address-copy'))
      expect(writeText).toHaveBeenCalledWith('i-0abc123def4567890')
      await waitFor(() => expect(screen.getByTestId('deploy-address-copy').textContent).toContain('Copied'))
      // A copy that landed shows no failure notice.
      expect(screen.queryByTestId('deploy-address-copy-error')).toBeNull()

      // The technical line survives, collapsed, with the raw identifiers.
      const details = screen.getByTestId('deploy-details')
      expect((details as HTMLDetailsElement).open).toBe(false)
      expect(within(details).getByTestId('deploy-launch').textContent).toContain('aws_ec2')
      expect(within(details).getByTestId('deploy-launch').textContent).toContain('crew-alpha')
    } finally {
      if (realClipboard) Object.defineProperty(navigator, 'clipboard', realClipboard)
    }
  })

  it('says so when the copy fails, and never paints Copied over an unchanged clipboard', async () => {
    // Both layers refuse: the async API rejects (a denied permission) and there
    // is no execCommand fallback, which is jsdom's shape and a plain-HTTP remote
    // dashboard's everyday one.
    const writeText = vi.fn().mockRejectedValue(new Error('denied'))
    const realClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard')
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    try {
      cloudLaunches.mockResolvedValue({ jobs: [job()] })
      renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)
      await waitFor(() => expect(screen.getByTestId('deploy-state-deployed')).toBeTruthy())

      fireEvent.click(screen.getByTestId('deploy-address-copy'))
      const notice = await screen.findByTestId('deploy-address-copy-error')
      expect(notice.textContent).toContain('Copy failed. Select the text and copy it manually.')
      expect(screen.getByTestId('deploy-address-copy').textContent).not.toContain('Copied')
      // The message is the whole remedy: no agent hand-off beside it.
      expect(within(notice).queryByRole('button')).toBeNull()
      // And the value is still there to select by hand.
      expect(screen.getByTestId('deploy-address-value').textContent).toBe('i-0abc123def4567890')
    } finally {
      if (realClipboard) Object.defineProperty(navigator, 'clipboard', realClipboard)
    }
  })

  it('draws the region as the unknown glyph when the record has none', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [job({ region: '' })] })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)
    await waitFor(() => expect(screen.getByTestId('deploy-state-deployed')).toBeTruthy())
    expect(screen.getByTestId('deploy-stat-region').textContent).toBe('\u2013')
  })

  it('offers no address for a finished launch that recorded no machine', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [job({ instance_id: undefined })] })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)
    await waitFor(() => expect(screen.getByTestId('deploy-state-deployed')).toBeTruthy())
    expect(screen.queryByTestId('deploy-address')).toBeNull()
    // No sentence either: "runs as a container task" would be false of an EC2
    // launch that simply recorded no machine.
    expect(screen.queryByTestId('deploy-no-target')).toBeNull()
  })

  it('offers no address for a finished Fargate launch, whose id is a task ARN', async () => {
    const arn = 'arn:aws:ecs:us-west-2:123456789012:task/crew/0f1e2d3c'
    cloudLaunches.mockResolvedValue({ jobs: [job({ provider_id: 'aws_fargate', instance_id: arn })] })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)
    await screen.findByTestId('deploy-state-deployed')
    expect(screen.getByTestId('deploy-stats')).toBeInTheDocument()
    expect(screen.queryByTestId('deploy-address')).toBeNull()
    // Said in words where the row would be, so the reader who saw the row on
    // an EC2 launch is not left guessing why this one has none.
    expect(screen.getByTestId('deploy-no-target').textContent).toContain('Details')
    // The ARN is still there for the owner who needs it, in the Details line.
    expect(within(screen.getByTestId('deploy-details')).getByTestId('deploy-launch').textContent).toContain(arn)
  })

  it('shows the recorded error through the error surface after a failed launch, and the deploy action', async () => {
    cloudLaunches.mockResolvedValue({
      jobs: [job({ status: 'failed', error: 'stack rolled back: the task role could not be assumed' })],
    })
    const onClose = vi.fn()
    renderWithProviders(<DeployMyCrewDialog open onClose={onClose} members={MEMBERS} />)

    await waitFor(() => expect(screen.getByTestId('deploy-state-failed')).toBeTruthy())
    const notice = screen.getByTestId('deploy-launch-error')
    expect(notice.getAttribute('role')).toBe('alert')
    expect(notice.textContent).toContain('the task role could not be assumed')
    // Never offered as reachable: the failed fixture still carries a machine.
    expect(screen.queryByTestId('deploy-address')).toBeNull()
    // The same label and where-it-leads line as the not-deployed state, for
    // the same button.
    expect(screen.getByTestId('deploy-action-deploy').textContent).toContain('Settings')
    expect(screen.getByTestId('deploy-action-hint').textContent).toContain('Nothing is created')

    fireEvent.click(screen.getByTestId('deploy-action-deploy'))
    expect(navigateSpy).toHaveBeenCalledWith('/settings/instances')
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('renders no error surface for a cancelled launch that recorded no error', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [job({ status: 'cancelled', error: undefined })] })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={MEMBERS} />)
    await waitFor(() => expect(screen.getByTestId('deploy-state-failed')).toBeTruthy())
    expect(screen.queryByTestId('deploy-launch-error')).toBeNull()
  })

  it('leads with the crew\'s faces, and folds a large roster into a count', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [] })
    const many = Array.from({ length: 9 }, (_, i) => ({ name: `Member ${i}` }))
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={many} />)

    const faces = screen.getByTestId('deploy-faces')
    expect(faces.querySelectorAll('img').length).toBe(6)
    expect(screen.getByTestId('deploy-faces-more').textContent).toBe('+3')
  })

  it('draws no face row for an empty roster', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [] })
    renderWithProviders(<DeployMyCrewDialog open onClose={() => {}} members={[]} />)
    await waitFor(() => expect(screen.getByTestId('deploy-state-none')).toBeTruthy())
    expect(screen.queryByTestId('deploy-faces')).toBeNull()
    expect(screen.queryByTestId('deploy-faces-more')).toBeNull()
  })

  it('makes no launch read at all while it is closed', async () => {
    cloudLaunches.mockResolvedValue({ jobs: [] })
    renderWithProviders(<DeployMyCrewDialog open={false} onClose={() => {}} members={MEMBERS} />)

    // Opening the page must not cost a request against the operator's account.
    await new Promise((r) => setTimeout(r, 20))
    expect(cloudLaunches).not.toHaveBeenCalled()
  })
})
