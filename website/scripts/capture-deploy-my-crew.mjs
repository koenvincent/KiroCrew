/**
 * Screenshot harness for the Crew Members page's "Your crew in the cloud"
 * panel (DeployMyCrew.tsx) and its labelled header trigger.
 *
 * The panel answers one question -- "is my crew deployed, and what do I do
 * next?" -- with the crew's faces, one sentence, and at most one button. A
 * render test can pin that DOM; it cannot show whether those parts compose
 * readably, which is the exact thing the redesign exists to fix. These frames
 * are that evidence, one per state the panel can be in:
 *
 *   01-trigger          the Members page header with the labelled Cloud control
 *   02-none             no launch on record: one sentence, one button
 *   03-deploying        a moving launch: Step N of M and the rough ETA, no button
 *   04-signin           a launch holding for the reader's sign-in: one button
 *   05-deployed         since-deploy, region, the Session Manager target with
 *                       its Copy button, Details collapsed
 *   06-deployed-details the same launch with Details expanded (raw identifiers)
 *   07-deployed-fargate a finished Fargate launch: deployed, but no target row,
 *                       because a task ARN is not a Session Manager target
 *   08-failed           the recorded error through ErrorNotice, one button
 *   09-loading          the list is still being read
 *   10-read-failed      the list could not be read: error and Try again
 *   11-copy-failed      both clipboard layers refused: the row says so, no tick
 *   12-copied           the copy landed: the button reads Copied, no notice
 *
 * Runs the REAL built SPA (website/dist) behind the shared loopback static
 * server with every /api/** call answered from fixtures -- no gateway, no AWS.
 * Every claim is ASSERTED before its PNG is written: a harness that only writes
 * PNGs can hand a PR a picture of an error boundary as evidence.
 *
 * Usage: npm run build && node scripts/capture-deploy-my-crew.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'
import { MEMBERS } from './lib/members-fixtures.mjs'

const OUT = process.argv[2] || '../temp-screenshots/deploy-my-crew'
mkdirSync(OUT, { recursive: true })

const NOW_SEC = Math.floor(Date.now() / 1000)

const STEPS_DONE = [
  { key: 'preflight', label: 'Check your AWS setup', state: 'done', detail: '' },
  { key: 'provision', label: 'Create the instance and install Kiro Crew', state: 'done', detail: 'i-0abc123456789def0' },
  { key: 'signin', label: 'Sign in to Kiro', state: 'done', detail: '' },
  { key: 'connect', label: 'Connect', state: 'done', detail: 'Added to your instances.' },
]

/** A launch that finished on the built-in EC2 lane three hours ago. */
const DEPLOYED_JOB = {
  id: 'j-deployed',
  provider_id: 'aws_ec2',
  tag: 'kc-5e10bb',
  instance_id: 'i-0abc123456789def0',
  profile: 'default',
  region: 'us-east-1',
  size_key: 'balanced',
  status: 'done',
  steps: STEPS_DONE,
  signin: null,
  signin_detected: true,
  error: '',
  created_at: NOW_SEC - 3 * 3600 - 12 * 60,
  updated_at: NOW_SEC - 3 * 3600,
}

/** The same crew, still on step 2 of 4. */
const DEPLOYING_JOB = {
  ...DEPLOYED_JOB,
  id: 'j-deploying',
  instance_id: '',
  status: 'running',
  steps: [
    { ...STEPS_DONE[0] },
    { ...STEPS_DONE[1], state: 'active', detail: '' },
    { ...STEPS_DONE[2], state: 'pending' },
    { ...STEPS_DONE[3], state: 'pending' },
  ],
  created_at: NOW_SEC - 90,
  updated_at: NOW_SEC - 5,
}

/** Holding for the reader to approve a device code in Settings. */
const SIGNIN_JOB = {
  ...DEPLOYED_JOB,
  id: 'j-signin',
  status: 'awaiting_signin',
  steps: STEPS_DONE.map((s) => (s.key === 'signin' ? { ...s, state: 'active' } : s.key === 'connect' ? { ...s, state: 'pending', detail: '' } : s)),
  signin: { url: 'https://example.invalid/device', code: 'GFXK-MNKS', ports: [] },
  signin_detected: false,
  created_at: NOW_SEC - 6 * 60,
  updated_at: NOW_SEC - 20,
}

/** A finished Fargate launch: `instance_id` carries the task ARN. */
const FARGATE_JOB = {
  ...DEPLOYED_JOB,
  id: 'j-fargate',
  provider_id: 'aws_fargate',
  tag: 'kc-7a21cd',
  instance_id: 'arn:aws:ecs:us-east-1:123456787890:task/kc-7a21cd/8f3e2a1b9c0d4e5f6a7b8c9d0e1f2a3b',
  created_at: NOW_SEC - 2 * 86400 - 5 * 3600,
  updated_at: NOW_SEC - 2 * 86400,
}

/** A launch that died in provisioning, with the gateway's own sentence. */
const FAILED_JOB = {
  ...DEPLOYED_JOB,
  id: 'j-failed',
  instance_id: '',
  status: 'failed',
  steps: [
    { ...STEPS_DONE[0] },
    { ...STEPS_DONE[1], state: 'error', detail: '' },
    { ...STEPS_DONE[2], state: 'pending', detail: '' },
    { ...STEPS_DONE[3], state: 'pending', detail: '' },
  ],
  error: 'CloudFormation stack kc-5e10bb rolled back: VcpuLimitExceeded in us-east-1.',
  created_at: NOW_SEC - 40 * 60,
  updated_at: NOW_SEC - 32 * 60,
}

/**
 * The fixture world, switched per frame. `jobs` decides the panel's state;
 * `launchAnswer` overrides the whole list response (a 500, or a hold that
 * never answers, for the loading and read-failed frames).
 */
const world = { jobs: [], launchAnswer: null }

const extra = async (path, route) => {
  if (path === '/api/members') return json(route, { members: MEMBERS, default_agent: 'kirocrew' }), true
  if (path === '/api/crons') return json(route, { jobs: [] }), true
  if (path === '/api/webhooks') return json(route, { tokens: [] }), true
  if (path === '/api/agents') return json(route, { agents: [], default_agent: 'kirocrew' }), true
  if (path === '/api/cloud/launch') {
    if (world.launchAnswer === 'hold') return true // never fulfilled: the read stays in flight
    if (world.launchAnswer) return json(route, world.launchAnswer.body, world.launchAnswer.status), true
    return json(route, { jobs: world.jobs }), true
  }
  const thread = /^\/api\/members\/([^/]+)\/thread$/.exec(path)
  if (thread) {
    const slug = decodeURIComponent(thread[1])
    return json(route, { slot_key: `member-${slug}`, slug, member: slug, created: false }), true
  }
  if (/^\/api\/members\/[^/]+\/activity$/.test(path)) {
    return json(route, { slug: 'radar', member: 'radar', capped: false, entries: [] }), true
  }
  if (/^\/api\/chat\/slots\/[^/]+$/.test(path)) {
    return json(route, { key: 'member-radar', title: 'radar', running: false, messages: [] }), true
  }
  return false
}

let failures = 0
const fail = (msg) => { console.error(`FAIL: ${msg}`); failures++ }

const { srv, base } = await serveDist()
const browser = await chromium.launch()
const VIEW = { width: 1220, height: 900 }
const context = await browser.newContext({
  viewport: VIEW,
  deviceScaleFactor: 2, // the 11px hint copy renders soft at 1x on GitHub
})

/** A Members page on the current fixture world, in the given theme. */
const openMembers = async (theme) => {
  const page = await context.newPage()
  await stubDashboardApi(page, { extra, theme })
  logPageProblems(page)
  await page.goto(`${base}/members`, { waitUntil: 'domcontentloaded' })
  await page.getByTestId('member-deploy-open').waitFor({ timeout: 20000 })
  await page.waitForTimeout(400)
  return page
}

/** Open the panel and wait for the state it should land in. */
const openPanel = async (page, stateTestId) => {
  await page.getByTestId('member-deploy-open').click()
  await page.getByTestId(stateTestId).waitFor({ timeout: 20000 })
  await page.waitForTimeout(350)
  return page.getByRole('dialog')
}

/** A clip around one element, clamped to the viewport. */
const clipOf = async (locator, pad = 14) => {
  const box = await locator.boundingBox()
  const x = Math.max(0, Math.min(box.x - pad, VIEW.width - 1))
  const y = Math.max(0, Math.min(box.y - pad, VIEW.height - 1))
  return {
    x,
    y,
    width: Math.max(1, Math.min(VIEW.width - x, box.width + 2 * pad)),
    height: Math.max(1, Math.min(VIEW.height - y, box.height + 2 * pad)),
  }
}

/** The one-button invariant: at most one action besides Close and Copy. */
const countActions = async (dialog) => {
  const names = await dialog.locator('button').evaluateAll((els) =>
    els.map((el) => (el.textContent || '').trim()).filter(Boolean),
  )
  return names.filter((n) => !/^Close$/.test(n) && !/^(Copy|Copied)$/.test(n))
}

/** The faces row must lead: it sits above the state sentence. */
const facesLead = async (dialog, stateTestId) => {
  const faces = await dialog.getByTestId('deploy-faces').boundingBox()
  const state = await dialog.getByTestId(stateTestId).boundingBox()
  return !!faces && !!state && faces.y + faces.height <= state.y + 1
}

// ---- Frame 1: the labelled trigger in the Members page header --------------

for (const theme of ['dark', 'light']) {
  const page = await openMembers(theme)
  const trigger = page.getByTestId('member-deploy-open')
  const label = (await trigger.innerText()).trim()
  if (label !== 'Cloud') fail(`${theme}: the header trigger must read "Cloud", got ${JSON.stringify(label)}`)
  if (!/Your crew in the cloud/.test((await trigger.getAttribute('title')) || '')) {
    fail(`${theme}: the trigger's title must name the panel it opens`)
  }
  // Bordered, so it reads as a button and not as a status chip beside the
  // page title.
  const border = await trigger.evaluate((el) => {
    const cs = getComputedStyle(el)
    return { style: cs.borderTopStyle, width: parseFloat(cs.borderTopWidth) }
  })
  if (border.style === 'none' || !(border.width > 0)) fail(`${theme}: the trigger must carry a visible border: ${JSON.stringify(border)}`)
  // The full-width header strip the trigger lives in, so the reader sees the
  // control among its neighbours rather than a cropped button.
  const box = await trigger.boundingBox()
  const y = Math.max(0, box.y - 14)
  await page.screenshot({
    path: `${OUT}/01-trigger-${theme}.png`,
    clip: { x: 0, y, width: VIEW.width, height: Math.min(VIEW.height - y, box.height + 60) },
  })
  console.log(`wrote 01 (${theme})`)
  await page.close()
}

// ---- Frame 2: no launch on record ------------------------------------------

world.jobs = []
for (const theme of ['dark', 'light']) {
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-state-none')
  const text = await dialog.innerText()
  if (!/Right now your crew runs only on this computer\./.test(text)) fail(`${theme}: the no-launch sentence is missing`)
  if (!(await facesLead(dialog, 'deploy-state-none'))) fail(`${theme}: the faces must sit above the sentence`)
  if ((await dialog.getByTestId('deploy-faces').locator('img, [data-testid="deploy-faces-more"], svg, span').count()) < 1) {
    fail(`${theme}: the faces row rendered nothing`)
  }
  const actions = await countActions(dialog)
  if (actions.length !== 1 || actions[0] !== 'Open the cloud set-up in Settings') {
    fail(`${theme}: exactly one action, "Open the cloud set-up in Settings", got ${JSON.stringify(actions)}`)
  }
  if (await dialog.getByTestId('deploy-details').count()) fail(`${theme}: no Details line without a launch`)
  // The button only opens Settings, and the line under it says so: a reader
  // who takes the label as the act itself does not press it.
  const hint = dialog.getByTestId('deploy-action-hint')
  if (!/Nothing is created until you confirm the steps there\./.test(await hint.innerText())) {
    fail(`${theme}: the line under the deploy button must carry the cost/undo fact: ${JSON.stringify(await hint.innerText())}`)
  }
  const [btnBox, hintBox] = [await dialog.getByTestId('deploy-action-deploy').boundingBox(), await hint.boundingBox()]
  if (!(hintBox.y >= btnBox.y + btnBox.height - 1)) fail(`${theme}: the where-it-leads line sits under the button`)
  await page.screenshot({ path: `${OUT}/02-none-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 02 (${theme})`)
  await page.close()
}

// ---- Frame 3: a launch in motion -------------------------------------------

world.jobs = [DEPLOYING_JOB]
{
  const theme = 'dark'
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-state-deploying')
  const text = await dialog.innerText()
  if (!/Deploying your crew/.test(text)) fail(`${theme}: the deploying sentence is missing`)
  if (!/Step 2 of 4/.test(text)) fail(`${theme}: progress must read "Step 2 of 4": ${JSON.stringify(text)}`)
  if (!/Usually ready in about 6 minutes\./.test(text)) fail(`${theme}: the rough ETA is missing`)
  const actions = await countActions(dialog)
  if (actions.length !== 0) fail(`${theme}: a moving launch offers nothing to press, got ${JSON.stringify(actions)}`)
  if (await dialog.getByTestId('deploy-address').count()) fail(`${theme}: no target row while deploying`)
  if (!/The deploy keeps running if you close this window\./.test(text)) fail(`${theme}: the close hint is missing`)
  await page.screenshot({ path: `${OUT}/03-deploying-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 03 (${theme})`)
  await page.close()
}

// ---- Frame 4: holding for the sign-in --------------------------------------

world.jobs = [SIGNIN_JOB]
{
  const theme = 'dark'
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-state-signin')
  const text = await dialog.innerText()
  if (!/Your crew is waiting for you to sign in\./.test(text)) fail(`${theme}: the sign-in sentence is missing`)
  const actions = await countActions(dialog)
  if (actions.length !== 1 || actions[0] !== 'Open the sign-in step in Settings') {
    fail(`${theme}: exactly one action, "Open the sign-in step in Settings", got ${JSON.stringify(actions)}`)
  }
  await page.screenshot({ path: `${OUT}/04-signin-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 04 (${theme})`)
  await page.close()
}

// ---- Frames 5 and 6: deployed, Details collapsed then expanded -------------

world.jobs = [DEPLOYED_JOB]
for (const theme of ['dark', 'light']) {
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-state-deployed')
  const text = await dialog.innerText()
  if (!/Your crew is deployed in the cloud\./.test(text)) fail(`${theme}: the deployed sentence is missing`)
  // NOW_SEC is read once at start, so the minute may have turned by this frame.
  if (!/Since deploy/.test(text) || !/\b3h 1[2-4]m\b/.test(text)) fail(`${theme}: since-deploy must read about 3h 12m: ${JSON.stringify(text)}`)
  if (!/Region/.test(text) || !/us-east-1/.test(text)) fail(`${theme}: the region stat is missing`)
  // The target row: named for what it is, not "Address".
  const row = dialog.getByTestId('deploy-address')
  if (!(await row.isVisible())) fail(`${theme}: the Session Manager target row must be offered on an EC2 launch`)
  const rowText = await row.innerText()
  if (!/Session Manager target/.test(rowText)) fail(`${theme}: the row must be labelled "Session Manager target": ${JSON.stringify(rowText)}`)
  if (/^Address$/m.test(rowText)) fail(`${theme}: the label must not read "Address"`)
  if (!/i-0abc123456789def0/.test(rowText)) fail(`${theme}: the instance id must be shown in the row`)
  if (!/Paste this ID into AWS Session Manager \(in the AWS console\) to open a terminal on the machine\./.test(rowText)) {
    fail(`${theme}: the hint under the target is missing or stale: ${JSON.stringify(rowText)}`)
  }
  if (await dialog.getByTestId('deploy-no-target').count()) fail(`${theme}: the row and the no-target sentence are exclusive`)
  if (!(await dialog.getByTestId('deploy-address-copy').isVisible())) fail(`${theme}: the Copy button is missing`)
  const actions = await countActions(dialog)
  if (actions.length !== 0) fail(`${theme}: a deployed crew offers Copy only, got ${JSON.stringify(actions)}`)
  // Details: present, collapsed, and BELOW the target row.
  const details = dialog.getByTestId('deploy-details')
  if (!(await details.isVisible())) fail(`${theme}: the Details line is missing`)
  if (await details.evaluate((el) => el.open)) fail(`${theme}: Details must start collapsed`)
  const [rowBox, detailsBox] = [await row.boundingBox(), await details.boundingBox()]
  if (!(detailsBox.y >= rowBox.y + rowBox.height - 1)) fail(`${theme}: Details must sit below the target row`)
  await page.screenshot({ path: `${OUT}/05-deployed-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 05 (${theme})`)

  if (theme === 'dark') {
    await details.locator('summary').click()
    await page.waitForTimeout(250)
    if (!(await details.evaluate((el) => el.open))) fail(`${theme}: Details must open on click`)
    const line = await dialog.getByTestId('deploy-launch').first().innerText()
    for (const token of ['kc-5e10bb', 'aws_ec2', 'done', 'default', 'us-east-1', 'i-0abc123456789def0']) {
      if (!line.includes(token)) fail(`${theme}: the Details line must carry ${token}: ${JSON.stringify(line)}`)
    }
    await page.screenshot({ path: `${OUT}/06-deployed-details-${theme}.png`, clip: await clipOf(dialog) })
    console.log(`wrote 06 (${theme})`)
  }
  await page.close()
}

// ---- Frame 7: a finished Fargate launch has no Session Manager target -------

world.jobs = [FARGATE_JOB]
{
  const theme = 'dark'
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-state-deployed')
  const text = await dialog.innerText()
  if (!/Your crew is deployed in the cloud\./.test(text)) fail(`${theme}: a finished Fargate launch is deployed`)
  if (!/2d 5h/.test(text)) fail(`${theme}: since-deploy must read 2d 5h: ${JSON.stringify(text)}`)
  // The whole point of the frame: no target row, because the ARN is not one.
  if (await dialog.getByTestId('deploy-address').count()) fail(`${theme}: a Fargate task ARN must not be offered as a Session Manager target`)
  if (/Session Manager target\n|Paste this ID into AWS Session Manager/.test(text)) fail(`${theme}: no target copy on the Fargate lane`)
  // Said in words where the row would be.
  const noTarget = dialog.getByTestId('deploy-no-target')
  if (!/runs as a container task, not a machine, so there is no Session Manager target\. Its task ID is in Details\./.test(await noTarget.innerText())) {
    fail(`${theme}: the Fargate lane must say why there is no target: ${JSON.stringify(await noTarget.innerText())}`)
  }
  // The ARN survives where an owner debugging the deploy looks for it.
  const details = dialog.getByTestId('deploy-details')
  await details.locator('summary').click()
  await page.waitForTimeout(250)
  const line = await dialog.getByTestId('deploy-launch').first().innerText()
  if (!/aws_fargate/.test(line) || !/arn:aws:ecs:/.test(line)) fail(`${theme}: Details must carry the provisioner id and the task ARN: ${JSON.stringify(line)}`)
  await page.screenshot({ path: `${OUT}/07-deployed-fargate-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 07 (${theme})`)
  await page.close()
}

// ---- Frame 8: the last deploy did not finish -------------------------------

world.jobs = [FAILED_JOB]
{
  const theme = 'dark'
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-state-failed')
  const text = await dialog.innerText()
  if (!/The last deploy did not finish\./.test(text)) fail(`${theme}: the failed sentence is missing`)
  const notice = dialog.getByTestId('deploy-launch-error')
  if (!(await notice.isVisible())) fail(`${theme}: the recorded error must render through ErrorNotice`)
  if (!/VcpuLimitExceeded/.test(await notice.innerText())) fail(`${theme}: the gateway's own error sentence must be shown verbatim`)
  if (await dialog.getByTestId('deploy-address').count()) fail(`${theme}: a failed launch offers no target`)
  const actions = await countActions(dialog)
  // ErrorNotice carries its own agent hand-off; the panel's single action is the deploy button.
  const own = actions.filter((a) => !/ask/i.test(a))
  if (own.length !== 1 || own[0] !== 'Open the cloud set-up in Settings') {
    fail(`${theme}: exactly one panel action, "Open the cloud set-up in Settings", got ${JSON.stringify(actions)}`)
  }
  if (!/Nothing is created until you confirm the steps there\./.test(text)) fail(`${theme}: the cost/undo line must accompany the deploy button here too`)
  await page.screenshot({ path: `${OUT}/08-failed-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 08 (${theme})`)
  await page.close()
}

// ---- Frame 9: the list is still being read ---------------------------------

world.jobs = []
world.launchAnswer = 'hold'
{
  const theme = 'dark'
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-loading')
  const text = await dialog.innerText()
  if (!/Reading your deployments/.test(text)) fail(`${theme}: the loading sentence is missing`)
  if (/runs only on this computer/.test(text)) fail(`${theme}: a read in flight must never claim "runs only on this computer"`)
  if ((await countActions(dialog)).length !== 0) fail(`${theme}: nothing to press while the list is being read`)
  await page.screenshot({ path: `${OUT}/09-loading-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 09 (${theme})`)
  await page.close()
}

// ---- Frame 10: the list could not be read ----------------------------------

world.launchAnswer = { status: 500, body: { error: 'launch store unavailable' } }
{
  const theme = 'dark'
  const page = await openMembers(theme)
  const dialog = await openPanel(page, 'deploy-error')
  const text = await dialog.innerText()
  if (!/Could not read your deployments\./.test(text)) fail(`${theme}: the read-failed sentence is missing`)
  if (/runs only on this computer/.test(text)) fail(`${theme}: a failed read must never claim "runs only on this computer"`)
  const retry = dialog.getByTestId('deploy-retry')
  if (!(await retry.isVisible())) fail(`${theme}: a failed read must offer Try again`)
  if ((await retry.innerText()).trim() !== 'Try again') fail(`${theme}: the retry must read "Try again"`)
  await page.screenshot({ path: `${OUT}/10-read-failed-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 10 (${theme})`)
  await page.close()
}
world.launchAnswer = null

// ---- Frame 11: the copy failed -------------------------------------------
//
// Both layers refuse: the async API rejects (a denied permission) and
// execCommand answers false (no clipboard at all), which is a plain-HTTP LAN
// dashboard's everyday shape. A tick over an unchanged clipboard is the worst
// affordance there is, so the row must say the copy failed, in words.

world.jobs = [DEPLOYED_JOB]
{
  const theme = 'dark'
  const page = await context.newPage()
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: () => Promise.reject(new DOMException('denied', 'NotAllowedError')) },
    })
    document.execCommand = () => false
  })
  await stubDashboardApi(page, { extra, theme })
  logPageProblems(page)
  await page.goto(`${base}/members`, { waitUntil: 'domcontentloaded' })
  await page.getByTestId('member-deploy-open').waitFor({ timeout: 20000 })
  const dialog = await openPanel(page, 'deploy-state-deployed')
  await dialog.getByTestId('deploy-address-copy').click()
  const notice = dialog.getByTestId('deploy-address-copy-error')
  await notice.waitFor({ timeout: 5000 })
  if (!/Copy failed\. Select the text and copy it manually\./.test(await notice.innerText())) {
    fail(`${theme}: a failed copy must say so and name the remedy: ${JSON.stringify(await notice.innerText())}`)
  }
  if (/Copied/.test(await dialog.getByTestId('deploy-address-copy').innerText())) {
    fail(`${theme}: a failed copy must not paint "Copied"`)
  }
  // The message is the whole remedy: an agent cannot supply a clipboard the
  // browser refused, so no hand-off sits beside it.
  if (await notice.getByRole('button').count()) fail(`${theme}: the copy failure must not offer an agent hand-off`)
  if (!/i-0abc123456789def0/.test(await dialog.innerText())) fail(`${theme}: the target must stay readable after a failed copy`)
  await page.screenshot({ path: `${OUT}/11-copy-failed-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 11 (${theme})`)
  await page.close()
}

// ---- Frame 12: the copy landed ---------------------------------------------
//
// The async clipboard resolves, so the button flips to Copied and no failure
// notice appears. The positive twin of frame 11.

world.jobs = [DEPLOYED_JOB]
{
  const theme = 'dark'
  const page = await context.newPage()
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: () => Promise.resolve() },
    })
  })
  await stubDashboardApi(page, { extra, theme })
  logPageProblems(page)
  await page.goto(`${base}/members`, { waitUntil: 'domcontentloaded' })
  await page.getByTestId('member-deploy-open').waitFor({ timeout: 20000 })
  const dialog = await openPanel(page, 'deploy-state-deployed')
  const copy = dialog.getByTestId('deploy-address-copy')
  await copy.click()
  await page.waitForFunction(() => /Copied/.test(document.querySelector('[data-testid="deploy-address-copy"]')?.textContent ?? ''), null, { timeout: 5000 })
  if (await dialog.getByTestId('deploy-address-copy-error').count()) fail(`${theme}: a copy that landed must not show the failure notice`)
  await page.screenshot({ path: `${OUT}/12-copied-${theme}.png`, clip: await clipOf(dialog) })
  console.log(`wrote 12 (${theme})`)
  await page.close()
}

await browser.close()
srv.close()
if (failures) { console.error(`${failures} assertion(s) failed`); process.exit(1) }
console.log('OK')
