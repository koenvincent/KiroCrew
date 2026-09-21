/**
 * Screenshot harness for the two surfaces `compaction.keep` adds.
 *
 * Runs the REAL built SPA (website/dist) with every /api/** call answered from
 * fixtures. No gateway, no agent and no Jev call: only the network is stubbed, so
 * `readCompactionKeepRecord`, `CompactionCard`'s three shapes, the transcript
 * virtualizer and the settings card render exactly as in production.
 *
 * Two surfaces, because the feature is a measurement and the consent that allows it:
 *
 *  - the LINE. One line under the compaction notice, saying what an oracle WOULD
 *    have kept. Three frames, because the record outlives whichever arm the
 *    compaction took: the success notice, the recycle notice, and the ⚠-led
 *    failure that renders through the shared error surface. A line placed inside
 *    one branch would be absent from the other two, and only a picture of all
 *    three shows it is not.
 *  - the SWITCH. The third consent row on the Decisions (Jev) card, plus the
 *    point row it unlocks. The `withheld` pass serves the SAME fixtures with the
 *    scope revoked and ASSERTS both are absent, because "the row appears" is only
 *    half the claim.
 *
 * WHAT THIS HARNESS ASSERTS, and why it asserts anything at all: a capture script
 * that only writes PNGs fails toward a false pass -- a fixture typo, a clipped
 * card or a state that never arrived all still produce a tidy image a PR can cite.
 * So every frame also checks the words that make it that state, and the two
 * absence claims are checked rather than described.
 *
 * Usage: node scripts/capture-decision-compaction-keep.mjs [outDir]
 */
import { mkdirSync } from 'node:fs'

import { json } from './lib/boot-api.mjs'
import { openTranscriptHarness } from './lib/transcript-harness.mjs'

const OUT = process.argv[2] || '../temp-screenshots/decision-compaction-keep'
const SLOT = 'chat-compaction-keep'
const PROJECT = '/home/user/workspace/KiroCrew'

mkdirSync(OUT, { recursive: true })

const LINE = '[data-testid="compaction-keep-line"]'

/** Consent is on, pointed at the address it was given for, and scoped. */
const CONSENT = {
  enabled: true,
  endpoint: 'https://api.typesafe.ai/v1/systemone',
  configured_endpoint: 'https://api.typesafe.ai/v1/systemone',
  permits: true,
  tool_args: true,
  compaction: true,
}

/**
 * The record the gateway stamps, as § 11 of the decisions spec pins it.
 *
 * 14 kept whole + 9 kept without their result + 38 dropped = 61, because the
 * reader refuses a record whose tallies do not sum to the total.
 */
const RECORD = {
  turn_id: 'cmp-7ab419',
  point: 'compaction.keep',
  total_calls: 61,
  pinned_calls: 7,
  kept_both: 14,
  kept_call: 9,
  dropped: 38,
  chars_all: 1482300,
  chars_today: 74110,
  chars_jev: 607743,
  requests: 3,
  fitting_stage: 'inputs_200',
  latency_ms: 2140,
  error: null,
}

const RECYCLE_RECORD = { ...RECORD, turn_id: 'cmp-91d004', kept_both: 3, kept_call: 4, dropped: 54, chars_jev: 190400 }
const FAILED_RECORD = { ...RECORD, turn_id: 'cmp-2f8c55', kept_both: 20, kept_call: 11, dropped: 30, chars_jev: 902110 }

const t0 = Date.now() / 1000 - 1800

const slots = [
  {
    key: SLOT,
    title: 'Rewrite the parser',
    running: false,
    last_message: 'Switched to config/read.py.',
    messages: 7,
    agent: 'kirocrew',
    memory_mode: 'persistent',
    project: PROJECT,
    modified: Math.floor(Date.now() / 1000),
    source_links: [],
    source_links_total: 0,
  },
]

const detail = {
  running: false,
  has_more: false,
  total: 7,
  queue: [],
  project: PROJECT,
  messages: [
    { role: 'user', ts: t0, content: 'Rewrite the config parser to accept TOML.' },
    { role: 'assistant', ts: t0 + 12, content: 'Editing parser.py now — the TOML reader is in.' },
    // The threshold success notice: the conversation was summarized and kept going.
    {
      role: 'assistant',
      ts: t0 + 300,
      content: '\u{1F504} Auto-compacted at 85%.',
      meta: { kind: 'compaction', decisions_strip: RECORD },
    },
    { role: 'assistant', ts: t0 + 320, content: 'Carrying on with the schema check.' },
    // The RECYCLE notice: compaction did not succeed, so the session was replaced.
    {
      role: 'assistant',
      ts: t0 + 700,
      content:
        '\u267B\uFE0F Compaction didn\u2019t succeed at 91%, so the session was restarted '
        + 'instead. The conversation above is still here; the agent no longer remembers it.',
      meta: { kind: 'compaction', decisions_strip: RECYCLE_RECORD },
    },
    // The FAILURE shape, which renders through the shared error surface.
    {
      role: 'assistant',
      ts: t0 + 1100,
      content: '\u26A0 Auto-compact failed at 88% — will retry after cooldown. You can run `/compact` manually.',
      meta: { kind: 'compaction', decisions_strip: FAILED_RECORD },
    },
    { role: 'assistant', ts: t0 + 1200, content: 'Switched to config/read.py.' },
  ],
}

const failures = []
const check = (ok, msg) => {
  if (!ok) failures.push(msg)
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`)
}

const shot = []

async function main() {
  const { page, load, close } = await openTranscriptHarness({
    slot: SLOT,
    project: PROJECT,
    slots,
    detail,
  })

  // The fleet answer and the keystone. Registered AFTER the harness's catch-all so
  // they win: Playwright matches route handlers in reverse order.
  let consent = CONSENT
  await page.route('**/api/decisions/consent', route => json(route, consent))
  await page.route('**/api/dashboard/config', route =>
    json(route, {
      restore_sessions: false,
      restore_window_minutes: 30,
      merge_queued_messages: false,
      widget_density: 'more',
      decisions_enabled: true,
    }),
  )

  /**
   * Pin the locale to English. The harness's init script CLEARS localStorage on
   * every navigation, so the key is written by an init script registered AFTER the
   * harness's first navigation, and a reload is what makes it stick.
   */
  let localePinned = false
  async function loadInEnglish(theme) {
    await load(theme, { selector: LINE })
    if (!localePinned) {
      await page.addInitScript(() => localStorage.setItem('mc-lang', 'en'))
      localePinned = true
    }
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.waitForSelector(LINE, { timeout: 20000 })
    await page.waitForTimeout(800)
  }

  /** The card the line hangs under, not the line alone: the claim is that the
   *  receipt sits on the notice it describes, which a clip of the line cannot show. */
  async function shotCard(index, name, theme) {
    const line = page.locator(LINE).nth(index)
    await line.waitFor({ timeout: 20000 })
    await line.evaluate(el => el.scrollIntoView({ block: 'center' }))
    await page.waitForTimeout(300)
    const host = line.locator('xpath=ancestor::div[@data-testid="compaction-card-host"][1]')
    const file = `${OUT}/${name}-${theme}.png`
    await host.screenshot({ path: file })
    shot.push(`${name}-${theme}.png`)
    console.log('wrote', file)
    return (await line.textContent()) ?? ''
  }

  for (const theme of ['light', 'dark']) {
    await loadInEnglish(theme)

    check(
      await page.locator(LINE).count() === 3,
      `${theme}: the line draws on all three compaction shapes, not only the success one`,
    )

    const success = await shotCard(0, '01-compacted', theme)
    check(
      /would keep/.test(success),
      `${theme}: the line says WOULD, so it cannot be read as what the compaction did (${success.trim().slice(0, 90)})`,
    )
    check(
      /23/.test(success) && /61/.test(success),
      `${theme}: the kept count is derived from the log's own tallies (${success.trim().slice(0, 90)})`,
    )
    check(/%/.test(success), `${theme}: the character share is on the line`)

    const recycled = await shotCard(1, '02-recycled', theme)
    check(/would keep/.test(recycled), `${theme}: the recycle notice carries the line too`)

    const failed = await shotCard(2, '03-failed', theme)
    check(
      /would keep/.test(failed),
      `${theme}: the failure shape carries it too -- the scoring runs before the arm is picked`,
    )
    check(
      await page.getByTestId('compaction-card-error').count() > 0,
      `${theme}: the failure still renders through the shared error surface`,
    )
  }

  // ── The consent card, and the absence claim ──
  async function openSettings(theme) {
    await page.goto(`${new URL(page.url()).origin}/settings/developer`, {
      waitUntil: 'domcontentloaded',
    })
    await page.emulateMedia({ colorScheme: theme })
    await page.waitForTimeout(1200)
  }

  const scopeSwitch = () =>
    page.getByRole('switch', {
      name: 'Also send the conversation and tool-call inputs so Jev can score compaction',
    })

  for (const theme of ['light', 'dark']) {
    await openSettings(theme)
    const row = scopeSwitch()
    await row.waitFor({ timeout: 20000 })
    await row.evaluate(el => el.scrollIntoView({ block: 'center' }))
    await page.waitForTimeout(400)
    check(
      await row.getAttribute('aria-checked') === 'true',
      `${theme}: the third switch draws the recorded scope`,
    )
    check(
      await page.getByTitle('compaction.keep').count() > 0,
      `${theme}: the point is named while its scope is granted`,
    )
    const card = row.locator('xpath=ancestor::*[contains(@class,"rounded")][1]')
    const file = `${OUT}/04-consent-${theme}.png`
    await card.screenshot({ path: file })
    shot.push(`04-consent-${theme}.png`)
    console.log('wrote', file)
  }

  // The SAME fixtures with the scope revoked: the switch stays, unchecked, and the
  // point row is gone. Asserted rather than described, because "it appears when
  // granted" says nothing about what an ungranted install sees.
  consent = { ...CONSENT, compaction: false }
  await openSettings('light')
  await scopeSwitch().waitFor({ timeout: 20000 })
  check(
    await scopeSwitch().getAttribute('aria-checked') === 'false',
    'withheld: the scope draws OFF for a keystone that never recorded it',
  )
  check(
    await page.getByTitle('compaction.keep').count() === 0,
    'withheld: the point is NOT named while its scope is ungranted',
  )

  await close()

  if (failures.length) {
    console.error(`\n${failures.length} check(s) failed:`)
    for (const line of failures) console.error(`  - ${line}`)
    process.exit(1)
  }
  console.log(`\nwrote ${shot.length} shot(s) to ${OUT}: ${shot.join(', ')}`)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
