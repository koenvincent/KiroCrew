/**
 * Screenshot harness for the line `task.split` adds to an assistant reply.
 *
 * Runs the REAL built SPA (website/dist) with every /api/** call answered from
 * fixtures. No gateway, no agent and no Jev call: only the network is stubbed, so
 * `readSplitRecord`, the transcript virtualizer and the assistant rows render
 * exactly as in production.
 *
 * Three frames, because the line's whole job is to let a reader score advice:
 *
 *  - TAKEN. Jev suggested parallel sub-agents and the agent spawned two, so the
 *    line says both arms name the same shape.
 *  - IGNORED. The same suggestion, answered inline. This is the frame the feature
 *    exists for: a reader who cannot tell it from the first one cannot tell a
 *    useful advisory from one nobody follows.
 *  - BESIDE THE STRIP. One reply carrying a `skills.select` strip AND this line,
 *    because they are two records under two keys and the receipt must not
 *    displace the older one.
 *
 * Usage: node scripts/capture-decision-task-split.mjs [outDir]
 */
import { mkdirSync } from 'node:fs'

import { json } from './lib/boot-api.mjs'
import { openTranscriptHarness } from './lib/transcript-harness.mjs'

const OUT = process.argv[2] || '../temp-screenshots/decision-task-split'
const SLOT = 'chat-task-split'
const PROJECT = '/home/user/workspace/KiroCrew'

mkdirSync(OUT, { recursive: true })

const LINE = '[data-testid="split-decision-line"]'
const TAKEN = `${LINE}[data-agree="true"]`
const IGNORED = `${LINE}[data-agree="false"]`

/** Consent is on and pointed at the address it was given for. */
const CONSENT = {
  enabled: true,
  endpoint: 'https://api.typesafe.ai/v1/systemone',
  configured_endpoint: 'https://api.typesafe.ai/v1/systemone',
  permits: true,
}

/** Jev said split; the agent spawned two helpers. */
const TAKEN_RECORD = {
  turn_id: 'ts-4b19ce',
  ts: '2026-09-21T07:04:11Z',
  point: 'task.split',
  jev_choice: 'split',
  agent_choice: 'split',
  spawn_calls: 2,
  agree: true,
  p: 0.84,
  latency_ms: 190,
}

/** The same suggestion, answered inline: the arms differ. */
const IGNORED_RECORD = {
  ...TAKEN_RECORD,
  turn_id: 'ts-7c02fa',
  agent_choice: 'single',
  spawn_calls: 0,
  agree: false,
  p: 0.77,
  latency_ms: 210,
}

/** A `skills.select` strip, so one reply can be photographed carrying both. */
const SKILLS_RECORD = {
  turn_id: 'sk-2f81bd',
  ts: '2026-09-21T07:06:02Z',
  point: 'skills.select',
  baseline: ['python/testing'],
  jev: ['python/testing'],
  agree: true,
  p: 0.79,
  tokens_saved: 0,
  candidates: 34,
  message_chars: 61,
  history_chars: 0,
  latency_ms: 140,
}

const t0 = Date.now() / 1000 - 900

const slots = [
  {
    key: SLOT,
    title: 'Audit the four handler modules',
    running: false,
    last_message: 'Both audits are in.',
    messages: 6,
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
  total: 6,
  queue: [],
  project: PROJECT,
  messages: [
    { role: 'user', ts: t0, content: 'Audit chat_handlers.py and messaging.py independently.' },
    {
      role: 'assistant',
      ts: t0 + 64,
      content: 'Both audits are in. chat_handlers.py is clean; messaging.py has one unchecked cast at line 812.',
      meta: { decisions_split: TAKEN_RECORD },
    },
    { role: 'user', ts: t0 + 120, content: 'Now rename that cast’s local and re-run its test.' },
    {
      role: 'assistant',
      ts: t0 + 154,
      content: 'Renamed it to `raw_payload` and re-ran test_messaging.py — 41 passed.',
      meta: { decisions_split: IGNORED_RECORD },
    },
    { role: 'user', ts: t0 + 200, content: 'Add a regression test for the cast.' },
    {
      role: 'assistant',
      ts: t0 + 240,
      content: 'Added one in test_messaging.py that fails without the check.',
      meta: { decisions_split: TAKEN_RECORD, decisions_strip: SKILLS_RECORD },
    },
  ],
}

const failures = []
const check = (ok, msg) => {
  if (!ok) failures.push(msg)
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`)
}

async function main() {
  const { page, load, close } = await openTranscriptHarness({
    slot: SLOT,
    project: PROJECT,
    slots,
    detail,
  })

  // The fleet answer and the keystone. Registered AFTER the harness's catch-all so
  // they win: Playwright matches route handlers in reverse order.
  let decisionsEnabled = true
  await page.route('**/api/decisions/consent', route => json(route, CONSENT))
  await page.route('**/api/dashboard/config', route =>
    json(route, {
      restore_sessions: false,
      restore_window_minutes: 30,
      merge_queued_messages: false,
      widget_density: 'more',
      decisions_enabled: decisionsEnabled,
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

  /** The line's own assistant row, which is the claim a clip of the line cannot make. */
  const rowOf = locator => locator.locator('xpath=ancestor::div[@data-role="assistant"][1]')

  for (const theme of ['light', 'dark']) {
    await loadInEnglish(theme)

    // TAKEN: both arms name the same shape.
    const taken = page.locator(TAKEN).first()
    await taken.evaluate(el => el.scrollIntoView({ block: 'center' }))
    await page.waitForTimeout(300)
    await rowOf(taken).screenshot({ path: `${OUT}/01-taken-${theme}.png` })
    console.log('wrote', `${OUT}/01-taken-${theme}.png`)
    const takenText = (await taken.textContent()) ?? ''
    check(
      /Jev suggested/.test(takenText) && /agent/.test(takenText),
      `${theme}: the line names BOTH arms (${takenText.trim().slice(0, 90)})`,
    )
    check(
      /\b2\b/.test(takenText),
      `${theme}: the agent's arm prints its spawn count, because the word is a bucket`,
    )
    check(
      /confidence 0\.84/.test(takenText) && /190/.test(takenText),
      `${theme}: the labelled score and the latency are on the line`,
    )

    // IGNORED: the frame the feature exists for.
    const ignored = page.locator(IGNORED).first()
    await ignored.evaluate(el => el.scrollIntoView({ block: 'center' }))
    await page.waitForTimeout(300)
    await rowOf(ignored).screenshot({ path: `${OUT}/02-ignored-${theme}.png` })
    console.log('wrote', `${OUT}/02-ignored-${theme}.png`)
    const ignoredText = (await ignored.textContent()) ?? ''
    check(
      ignoredText.trim() !== takenText.trim(),
      `${theme}: advice taken and advice ignored are told apart in words`,
    )
    check(
      (await ignored.getAttribute('data-agent-choice')) === 'single'
        && (await ignored.getAttribute('data-jev-choice')) === 'split',
      `${theme}: the two arms are readable apart`,
    )

    // BESIDE THE STRIP: two records, two keys, one reply.
    const strip = page.locator('[data-testid="decision-strip"]').first()
    await strip.evaluate(el => el.scrollIntoView({ block: 'center' }))
    await page.waitForTimeout(300)
    await rowOf(strip).screenshot({ path: `${OUT}/03-beside-strip-${theme}.png` })
    console.log('wrote', `${OUT}/03-beside-strip-${theme}.png`)
    check(
      await strip.isVisible(),
      `${theme}: the skill strip is still drawn on a reply that also carries this line`,
    )
  }

  // ── The withheld pass: the fleet says no, and the receipts survive ──
  decisionsEnabled = false
  await loadInEnglish('dark')
  check(
    (await page.locator(LINE).count()) >= 2,
    'withheld: the receipts for past suggestions are still drawn',
  )
  await rowOf(page.locator(TAKEN).first()).screenshot({ path: `${OUT}/04-withheld.png` })
  console.log('wrote', `${OUT}/04-withheld.png`)

  await close()

  if (failures.length) {
    console.error(`\n${failures.length} assertion(s) failed:`)
    for (const f of failures) console.error(`  - ${f}`)
    process.exit(1)
  }
  console.log('\nall assertions passed')
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
