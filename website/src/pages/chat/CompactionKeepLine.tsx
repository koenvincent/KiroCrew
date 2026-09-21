import { memo } from 'react'
import { Scissors } from 'lucide-react'

import { fmtNumber, fmtPercent } from '../../i18n/format'
import { i18nT } from '../../i18n/t'
import { useLanguageGeneration } from '../../i18n/useLanguageGeneration'
import VerdictThumbs from './DecisionVerdictThumbs'
import type { CompactionKeepRecord } from './decisionRecord'

/**
 * The compaction card's receipt for one SHADOW scoring of that compaction.
 *
 * A compaction discards everything the harness does not summarize, and today's
 * recycle replay keeps the conversation text alone — every tool call and every tool
 * result is already gone. This line says what an oracle would have kept instead, and
 * it is the only place that number is visible: the compaction itself is byte-identical
 * whatever Jev answered, so without the line the measurement would exist only in a
 * log file.
 *
 * ONE line, not a disclosure card. Everything the record holds fits on it — how many
 * of the session's tool calls would have been kept, and what share of the
 * conversation's characters that is — and a collapsed card that hides nothing is a
 * control that does nothing (the same reasoning `SteerDecisionLine` states).
 *
 * It says "would", not "did", and that wording is load-bearing rather than cautious:
 * nothing was kept. A reader who took this for a description of what the compaction
 * did would believe their tool history survived it.
 *
 * The record on the row is the ONLY condition, the rule the other two receipts share:
 * a stamped record is history that already sits on this machine, so drawing it sends
 * nothing, and gating it on the current consent switch would lose the receipt for
 * every past compaction the moment the preview is turned off.
 *
 * The thumbs are `DecisionStrip`'s own pair, shared rather than copied. Side `jev`:
 * the record names one answer by one party, and the arm it is compared against is a
 * replay rule with no opinion to rate.
 */
const CompactionKeepLine = memo(function CompactionKeepLine({
  record,
}: {
  record: CompactionKeepRecord
}) {
  // memo() bails out of the provider-level repaint; subscribe so a language switch
  // repaints this row's strings.
  useLanguageGeneration()

  // Through the format seam, not a template literal: Latin digits are wrong for bn,
  // and a bare `${n}%` would follow the browser's locale rather than the app's.
  const kept = i18nT('pages.chat.compactionKeep.summary', {
    kept: fmtNumber(record.keptCalls),
    total: fmtNumber(record.totalCalls),
  })
  const share = record.charsShare === null
    ? null
    : i18nT('pages.chat.compactionKeep.chars_share', { percent: fmtPercent(record.charsShare) })

  return (
    <div
      className="flex items-center gap-1.5 px-3 pb-2 text-[12px] leading-5 text-muted min-w-0"
      data-testid="compaction-keep-line"
    >
      <Scissors size={12} className="shrink-0" aria-hidden="true" />
      <span className="truncate min-w-0">
        {share === null ? kept : `${kept} · ${share}`}
      </span>
      <VerdictThumbs
        turnId={record.turnId}
        side="jev"
        // The label names the ACT, not Jev: the sentence beside it already opens with
        // Jev's name, and a second mention read as an unexplained repeat on the steer
        // line this pair is shared with.
        label={i18nT('pages.chat.compactionKeep.rate_label')}
        rightLabel={i18nT('pages.chat.compactionKeep.rate_right')}
        wrongLabel={i18nT('pages.chat.compactionKeep.rate_wrong')}
        testIdStem="compaction-keep"
      />
    </div>
  )
})

export default CompactionKeepLine
