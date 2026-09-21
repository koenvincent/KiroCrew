import { memo } from 'react'
import { Split } from 'lucide-react'

import { fmtList, fmtNumber } from '../../i18n/format'
import { i18nT } from '../../i18n/t'
import { useLanguageGeneration } from '../../i18n/useLanguageGeneration'
import VerdictThumbs from './DecisionVerdictThumbs'
import type { SplitChoice, SplitDecisionRecord } from './decisionRecord'

/** Two decimals, so `0.84` reads as a score and not as a rounded `0.8`. */
const confidence = (p: number) => fmtNumber(p, { minimumFractionDigits: 2, maximumFractionDigits: 2 })

/** The word for one shape, in the reader's language. */
function shapeWord(choice: SplitChoice): string {
  if (choice === 'delegate') return i18nT('pages.chat.decisionStrip.split_shape_delegate')
  if (choice === 'split') return i18nT('pages.chat.decisionStrip.split_shape_split')
  return i18nT('pages.chat.decisionStrip.split_shape_single')
}

/**
 * The transcript's receipt for one task-shape suggestion.
 *
 * `task.split` is ADVICE: the suggestion reached the turn as one prepended line and
 * the agent chose for itself. So the line has to name BOTH arms -- otherwise a
 * reader cannot tell a suggestion that was taken from one that was ignored, and an
 * advisory nobody can score is an advisory nobody can withdraw.
 *
 * The agent's arm carries its own COUNT beside the word, because the word is a
 * bucket: `single` is no spawn call, `delegate` is one, `split` is two or more.
 * Printing only the bucket would hide the difference between two helpers and ten.
 *
 * ONE line, not a disclosure card like `DecisionStrip`. Everything this decision
 * produced fits on it: the two shapes, the score, the latency. There is no menu to
 * expand, and a collapsed card that hides nothing is a control that does nothing.
 *
 * The record on the row is the ONLY condition, the rule the strip states: a stamped
 * record is history that already sits on this machine, so drawing it sends nothing,
 * and gating it on the current consent switch would lose the receipt for every past
 * turn the moment the preview is turned off.
 *
 * The thumbs are `DecisionStrip`'s own pair, shared rather than copied. Side `jev`:
 * the rateable claim is the SUGGESTION, and the other arm is what the agent did,
 * which is behaviour rather than an answer to rate.
 */
const SplitDecisionLine = memo(function SplitDecisionLine({
  record,
}: {
  record: SplitDecisionRecord
}) {
  // memo() bails out of the provider-level repaint; subscribe so a language
  // switch repaints this row's strings.
  useLanguageGeneration()

  const jev = i18nT('pages.chat.decisionStrip.split_jev_chose', { shape: shapeWord(record.jevChoice) })
  // The agent's arm names the count as well as the shape: `split` covers two
  // helpers and twenty, and the number is the only part of it a reader can act on.
  const agent = record.spawnCalls > 0
    ? i18nT('pages.chat.decisionStrip.split_agent_spawned', {
      shape: shapeWord(record.agentChoice),
      count: fmtNumber(record.spawnCalls),
    })
    : i18nT('pages.chat.decisionStrip.split_agent_inline', { shape: shapeWord(record.agentChoice) })
  const verdict = record.agree
    ? i18nT('pages.chat.decisionStrip.split_agreed')
    : i18nT('pages.chat.decisionStrip.split_differed')
  const latency = record.latencyMs > 0
    ? i18nT('pages.chat.decisionStrip.latency_value', { ms: fmtNumber(record.latencyMs) })
    : null
  // Both numbers describe the answer, so they share one parenthetical, joined the
  // way the strip joins its own: the separator between two measurements read as one
  // value is a locale's decision, not this file's. The score carries its own WORD,
  // because a bare number beside a sentence is unreadable on touch and silent to a
  // screen reader.
  const scores = [
    record.p !== null
      ? i18nT('pages.chat.decisionStrip.steer_confidence', { p: confidence(record.p) })
      : null,
    latency,
  ].filter((part): part is string => part !== null)
  // The legend names what the group actually holds, so all three cases get their
  // own sentence instead of one describing a number that is not there.
  const scoresTitle = record.p !== null && latency !== null
    ? i18nT('pages.chat.decisionStrip.confidence_latency_title')
    : record.p !== null
      ? i18nT('pages.chat.decisionStrip.confidence_title')
      : i18nT('pages.chat.decisionStrip.latency_title')

  return (
    <div
      className="inline-flex items-center gap-1.5 text-[12px] leading-5 text-muted mb-1 pr-1 min-w-0"
      data-testid="split-decision-line"
      data-jev-choice={record.jevChoice}
      data-agent-choice={record.agentChoice}
      data-agree={record.agree}
    >
      <Split size={12} className="shrink-0" aria-hidden="true" />
      <span className="truncate min-w-0">
        {fmtList([jev, agent, verdict], { type: 'unit' })}
      </span>
      {scores.length > 0 && (
        <span className="shrink-0 tabular-nums" title={scoresTitle} data-testid="split-decision-scores">
          ({fmtList(scores, { type: 'unit' })})
        </span>
      )}
      <VerdictThumbs
        turnId={record.turnId}
        side="jev"
        // NOT the strip's "Jev" label: the line already opens with Jev's name, so
        // the pair beside it would read as a second, unexplained mention of it.
        // What is rateable here is the SUGGESTION, so the label names that.
        label={i18nT('pages.chat.decisionStrip.split_rate_label')}
        rightLabel={i18nT('pages.chat.decisionStrip.split_rate_right')}
        wrongLabel={i18nT('pages.chat.decisionStrip.split_rate_wrong')}
      />
    </div>
  )
})

export default SplitDecisionLine
