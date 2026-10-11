import type { NodeState, SubjectState, Verdict } from '../api/types'
import { nodeStateLabel, subjectStateLabel, verdictLabel } from './labels'

/** 状态一律"文字 + 颜色"双通道，颜色不作为唯一标识（NFR-118）。 */
export function SubjectStateBadge({ state }: { state: SubjectState }) {
  return <span className={`badge badge-subject badge-${state}`}>{subjectStateLabel(state)}</span>
}

export function NodeStateBadge({ state }: { state: NodeState }) {
  return <span className={`badge badge-node badge-${state}`}>{nodeStateLabel(state)}</span>
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return <span className={`badge badge-verdict badge-verdict-${verdict}`}>{verdictLabel(verdict)}</span>
}
