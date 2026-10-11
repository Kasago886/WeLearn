import { Markdown } from './Markdown'
import { VerdictBadge } from './StatusBadge'
import { nodeStateLabel } from './labels'
import type { NodeStatePayload, VerdictPayload } from '../api/types'

export function Banner({
  tone = 'error',
  children,
  onRetry,
}: {
  tone?: 'error' | 'notice'
  children: React.ReactNode
  onRetry?: () => void
}) {
  return (
    <div className={`banner banner-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <span>{children}</span>
      {onRetry ? (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onRetry}>
          重试
        </button>
      ) : null}
    </div>
  )
}

export function Spinner({ label = '处理中…' }: { label?: string }) {
  return (
    <span className="spinner" role="status">
      <span className="spinner-dot" />
      {label}
    </span>
  )
}

/** 判定结果卡片（FR-404）：结论 + 命中要点 + 缺失要点都要可见。 */
export function VerdictCard({ payload }: { payload: VerdictPayload }) {
  return (
    <div className={`card verdict-card verdict-${payload.verdict}`}>
      <div className="verdict-head">
        <VerdictBadge verdict={payload.verdict} />
        <span className="muted">置信度 {(payload.confidence * 100).toFixed(0)}%</span>
      </div>
      {payload.feedback ? <Markdown text={payload.feedback} /> : null}
      {payload.hit_points.length > 0 ? (
        <div className="points">
          <span className="points-title">命中的要点</span>
          <ul>
            {payload.hit_points.map((point) => (
              <li key={point}>{point}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {payload.missing_points.length > 0 ? (
        <div className="points points-missing">
          <span className="points-title">还缺什么</span>
          <ul>
            {payload.missing_points.map((point) => (
              <li key={point}>{point}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}

export function NodeStateNotice({ payload }: { payload: NodeStatePayload }) {
  return (
    <p className="node-state-notice">
      节点 {payload.index}/{payload.total} → <strong>{nodeStateLabel(payload.state)}</strong>
    </p>
  )
}

export function ReasoningPanel({ text, open }: { text: string; open: boolean }) {
  if (!text.trim()) {
    return null
  }
  return (
    <details className="reasoning" open={open}>
      <summary>思维链（推理过程，可折叠）</summary>
      <pre>{text}</pre>
    </details>
  )
}
