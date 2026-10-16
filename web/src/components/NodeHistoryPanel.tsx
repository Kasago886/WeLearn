import { subjectsApi } from '../api/endpoints'
import type { AnswerRecord } from '../api/types'
import { useResource } from '../hooks/useResource'
import { Banner, Spinner, VerdictCard } from './Banner'

const SOURCE_LABELS: Record<AnswerRecord['source'], string> = {
  teach: '教学作答',
  review: '复习作答',
}

function formatTime(value: string | null): string {
  if (!value) return '时间未知'
  const date = new Date(value.endsWith('Z') ? value : `${value}Z`)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

/**
 * 单个节点的历次提问 / 作答 / 判定 / 反馈（FR-802、FR-507）。
 *
 * 教学页与档案页共用它：两处要看到的是同一份记录，分成两套渲染会慢慢长歪。
 * 「来源」必须显式标出——复习作答与教学作答混在一起时，用户会以为自己答过的问题
 * 又被问了一遍。
 */
export function NodeHistoryPanel({
  subjectId,
  nodeId,
  refreshKey = 0,
}: {
  subjectId: string
  nodeId: string
  refreshKey?: number
}) {
  const { data, error, loading, reload } = useResource(
    () => subjectsApi.nodeHistory(subjectId, nodeId),
    [subjectId, nodeId, refreshKey],
  )

  if (data === null) {
    return loading ? <Spinner label="正在加载作答历史…" /> : <Banner onRetry={reload}>{error}</Banner>
  }

  return (
    <div className="stack">
      {error ? <Banner onRetry={reload}>{error}</Banner> : null}
      <p className="muted small">
        共 {data.items.length} 次作答 · 教学 {data.teach_count} 次 · 复习 {data.review_count} 次
      </p>
      {data.items.length === 0 ? (
        <p className="muted small">这个节点还没有作答记录。</p>
      ) : (
        data.items.map((record) => (
          <article className="history-item" key={record.id}>
            <div className="row-between">
              <span className="badge badge-node">{SOURCE_LABELS[record.source] ?? record.source}</span>
              <span className="muted small">
                {/* attempt_index 从 0 起（后端按数组下标记），展示给人看要 +1 */}
                第 {record.attempt_index + 1} 次 · {formatTime(record.created_at)}
              </span>
            </div>
            <p className="small">
              <strong>问题：</strong>
              {record.question}
            </p>
            <p className="small">
              <strong>你的回答：</strong>
              {record.answer}
            </p>
            <VerdictCard
              payload={{
                verdict: record.verdict,
                confidence: record.confidence ?? 0,
                feedback: record.feedback,
                hit_points: record.hit_points,
                missing_points: record.missing_points,
              }}
            />
          </article>
        ))
      )}
    </div>
  )
}
