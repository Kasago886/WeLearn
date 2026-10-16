import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { subjectsApi } from '../api/endpoints'
import type { ArchiveNode, ReportEntry } from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { NodeHistoryPanel } from '../components/NodeHistoryPanel'
import { NodeStateBadge } from '../components/StatusBadge'
import { useResource } from '../hooks/useResource'

const WEAK_REASON_LABELS: Record<string, string> = {
  partial: '部分掌握',
  skipped: '曾跳过',
  show_answer: '查看过参考答案',
  advanced: '答错后选择继续',
}

function formatTime(value: string | null): string {
  if (!value) return '暂无记录'
  const date = new Date(value.endsWith('Z') ? value : `${value}Z`)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

/** 累计模型调用耗时（FR-417）。这是"生成花了多久"，不含用户阅读时间。 */
function formatDuration(durationMs: number): string {
  if (durationMs <= 0) return '—'
  const seconds = Math.round(durationMs / 1000)
  if (seconds < 60) return `${seconds} 秒`
  return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`
}

function ReportList({
  title,
  entries,
  emptyText,
}: {
  title: string
  entries: ReportEntry[]
  emptyText: string
}) {
  return (
    <div>
      <h3>
        {title}（{entries.length}）
      </h3>
      {entries.length === 0 ? (
        <p className="muted small">{emptyText}</p>
      ) : (
        <ul className="small">
          {entries.map((entry) => (
            <li key={entry.id}>
              {entry.index}. {entry.title}
              {entry.reasons && entry.reasons.length > 0 ? (
                <span className="muted">
                  （{entry.reasons.map((reason) => WEAK_REASON_LABELS[reason] ?? reason).join('、')}）
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** 单个节点的档案条目：要点 + 笔记（可编辑）+ 作答历史。 */
function NodeSection({
  subjectId,
  node,
  expanded,
  saving,
  onToggle,
  onSaveNote,
}: {
  subjectId: string
  node: ArchiveNode
  expanded: boolean
  saving: boolean
  onToggle: () => void
  onSaveNote: (content: string) => void
}) {
  const [draft, setDraft] = useState(node.note?.content ?? '')
  const [editing, setEditing] = useState(false)

  return (
    <li className="card stack">
      <div className="row-between">
        <div>
          <span className="node-title">
            {node.index}. {node.title}
          </span>{' '}
          <NodeStateBadge state={node.state} />
          <p className="muted small">
            作答 {node.answer_count} 次{node.review_count > 0 ? ` · 其中复习 ${node.review_count} 次` : ''}
            {node.mastered_at ? ` · 掌握于 ${formatTime(node.mastered_at)}` : ''}
          </p>
        </div>
        <button type="button" className="btn btn-sm" onClick={onToggle}>
          {expanded ? '收起' : '查看笔记与历史'}
        </button>
      </div>

      {expanded ? (
        <>
          {node.key_points.length > 0 ? (
            <ul className="small">
              {node.key_points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          ) : null}

          <div className="stack">
            <div className="row-between">
              <strong className="small">
                节点笔记
                {node.note ? (
                  <span className="muted">（{node.note.source === 'user' ? '手写' : '自动小结'}）</span>
                ) : (
                  <span className="muted">（还没有笔记）</span>
                )}
              </strong>
              {editing ? null : (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => {
                    setDraft(node.note?.content ?? '')
                    setEditing(true)
                  }}
                >
                  编辑
                </button>
              )}
            </div>

            {editing ? (
              <form
                className="stack"
                onSubmit={(event) => {
                  event.preventDefault()
                  onSaveNote(draft)
                  setEditing(false)
                }}
              >
                <textarea
                  aria-label="节点笔记"
                  rows={5}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="写下这个节点你自己的理解、易错点或例子"
                />
                <div className="btn-row">
                  <button type="submit" className="btn btn-sm btn-primary" disabled={saving}>
                    {saving ? '保存中…' : '保存笔记'}
                  </button>
                  <button type="button" className="btn btn-sm" disabled={saving} onClick={() => setEditing(false)}>
                    取消
                  </button>
                  <span className="muted small">清空后保存即可删除这条笔记。</span>
                </div>
              </form>
            ) : node.note && node.note.content ? (
              <p className="small" style={{ whiteSpace: 'pre-wrap' }}>
                {node.note.content}
              </p>
            ) : (
              <p className="muted small">这个节点还没有笔记。</p>
            )}
          </div>

          <div className="stack">
            <strong className="small">作答历史</strong>
            <NodeHistoryPanel subjectId={subjectId} nodeId={node.id} />
          </div>
        </>
      ) : null}
    </li>
  )
}

/**
 * 学习档案页（P9）：进度可视化 + 逐节点笔记与作答历史 + 完成报告（FR-801/802/804/417）。
 *
 * 档案与完成报告分成两次请求：档案要能独立于"是否学完"打开，
 * 而报告里的用量与清单只在收尾时才有意义。
 */
export function ArchivePage() {
  const { subjectId = '' } = useParams()
  const archive = useResource(() => subjectsApi.archiveOverview(subjectId), [subjectId])
  const report = useResource(() => subjectsApi.completionReport(subjectId), [subjectId])
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [savingId, setSavingId] = useState<string | null>(null)
  const [problem, setProblem] = useState<string | null>(null)

  async function saveNote(nodeId: string, content: string): Promise<void> {
    setSavingId(nodeId)
    setProblem(null)
    try {
      await subjectsApi.saveNote(subjectId, nodeId, content)
      archive.reload()
    } catch (cause) {
      setProblem(errorMessage(cause))
    } finally {
      setSavingId(null)
    }
  }

  const data = archive.data
  if (data === null) {
    return archive.error !== null ? (
      <Banner onRetry={archive.reload}>{archive.error}</Banner>
    ) : (
      <Spinner label="正在加载学习档案…" />
    )
  }

  const progress = data.progress
  const ratio = progress.total_nodes > 0 ? progress.mastery_ratio : 0
  const summary = report.data

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>学习档案</h1>
          <p className="muted">
            这里汇总了每个节点的笔记、作答历史与整体进度；复习记录会与教学作答分开标注。
          </p>
        </div>
        <div className="btn-row">
          <Link className="btn btn-sm" to={`/subjects/${subjectId}/review`}>
            薄弱点复习
          </Link>
          <Link className="btn btn-sm" to={`/subjects/${subjectId}/teach`}>
            返回教学
          </Link>
        </div>
      </div>

      {problem ? <Banner>{problem}</Banner> : null}
      {archive.error ? <Banner onRetry={archive.reload}>{archive.error}</Banner> : null}

      <section className="card stack" aria-label="整体进度">
        <div className="row-between">
          <strong>整体进度</strong>
          <span className="small">
            已掌握 {progress.mastered_nodes}/{progress.total_nodes} · 部分掌握 {progress.partial_nodes} · 已跳过{' '}
            {progress.skipped_nodes} · 薄弱点 {progress.weak_points}
          </span>
        </div>
        <div className="progress-bar" aria-label={`掌握度 ${Math.round(ratio * 100)}%`}>
          <span style={{ width: `${Math.round(ratio * 100)}%` }} />
        </div>
        <dl className="detail-list">
          <dt>路径版本</dt>
          <dd>v{data.path_version ?? '—'}</dd>
          <dt>累计 token</dt>
          <dd>{progress.total_tokens.toLocaleString('zh-CN')}</dd>
          <dt>累计模型调用耗时</dt>
          <dd>{summary ? formatDuration(summary.usage.duration_ms) : '加载中…'}</dd>
          <dt>学习跨度</dt>
          <dd>
            {summary ? `${formatTime(summary.started_at)} → ${formatTime(summary.last_activity_at)}` : '加载中…'}
          </dd>
        </dl>
        {data.weak_points.length > 0 ? (
          <div>
            <strong className="small">待复习的薄弱点（{data.weak_points.length}）</strong>
            <ul className="small">
              {data.weak_points.map((point) => (
                <li key={point.id}>
                  {point.node_title} · {WEAK_REASON_LABELS[point.reason] ?? point.reason} · 未通过{' '}
                  {point.fail_count} 次 · 最近 {formatTime(point.last_attempt_at)}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="muted small">目前没有待处理的薄弱点。</p>
        )}
      </section>

      {data.nodes.length === 0 ? (
        <div className="card empty">
          <h2>还没有学习路径</h2>
          <p>生成并确认学习路径后，每个节点的笔记与作答历史都会汇总到这里。</p>
          <Link className="btn btn-primary" to={`/subjects/${subjectId}/path`}>
            去生成学习路径
          </Link>
        </div>
      ) : (
        <ol className="stack" style={{ listStyle: 'none', padding: 0 }}>
          {data.nodes.map((node) => (
            <NodeSection
              key={node.id}
              subjectId={subjectId}
              node={node}
              expanded={expandedId === node.id}
              saving={savingId === node.id}
              onToggle={() => setExpandedId((current) => (current === node.id ? null : node.id))}
              onSaveNote={(content) => void saveNote(node.id, content)}
            />
          ))}
        </ol>
      )}

      <section className="card stack" aria-label="完成报告">
        <h2>完成报告</h2>
        {summary === null ? (
          report.error !== null ? (
            <Banner onRetry={report.reload}>{report.error}</Banner>
          ) : (
            <Spinner label="正在汇总…" />
          )
        ) : (
          <>
            <div className="form-grid">
              <ReportList title="已掌握" entries={summary.mastered} emptyText="还没有已掌握的节点。" />
              <ReportList
                title="部分掌握"
                entries={summary.partial}
                emptyText="没有被放行的部分掌握节点。"
              />
              <ReportList title="已跳过" entries={summary.skipped} emptyText="没有跳过的节点。" />
            </div>
            <div>
              <h3>建议复习方向</h3>
              {summary.weak_points.length === 0 ? (
                <p className="muted small">没有薄弱点需要补强。</p>
              ) : (
                <ul className="small">
                  {summary.weak_points.map((point) => (
                    <li key={point.id}>
                      {point.node_title} · {WEAK_REASON_LABELS[point.reason] ?? point.reason} · 未通过{' '}
                      {point.fail_count} 次
                    </li>
                  ))}
                </ul>
              )}
              {summary.weak_points.length > 0 ? (
                <Link className="btn btn-sm btn-primary" to={`/subjects/${subjectId}/review`}>
                  去复习这些薄弱点
                </Link>
              ) : null}
            </div>
            <dl className="detail-list">
              <dt>模型调用</dt>
              <dd>
                {summary.usage.calls} 次 · 累计 {summary.usage.total_tokens.toLocaleString('zh-CN')} token
              </dd>
              <dt>缓存命中</dt>
              <dd>
                {summary.usage.cached_prompt_tokens.toLocaleString('zh-CN')} token（命中率{' '}
                {Math.round(summary.usage.cache_hit_rate * 100)}%）
              </dd>
              <dt>推理 token</dt>
              <dd>{summary.usage.reasoning_tokens.toLocaleString('zh-CN')}</dd>
              <dt>累计耗时</dt>
              <dd>{formatDuration(summary.usage.duration_ms)}</dd>
            </dl>
          </>
        )}
      </section>
    </div>
  )
}
