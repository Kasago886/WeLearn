import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { subjectsApi } from '../api/endpoints'
import type { AnswerRecord, ReviewTurn, VerdictPayload } from '../api/types'
import { Banner, Spinner, VerdictCard } from '../components/Banner'
import { AssistantText, Conversation, UserBubble } from '../components/Conversation'
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

function answerVerdict(record: AnswerRecord): VerdictPayload {
  return {
    verdict: record.verdict,
    confidence: record.confidence ?? 0,
    feedback: record.feedback,
    hit_points: record.hit_points,
    missing_points: record.missing_points,
  }
}

function reviewVerdict(turn: ReviewTurn): VerdictPayload | null {
  if (turn.verdict === null) return null
  return {
    verdict: turn.verdict,
    confidence: turn.confidence,
    feedback: turn.feedback,
    hit_points: turn.hit_points,
    missing_points: turn.missing_points,
  }
}

function ReviewVerdict({ turn }: { turn: ReviewTurn }) {
  const verdict = reviewVerdict(turn)
  if (verdict === null) return null
  return <VerdictCard payload={verdict} />
}

function ReviewConversation({
  subjectId,
  nodeId,
  prompt,
  submitted,
  turn,
  refreshKey,
}: {
  subjectId: string
  nodeId: string
  prompt: string | null
  submitted: { question: string; answer: string; turn: ReviewTurn } | null
  turn: ReviewTurn | null
  refreshKey: number
}) {
  const { data, error, loading, reload } = useResource(
    () => subjectsApi.nodeHistory(subjectId, nodeId),
    [subjectId, nodeId, refreshKey],
  )

  if (data === null && loading) return <Spinner label="正在加载复习记录…" />
  if (data === null && error) return <Banner onRetry={reload}>{error}</Banner>

  const records = (data?.items ?? []).filter((item) => item.source === 'review')
  const submittedIsSaved =
    submitted !== null &&
    records.some((item) => item.question === submitted.question && item.answer === submitted.answer)
  const currentPrompt = prompt ?? (turn?.phase === 'question' ? turn.text : null)
  const result = submitted?.turn ?? (turn?.phase !== 'question' ? turn : null)

  return (
    <div className="stack">
      {error ? <Banner onRetry={reload}>{error}</Banner> : null}
      {data ? (
        <p className="muted small">
          本节点共复习 {data.review_count} 次 · 教学作答 {data.teach_count} 次
        </p>
      ) : null}
      <Conversation>
        {records.map((record) => (
          <div key={record.id} style={{ display: 'contents' }}>
            <AssistantText text={record.question} />
            <UserBubble text={record.answer} />
            <div className="row row-assistant">
              <div style={{ maxWidth: '88%', width: '100%' }}>
                <VerdictCard payload={answerVerdict(record)} />
              </div>
            </div>
          </div>
        ))}
        {submitted && !submittedIsSaved ? (
          <>
            <AssistantText text={submitted.question} />
            <UserBubble text={submitted.answer} />
            {reviewVerdict(submitted.turn) ? (
              <div className="row row-assistant">
                <div style={{ maxWidth: '88%', width: '100%' }}>
                  <ReviewVerdict turn={submitted.turn} />
                </div>
              </div>
            ) : null}
            <AssistantText text={submitted.turn.text} />
          </>
        ) : null}
        {submitted && submittedIsSaved ? <AssistantText text={submitted.turn.text} /> : null}
        {currentPrompt ? <AssistantText text={currentPrompt} /> : null}
        {result && !submitted ? (
          <>
            <div className="row row-assistant">
              <div style={{ maxWidth: '88%', width: '100%' }}>
                <ReviewVerdict turn={result} />
              </div>
            </div>
            <AssistantText text={result.text} />
          </>
        ) : null}
      </Conversation>
    </div>
  )
}

export function ReviewPage() {
  const { subjectId = '' } = useParams()
  const { data, error, loading, reload } = useResource(() => subjectsApi.review(subjectId), [subjectId])
  const [draftTurn, setDraftTurn] = useState<ReviewTurn | null>(null)
  const [submitted, setSubmitted] = useState<{ question: string; answer: string; turn: ReviewTurn } | null>(null)
  const [answer, setAnswer] = useState('')
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0)

  const turn = data?.current ?? draftTurn
  const nodeId = turn?.node?.id ?? submitted?.turn.node?.id ?? null
  const question = turn?.phase === 'question' ? turn.text : null
  const isAwaitingAnswer = turn?.phase === 'question'
  const masteryRatio = data && data.node_total > 0 ? data.mastered_count / data.node_total : 0

  async function startReview(targetNodeId: string): Promise<void> {
    setBusy(true)
    setActionError(null)
    setDraftTurn(null)
    setSubmitted(null)
    setAnswer('')
    try {
      const started = await subjectsApi.startReview(subjectId, targetNodeId)
      setDraftTurn(started)
      await reload()
    } catch (cause) {
      setActionError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  async function submitAnswer(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const text = answer.trim()
    if (!text || !question || busy) return

    setBusy(true)
    setActionError(null)
    setAnswer('')
    try {
      const result = await subjectsApi.answerReview(subjectId, text)
      setSubmitted({ question, answer: text, turn: result })
      setDraftTurn(result)
      setHistoryRefreshKey((value) => value + 1)
      await reload()
    } catch (cause) {
      setAnswer(text)
      setActionError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  if (data === null) {
    return error ? <Banner onRetry={reload}>{error}</Banner> : <Spinner label="正在加载薄弱点…" />
  }

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>薄弱点与复习</h1>
          <p className="muted">从曾经卡住的节点开始，用原有检查点检验是否真正掌握。</p>
        </div>
        <Link className="btn btn-sm" to={`/subjects/${subjectId}`}>
          返回主题
        </Link>
      </div>

      {error ? <Banner onRetry={reload}>{error}</Banner> : null}
      {actionError ? <Banner>{actionError}</Banner> : null}
      {data.node_total > 0 ? (
        <section className="card" aria-label="学习进度">
          <div className="row-between">
            <strong>整体掌握进度</strong>
            <span className="small">
              已掌握 {data.mastered_count}/{data.node_total} 个节点 · 待复习 {data.open_count} 个薄弱点
            </span>
          </div>
          <div
            className="progress-bar"
            aria-label={`掌握度 ${Math.round(masteryRatio * 100)}%`}
            style={{ marginTop: 10 }}
          >
            <span style={{ width: `${Math.round(masteryRatio * 100)}%` }} />
          </div>
        </section>
      ) : null}

      {turn && nodeId ? (
        <section className="card stack" aria-label="复习会话">
          <div className="row-between">
            <div>
              <h2>
                {turn.node_index}/{turn.node_total} · {turn.node?.title ?? '当前节点'}
              </h2>
              {turn.node ? <NodeStateBadge state={turn.node.state} /> : null}
            </div>
            {turn.phase !== 'question' ? (
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => {
                  setDraftTurn(null)
                  setSubmitted(null)
                }}
              >
                收起复习会话
              </button>
            ) : null}
          </div>
          <ReviewConversation
            subjectId={subjectId}
            nodeId={nodeId}
            prompt={question}
            submitted={submitted}
            turn={turn}
            refreshKey={historyRefreshKey}
          />
          {isAwaitingAnswer ? (
            <form className="composer" onSubmit={(event) => void submitAnswer(event)}>
              <textarea
                aria-label="你的复习回答"
                value={answer}
                disabled={busy}
                placeholder="写出你对这个节点的理解"
                onChange={(event) => setAnswer(event.target.value)}
              />
              <div className="btn-row">
                <button type="submit" className="btn btn-primary" disabled={busy || !answer.trim()}>
                  {busy ? '正在判定…' : '提交复习回答'}
                </button>
              </div>
            </form>
          ) : null}
        </section>
      ) : null}

      <section className="stack" aria-label="待复习薄弱点">
        <div className="row-between">
          <h2>待复习薄弱点</h2>
          {loading ? <Spinner label="正在刷新…" /> : null}
        </div>
        {data.items.length === 0 ? (
          <div className="card empty">
            <h3>{data.node_total === 0 ? '还没有可复习的学习节点' : '目前没有待处理的薄弱点'}</h3>
            <p>
              {data.node_total === 0
                ? '完成学习路径并开始教学后，遇到的薄弱点会出现在这里。'
                : '当前薄弱点都已补齐。新的薄弱点会在教学过程中自动记录。'}
            </p>
          </div>
        ) : (
          <div className="grid-cards">
            {data.items.map((item) => (
              <article className="card stack" key={item.id}>
                <div className="row-between">
                  <h3>
                    {item.node_index}/{item.node_total} · {item.node_title}
                  </h3>
                  <span className="badge badge-node badge-review_needed">
                    {WEAK_REASON_LABELS[item.reason] ?? item.reason}
                  </span>
                </div>
                <p className="muted small">
                  连续复习未通过 {item.fail_count} 次 · 最近记录 {formatTime(item.last_attempt_at)}
                </p>
                {turn?.node?.id === item.node_id && isAwaitingAnswer ? (
                  <p className="small">正在复习这个节点，可以在上方继续作答。</p>
                ) : null}
                <div className="btn-row">
                  {turn?.node?.id === item.node_id && isAwaitingAnswer ? (
                    <span className="muted small">当前复习已在上方恢复</span>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-sm btn-primary"
                      disabled={busy || isAwaitingAnswer}
                      onClick={() => void startReview(item.node_id)}
                    >
                      开始复习
                    </button>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
