import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { assessmentApi, subjectsApi, streamPaths } from '../api/endpoints'
import type { AssessmentEventPayload, ChatMessage, MasteryLevel, SseEvent } from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { AssistantText, Conversation, StreamingBubble, UserBubble } from '../components/Conversation'
import { Markdown } from '../components/Markdown'
import { useResource } from '../hooks/useResource'
import { useTurnStream } from '../hooks/useTurnStream'
import { useSubject } from './subjectContext'

const LEVEL_LABELS: Record<MasteryLevel, string> = {
  none: '无基础',
  basic: '初步了解',
  proficient: '熟练',
}

const LEVEL_OPTIONS: MasteryLevel[] = ['none', 'basic', 'proficient']

const EMPTY_TRANSCRIPT: ChatMessage[] = []

/** 画像页（P4）：子领域 × 水平 + 依据，并允许人工修正（FR-207）。 */
function ProfilePanel({ regenerate, onOverridden }: { regenerate: boolean; onOverridden: () => void }) {
  const { subjectId, state, reloadState } = useSubject()
  const navigate = useNavigate()
  const { data, error, loading, reload } = useResource(() => subjectsApi.assessment(subjectId), [subjectId])
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const assessment = data?.assessment ?? null
  // 测评对话完整保存着（FR-208）：画像页要能展开回顾"当时是怎么被问、怎么答的"
  const transcript: ChatMessage[] = data?.transcript ?? EMPTY_TRANSCRIPT

  if (loading) return <Spinner label="正在加载画像…" />
  if (error) return <Banner onRetry={reload}>{error}</Banner>
  if (!assessment) return null

  return (
    <div className="stack">
      {regenerate ? <Banner tone="notice">画像已修正，请重新生成学习路径，否则路径与画像会不一致。</Banner> : null}
      {problem ? <Banner>{problem}</Banner> : null}

      <div className="card">
        <h2>掌握度画像</h2>
        {assessment.summary ? <Markdown text={assessment.summary} /> : <p className="muted">未生成总体总结。</p>}
        {assessment.suggested_start_node ? (
          <p className="small muted">建议起始：{assessment.suggested_start_node}</p>
        ) : null}
        {assessment.skipped ? <p className="small muted">本次未进行知识基础确认，路径按零基础安排。</p> : null}
      </div>

      <div className="card">
        {assessment.levels.length === 0 ? (
          <p className="muted">没有子领域记录。</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: '26%' }}>子领域</th>
                <th style={{ width: '22%' }}>掌握水平</th>
                <th>判定依据</th>
              </tr>
            </thead>
            <tbody>
              {assessment.levels.map((level) => (
                <tr key={level.topic}>
                  <td>
                    {level.topic}
                    {level.user_overridden ? <span className="muted small">（已手动修正）</span> : null}
                  </td>
                  <td>
                    <select
                      aria-label={`${level.topic} 的掌握水平`}
                      value={level.level}
                      disabled={busy}
                      onChange={(event) => {
                        const next = event.target.value
                        setBusy(true)
                        setProblem(null)
                        void assessmentApi
                          .overrideLevel(subjectId, level.topic, next)
                          .then(() => {
                            onOverridden()
                            reload()
                            reloadState()
                          })
                          .catch((cause: unknown) => setProblem(errorMessage(cause)))
                          .finally(() => setBusy(false))
                      }}
                    >
                      {LEVEL_OPTIONS.map((option) => (
                        <option key={option} value={option}>
                          {LEVEL_LABELS[option]}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="small">{level.evidence || <span className="muted">（无）</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {transcript.length > 0 ? (
        <div className="card">
          <details>
            <summary>测评过程回顾（{transcript.length} 条）</summary>
            <p className="muted small">
              下面是这次知识基础确认的完整问答；画像里的每个水平都来自其中的具体表现。
            </p>
            <Conversation>
              {transcript.map((item, index) =>
                item.role === 'user' ? (
                  <UserBubble key={`r-u-${index}`} text={item.content} />
                ) : (
                  <AssistantText key={`r-a-${index}`} text={item.content} />
                ),
              )}
            </Conversation>
          </details>
        </div>
      ) : null}

      <div className="btn-row">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => navigate('../path', { relative: 'path', state: { generate: true } })}
        >
          生成学习路径
        </button>
        <button type="button" className="btn" onClick={() => navigate('../path')}>
          去路径页
        </button>
      </div>
      <p className="muted small">当前状态：{state.session.state} · 修正画像后需重新生成路径才能进入教学。</p>
    </div>
  )
}

/** 测评对话页（P3）：逐轮提问 + 流式 + 可跳过。 */
function AssessmentChat() {
  const { subjectId, reloadState } = useSubject()
  const stream = useTurnStream()
  const [pendingAnswer, setPendingAnswer] = useState<string | null>(null)
  const [answer, setAnswer] = useState('')
  const [turnInfo, setTurnInfo] = useState<{ total: number; topic: string | null } | null>(null)
  const [domainError, setDomainError] = useState<string | null>(null)

  const { data, error, loading, reload } = useResource(() => subjectsApi.assessment(subjectId), [subjectId])
  const transcript: ChatMessage[] = data?.transcript ?? EMPTY_TRANSCRIPT
  const pendingQuestion = data?.pending_question ?? null
  const assessment = data?.assessment ?? null

  // 本地乐观气泡与服务端记录**在渲染时对齐**：一旦服务端已经有了这条回答就不再显示它。
  // 用渲染期判断而不是 useEffect，避免"effect 里 setState → 再渲染一次"的级联。
  const showPendingAnswer = pendingAnswer !== null && !transcript.some((item) => item.role === 'user' && item.content === pendingAnswer)

  // 同理：流式正文一旦落进服务端记录，就不再用它渲染行内气泡，避免同一句话出现两次
  const lastMessage = transcript[transcript.length - 1]
  const streamConverged =
    !stream.streaming && stream.text !== '' && lastMessage?.role === 'assistant' && lastMessage.content.trim() === stream.text.trim()
  const showLiveBubble = stream.streaming || (stream.text !== '' && !streamConverged)

  function afterTurn(): void {
    reload()
    reloadState()
  }

  function handleEvent(event: SseEvent): void {
    if (event.type === 'error') {
      setDomainError(String(event.data.message ?? '测评出现问题，请重试。'))
      return
    }
    if (event.type !== 'assessment') return
    const payload = event.data as unknown as AssessmentEventPayload
    setTurnInfo({ total: payload.total_turns, topic: payload.current_topic })
    if (payload.status === 'profile') {
      afterTurn()
    }
  }

  function submit(text: string): void {
    if (stream.streaming) return
    setDomainError(null)
    setPendingAnswer(text)
    setAnswer('')
    void stream.run(streamPaths.assessmentAnswer(subjectId), { answer: text }, handleEvent).then(afterTurn)
  }

  if (loading) return <Spinner label="正在加载测评…" />
  if (error) return <Banner onRetry={reload}>{error}</Banner>
  if (!assessment) return null

  const topics = assessment.topics
  // 兜底：万一事件里没带轮次（例如历史版本后端），也不要显示成 "2/0"
  const total = Math.max(turnInfo?.total ?? 0, assessment.max_turns)
  const nextTopic = topics.length > assessment.turn_count ? topics[assessment.turn_count] : null
  const currentTopic = turnInfo?.topic ?? nextTopic
  const canAnswer = !stream.streaming && (pendingQuestion !== null || turnInfo !== null)

  return (
    <div className="stack">
      <div className="card">
        <div className="row-between">
          <h2>知识基础确认</h2>
          <span className="muted small">
            已考察 {assessment.turn_count}/{total} 个方面{currentTopic ? ` · 当前：${currentTopic}` : ''}
          </span>
        </div>
        <details>
          <summary>本次会考察的 {topics.length} 个方面</summary>
          <ol>
            {topics.map((topic) => (
              <li key={topic}>{topic}</li>
            ))}
          </ol>
        </details>
      </div>

      {error ? <Banner onRetry={reload}>{error}</Banner> : null}
      {domainError ? <Banner>{domainError}</Banner> : null}
      {stream.httpError ? <Banner>{stream.httpError}</Banner> : null}

      <div className="card">
        <Conversation>
          {transcript.map((item, index) =>
            item.role === 'user' ? (
              <UserBubble key={`u-${index}`} text={item.content} />
            ) : (
              <AssistantText key={`a-${index}`} text={item.content} />
            ),
          )}
          {showPendingAnswer ? <UserBubble text={pendingAnswer ?? ''} /> : null}
          {showLiveBubble ? (
            <StreamingBubble
              text={stream.text}
              reasoning={stream.reasoning}
              streaming={stream.streaming}
              stopped={stream.stopped}
              showReasoning={false}
            />
          ) : null}
        </Conversation>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault()
            const text = answer.trim()
            if (text && canAnswer) submit(text)
          }}
        >
          <textarea
            aria-label="你的回答"
            value={answer}
            placeholder="用自己的话回答，写不清也没关系——说不清本身就是有用的信息"
            onChange={(event) => setAnswer(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
          />
          <div className="composer-actions">
            {stream.streaming ? (
              <button type="button" className="btn" onClick={stream.stop}>
                停止生成
              </button>
            ) : (
              <button type="submit" className="btn btn-primary" disabled={!answer.trim() || !canAnswer}>
                提交回答
              </button>
            )}
            <button
              type="button"
              className="btn btn-sm"
              disabled={!canAnswer}
              onClick={() => submit('这方面我不了解。')}
            >
              这方面不了解
            </button>
          </div>
        </form>
      </div>

      <div className="btn-row">
        <button
          type="button"
          className="btn"
          disabled={stream.streaming}
          onClick={() => {
            void assessmentApi
              .skip(subjectId)
              .then(() => {
                reload()
                reloadState()
              })
              .catch((cause: unknown) => setDomainError(errorMessage(cause)))
          }}
        >
          跳过测评，按零基础安排
        </button>
      </div>
    </div>
  )
}

export function AssessmentPage() {
  const { subjectId, state, reloadState } = useSubject()
  const stream = useTurnStream()
  const [problem, setProblem] = useState<string | null>(null)
  const [regenerate, setRegenerate] = useState(false)

  const { data, error, reload } = useResource(
    () => subjectsApi.assessment(subjectId),
    // 依赖状态机：测评结束时状态会从 assessing 变成 path_review，
    // 只有把状态纳入依赖，本页才会重新拉取并切到画像视图
    [subjectId, state.session.state],
  )
  const assessment = data?.assessment ?? null

  // 刷新时保留现有界面：换回骨架屏会把正在流式生成的问题一起丢掉
  if (data === null) {
    return error !== null ? <Banner onRetry={reload}>{error}</Banner> : <Spinner label="正在加载…" />
  }

  if (assessment !== null && assessment.status !== 'running') {
    return (
      <>
        {error ? <Banner onRetry={reload}>{error}</Banner> : null}
        <ProfilePanel regenerate={regenerate} onOverridden={() => setRegenerate(true)} />
      </>
    )
  }

  if (assessment === null) {
    return (
      <div className="stack">
        {error ? <Banner onRetry={reload}>{error}</Banner> : null}
        <div className="card">
          <h2>先确认你的知识基础</h2>
          <p>
            接下来会围绕「{state.subject.title}」问几个方面的问题，每个方面一个问题。你只需要用自己的话说说目前的了解 ——
            答不上来也完全可以，那本身就是有用的信息。
          </p>
          <p className="muted small">
            这一步只用来决定从哪儿开始讲，不是考试；结束后你会看到一份掌握度画像，可以手动修正。
          </p>
        </div>

        {problem ? <Banner>{problem}</Banner> : null}
        {stream.httpError ? <Banner>{stream.httpError}</Banner> : null}

        <div className="btn-row">
          <button
            type="button"
            className="btn btn-primary"
            disabled={stream.streaming}
            onClick={() => {
              setProblem(null)
              void stream
                .run(streamPaths.assessmentStart(subjectId))
                .then(() => {
                  reload()
                  reloadState()
                })
                .catch((cause: unknown) => setProblem(errorMessage(cause)))
            }}
          >
            {stream.streaming ? '正在准备问题…' : '开始知识基础确认'}
          </button>
          <button
            type="button"
            className="btn"
            disabled={stream.streaming}
            onClick={() => {
              void assessmentApi
                .skip(subjectId)
                .then(() => {
                  reload()
                  reloadState()
                })
                .catch((cause: unknown) => setProblem(errorMessage(cause)))
            }}
          >
            跳过测评，按零基础安排
          </button>
        </div>
      </div>
    )
  }

  return <AssessmentChat />
}
