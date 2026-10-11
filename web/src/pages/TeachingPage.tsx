import { useState } from 'react'

import { errorMessage } from '../api/client'
import { subjectsApi, streamPaths } from '../api/endpoints'
import type { AnswerRecord, NodeDecision, OptionsPayload, SseEvent, VerdictPayload } from '../api/types'
import { Banner, Spinner, VerdictCard } from '../components/Banner'
import { AssistantText, Conversation, StreamingBubble, UserBubble } from '../components/Conversation'
import { Markdown } from '../components/Markdown'
import { NodeStateBadge } from '../components/StatusBadge'
import { useResource } from '../hooks/useResource'
import { useTurnStream } from '../hooks/useTurnStream'
import { useSubject } from './subjectContext'

const DIRECTION_LABELS: Record<string, string> = {
  simpler: '讲得更基础一些',
  example: '换一个例子',
  hint: '只给提示，别直接说答案',
}

const CHOICE_LABELS: Record<NodeDecision, string> = {
  reteach: '继续换角度重讲',
  show_answer: '查看参考答案',
  advance: '直接进入下一个节点',
}

const EMPTY_ANSWERS: AnswerRecord[] = []

/** 路径侧栏：节点目录 + 状态 + 当前节点高亮 + 整体进度（P6 布局的左栏）。 */
function PathSidebar({ currentNodeId }: { currentNodeId: string | null }) {
  const { state } = useSubject()
  const nodes = state.path?.nodes ?? []
  const mastered = nodes.filter((node) => node.state === 'mastered').length
  const ratio = nodes.length > 0 ? mastered / nodes.length : 0

  return (
    <aside className="card card-tight">
      <h2 style={{ fontSize: '1rem' }}>学习路径</h2>
      <p className="muted small">
        已掌握 {mastered}/{nodes.length}
      </p>
      <div className="progress-bar" aria-label={`掌握度 ${Math.round(ratio * 100)}%`}>
        <span style={{ width: `${Math.round(ratio * 100)}%` }} />
      </div>
      <ol className="node-list" style={{ marginTop: 12 }}>
        {nodes.map((node) => (
          <li key={node.id} className={`node-item${node.id === currentNodeId ? ' current' : ''}`}>
            <div className="node-item-head">
              <span className="small">
                {node.index}. {node.title}
              </span>
            </div>
            <NodeStateBadge state={node.state} />
            {node.attempts > 0 ? <span className="muted small"> 尝试 {node.attempts} 次</span> : null}
          </li>
        ))}
      </ol>
    </aside>
  )
}

/** 三选一（FR-408）：重试耗尽后必须由用户决定下一步，后端不会自动前进。 */
function OptionsCard({
  payload,
  disabled,
  onDecide,
}: {
  payload: OptionsPayload
  disabled: boolean
  onDecide: (decision: NodeDecision, direction?: string) => void
}) {
  const [direction, setDirection] = useState('simpler')
  const directions = payload.directions.length > 0 ? payload.directions : Object.keys(DIRECTION_LABELS)

  return (
    <div className="card">
      <h2>接下来怎么做？</h2>
      <p>{payload.message}</p>
      <div className="field" style={{ maxWidth: 320 }}>
        <label htmlFor="direction">重讲的方向</label>
        <select id="direction" value={direction} onChange={(event) => setDirection(event.target.value)}>
          {directions.map((item) => (
            <option key={item} value={item}>
              {DIRECTION_LABELS[item] ?? item}
            </option>
          ))}
        </select>
      </div>
      <div className="btn-row">
        {(payload.choices.length > 0 ? payload.choices : (['reteach', 'show_answer', 'advance'] as NodeDecision[])).map(
          (choice) => (
            <button
              key={choice}
              type="button"
              className={choice === 'advance' ? 'btn' : 'btn btn-primary'}
              disabled={disabled}
              onClick={() => onDecide(choice, choice === 'reteach' ? direction : undefined)}
            >
              {CHOICE_LABELS[choice] ?? choice}
            </button>
          ),
        )}
      </div>
    </div>
  )
}

export function TeachingPage() {
  const { subjectId, state, reloadState } = useSubject()
  const stream = useTurnStream()
  const [verdict, setVerdict] = useState<VerdictPayload | null>(null)
  const [options, setOptions] = useState<OptionsPayload | null>(null)
  const [domainError, setDomainError] = useState<string | null>(null)
  const [pendingAnswer, setPendingAnswer] = useState<string | null>(null)
  const [answer, setAnswer] = useState('')
  const [busy, setBusy] = useState(false)

  const { data, error, reload } = useResource(() => subjectsApi.teaching(subjectId), [subjectId])

  const session = data?.session ?? state.session
  const node = data?.node ?? state.node
  const pendingQuestion = session.pending_question
  const answers: AnswerRecord[] = data?.answers ?? EMPTY_ANSWERS
  const paused = session.state === 'paused'
  const completed = session.state === 'completed'
  const needsDecision = data?.needs_decision ?? state.needs_decision

  // 本地乐观气泡与服务端记录在渲染时对齐：服务端一旦有了这条作答就不再显示本地副本
  const showPendingAnswer =
    pendingAnswer !== null && !answers.some((record) => record.answer === pendingAnswer)

  function afterTurn(): void {
    reload()
    reloadState()
  }

  function handleEvent(event: SseEvent): void {
    if (event.type === 'error') {
      setDomainError(String(event.data.message ?? '这一步没能完成，请重试。'))
      return
    }
    if (event.type === 'verdict') {
      setVerdict(event.data as unknown as VerdictPayload)
      return
    }
    if (event.type === 'options') {
      setOptions(event.data as unknown as OptionsPayload)
      return
    }
    if (event.type === 'node_state') {
      setOptions(null)
    }
  }

  function start(): void {
    setDomainError(null)
    setVerdict(null)
    setOptions(null)
    void stream.run(streamPaths.teachStart(subjectId), undefined, handleEvent).then(afterTurn)
  }

  function submit(text: string): void {
    if (stream.streaming) return
    setDomainError(null)
    setVerdict(null)
    setOptions(null)
    setPendingAnswer(text)
    setAnswer('')
    void stream.run(streamPaths.teachAnswer(subjectId), { answer: text }, handleEvent).then(afterTurn)
  }

  function decide(decision: NodeDecision, direction?: string): void {
    setDomainError(null)
    setOptions(null)
    void stream
      .run(
        streamPaths.teachDecision(subjectId),
        direction ? { decision, direction } : { decision },
        handleEvent,
      )
      .then(afterTurn)
  }

  function act(action: Promise<unknown>): void {
    setBusy(true)
    setDomainError(null)
    void action
      .then(afterTurn)
      .catch((cause: unknown) => setDomainError(errorMessage(cause)))
      .finally(() => setBusy(false))
  }

  // 同样地：刷新数据时保留现有界面，避免把手里的流式内容换掉
  if (data === null) {
    return error !== null ? <Banner onRetry={reload}>{error}</Banner> : <Spinner label="正在加载教学页…" />
  }

  const generating = stream.streaming
  const canAnswer = (data?.can_answer ?? state.can_answer) && !generating && !paused

  // 判定卡片会被写进作答记录（刷新后仍在），因此一旦服务端已经有了这张卡，
  // 就不再单独浮一张同样内容的卡，否则同一份判定会显示两遍
  const verdictInHistory =
    verdict !== null &&
    answers.some(
      (record) =>
        record.verdict === verdict.verdict &&
        record.feedback === verdict.feedback &&
        record.missing_points.join('|') === verdict.missing_points.join('|'),
    )
  const showFloatingVerdict = verdict !== null && !verdictInHistory

  return (
    <div className="workspace">
      <PathSidebar currentNodeId={node?.id ?? null} />

      <div className="stack">
        {error ? <Banner onRetry={reload}>{error}</Banner> : null}
        {domainError ? <Banner>{domainError}</Banner> : null}
        {stream.httpError ? <Banner>{stream.httpError}</Banner> : null}
        {paused ? (
          <Banner tone="notice">
            学习已暂停，作答通道已关闭。
            <button type="button" className="btn btn-sm" disabled={busy} onClick={() => act(subjectsApi.resume(subjectId))}>
              继续学习
            </button>
          </Banner>
        ) : null}
        {completed ? (
          <Banner tone="notice">本次学习的所有节点都已完成，可以回到主题列表开始新的主题。</Banner>
        ) : null}

        <div className="card">
          <div className="row-between">
            <h2>
              {node ? `${data?.node_index ?? node.index}/${data?.node_total ?? state.path?.nodes.length ?? 1} · ${node.title}` : '当前节点'}
              {node ? <NodeStateBadge state={node.state} /> : null}
            </h2>
            <div className="btn-row">
              {paused ? (
                <button type="button" className="btn btn-sm" disabled={busy} onClick={() => act(subjectsApi.resume(subjectId))}>
                  继续学习
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={busy || generating || completed}
                  onClick={() => act(subjectsApi.pause(subjectId))}
                >
                  暂停
                </button>
              )}
              <button
                type="button"
                className="btn btn-sm"
                disabled={busy || generating || node === null || completed}
                onClick={() => {
                  if (window.confirm('确定跳过当前节点吗？它会记入薄弱点，之后可以复习补上。')) {
                    setVerdict(null)
                    setOptions(null)
                    void stream.run(streamPaths.teachSkip(subjectId), undefined, handleEvent).then(afterTurn)
                  }
                }}
              >
                跳过本节点
              </button>
            </div>
          </div>

          {node ? (
            <>
              <p className="small muted">目标：{node.objective}</p>
              {node.key_points.length > 0 ? (
                <details open>
                  <summary>本节点要点</summary>
                  <ul className="small">
                    {node.key_points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                </details>
              ) : null}
              {node.summary_note ? (
                <details>
                  <summary>本节点小结</summary>
                  <Markdown text={node.summary_note} />
                </details>
              ) : null}
            </>
          ) : (
            <p className="muted">还没有进行中的节点，点击「开始学习」进入第一个节点。</p>
          )}
        </div>

        <div className="card">
          <Conversation>
            {answers.map((record) => (
              <div key={record.id} style={{ display: 'contents' }}>
                <AssistantText text={record.question} />
                <UserBubble text={record.answer} />
                <div className="row row-assistant">
                  <div style={{ maxWidth: '88%', width: '100%' }}>
                    <VerdictCard
                      payload={{
                        verdict: record.verdict,
                        confidence: record.confidence ?? 0,
                        feedback: record.feedback,
                        hit_points: record.hit_points,
                        missing_points: record.missing_points,
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}

            {showPendingAnswer ? <UserBubble text={pendingAnswer ?? ''} /> : null}

            {pendingQuestion ? <AssistantText text={pendingQuestion} /> : null}

            {stream.streaming || stream.text ? (
              <StreamingBubble
                text={stream.text}
                reasoning={stream.reasoning}
                streaming={stream.streaming}
                stopped={stream.stopped}
                showReasoning={true}
              />
            ) : null}
          </Conversation>

          {showFloatingVerdict ? (
            <div className="row row-assistant">
              <div style={{ maxWidth: '88%', width: '100%' }}>
                <VerdictCard payload={verdict} />
              </div>
            </div>
          ) : null}

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
              disabled={!canAnswer && !generating}
              placeholder={
                pendingQuestion
                  ? '写出你的理解，尽量讲清关键点'
                  : '点击「开始学习」让老师讲解并提出检查点问题'
              }
              onChange={(event) => setAnswer(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
            />
            <div className="composer-actions">
              {generating ? (
                <button type="button" className="btn" onClick={stream.stop}>
                  停止生成
                </button>
              ) : (
                <button type="submit" className="btn btn-primary" disabled={!answer.trim() || !canAnswer}>
                  提交作答
                </button>
              )}
              {!pendingQuestion && !generating && !completed && !paused ? (
                <button type="button" className="btn" onClick={start}>
                  {session.node_taught ? '继续学习' : '开始学习'}
                </button>
              ) : null}
            </div>
          </form>
        </div>

        {needsDecision && options === null && !generating ? (
          <OptionsCard
            payload={{
              kind: 'exhausted',
              choices: ['reteach', 'show_answer', 'advance'],
              message: '这个节点已经尝试多次。你可以选择：继续换个角度重讲、查看参考答案，或直接进入下一个节点。',
              directions: ['simpler', 'example', 'hint'],
            }}
            disabled={busy}
            onDecide={decide}
          />
        ) : null}

        {options !== null && !generating ? (
          <OptionsCard payload={options} disabled={busy} onDecide={decide} />
        ) : null}
      </div>
    </div>
  )
}
