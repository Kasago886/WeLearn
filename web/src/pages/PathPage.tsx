import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { subjectsApi, streamPaths } from '../api/endpoints'
import type { KnowledgeNode, SseEvent } from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { Markdown } from '../components/Markdown'
import { NodeStateBadge } from '../components/StatusBadge'
import { useResource } from '../hooks/useResource'
import { useTurnStream } from '../hooks/useTurnStream'
import { useSubject } from './subjectContext'

const LEVEL_LABELS: Record<string, string> = {
  none: '无基础',
  basic: '初步了解',
  proficient: '熟练',
}

/**
 * 能否增删或移动这个节点。
 *
 * 改标题与要点**任何状态都允许**——那是纠错；增删移动只对"还没开始学"的节点开放，
 * 否则"已掌握"会落到别的位置上，或者会话指向一个被删掉的节点。
 */
function canRestructure(node: KnowledgeNode): boolean {
  return node.state === 'not_started' || node.state === 'review_needed'
}

/** 单节点的行内编辑表单。 */
function NodeEditor({
  node,
  busy,
  onSave,
  onCancel,
}: {
  node: KnowledgeNode
  busy: boolean
  onSave: (fields: { title: string; objective: string; key_points: string[] }) => void
  onCancel: () => void
}) {
  const [title, setTitle] = useState(node.title)
  const [objective, setObjective] = useState(node.objective)
  const [keyPoints, setKeyPoints] = useState(node.key_points.join('\n'))

  return (
    <form
      className="stack"
      onSubmit={(event) => {
        event.preventDefault()
        onSave({
          title: title.trim(),
          objective: objective.trim(),
          key_points: keyPoints
            .split('\n')
            .map((line) => line.trim())
            .filter(Boolean),
        })
      }}
    >
      <div className="field">
        <label htmlFor={`title-${node.id}`}>节点标题</label>
        <input id={`title-${node.id}`} value={title} onChange={(event) => setTitle(event.target.value)} />
      </div>
      <div className="field">
        <label htmlFor={`objective-${node.id}`}>节点目标</label>
        <input
          id={`objective-${node.id}`}
          value={objective}
          onChange={(event) => setObjective(event.target.value)}
        />
      </div>
      <div className="field">
        <label htmlFor={`points-${node.id}`}>要点（每行一条）</label>
        <textarea
          id={`points-${node.id}`}
          value={keyPoints}
          onChange={(event) => setKeyPoints(event.target.value)}
          rows={4}
        />
      </div>
      <div className="btn-row">
        <button type="submit" className="btn btn-sm btn-primary" disabled={busy || !title.trim()}>
          保存
        </button>
        <button type="button" className="btn btn-sm" onClick={onCancel} disabled={busy}>
          取消
        </button>
      </div>
    </form>
  )
}

/**
 * 学习路径页（P5）：画像回顾 + 生成理由 + 节点清单 + 编辑 + 确认门禁。
 *
 * 「确认」是硬门禁（FR-306）：没确认就不能进入教学。
 */
export function PathPage() {
  const { subjectId, state, reloadState } = useSubject()
  const location = useLocation()
  const navigate = useNavigate()
  const stream = useTurnStream()
  const [problem, setProblem] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [regeneratingId, setRegeneratingId] = useState<string | null>(null)
  const autoStarted = useRef(false)

  const { data, error, reload } = useResource(() => subjectsApi.path(subjectId), [subjectId])
  const path = data?.path ?? null
  const generating = stream.streaming
  const teaching = state.session.state === 'teaching'

  function generate(): void {
    setProblem(null)
    const onEvent = (event: SseEvent): void => {
      if (event.type === 'error') {
        setProblem(String(event.data.message ?? '生成路径失败，请重试。'))
      }
    }
    void stream.run(streamPaths.pathGenerate(subjectId), undefined, onEvent).then(() => {
      reload()
      reloadState()
    })
  }

  /** 编辑操作统一走这里：失败就是 409 的领域冲突，直接把后端文案显示出来。 */
  function edit(action: () => Promise<unknown>): Promise<void> {
    setBusy(true)
    setProblem(null)
    return action()
      .then(() => {
        reload()
      })
      .catch((cause: unknown) => setProblem(errorMessage(cause)))
      .finally(() => setBusy(false))
  }

  function regenerateNode(nodeId: string, requirement: string): void {
    setProblem(null)
    setRegeneratingId(nodeId)
    const onEvent = (event: SseEvent): void => {
      if (event.type === 'error') {
        setProblem(String(event.data.message ?? '重生成节点失败，请重试。'))
      }
    }
    void stream
      .run(streamPaths.pathRegenerateNode(subjectId, nodeId), { requirement }, onEvent)
      .then(() => {
        setEditingId(null)
        reload()
        reloadState()
      })
      .finally(() => setRegeneratingId(null))
  }

  // 从"跳过测评直接生成路径"或画像页过来时自动开始生成；用 ref 防止严格模式重复触发
  useEffect(() => {
    const wanted = (location.state as { generate?: boolean } | null)?.generate === true
    if (!wanted || autoStarted.current || path !== null) return
    autoStarted.current = true
    generate()
    // generate 每次渲染都会变，这里只关心"是否要求自动生成"
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state, path])

  if (data === null) {
    return error !== null ? <Banner onRetry={reload}>{error}</Banner> : <Spinner label="正在加载路径…" />
  }

  if (path === null) {
    const blocked = state.session.state === 'draft' || state.session.state === 'assessing'
    return (
      <div className="stack">
        {problem ? <Banner>{problem}</Banner> : null}
        {stream.httpError ? <Banner>{stream.httpError}</Banner> : null}
        <div className="card">
          <h2>还没有学习路径</h2>
          {blocked ? (
            <p>需要先完成知识基础确认（或点击「跳过测评」）才能生成路径。</p>
          ) : (
            <p>会根据你的掌握度画像排出 3~15 个节点，并为每个节点准备一个检查点问题。</p>
          )}
          <div className="btn-row">
            <button type="button" className="btn btn-primary" disabled={generating || blocked} onClick={generate}>
              {generating ? '正在生成…' : '生成学习路径'}
            </button>
            {generating ? (
              <button type="button" className="btn" onClick={stream.stop}>
                停止生成
              </button>
            ) : null}
            <button type="button" className="btn" onClick={() => navigate('../assessment', { relative: 'path' })}>
              回到知识基础确认
            </button>
          </div>
        </div>
      </div>
    )
  }

  const profile = data.profile ?? []
  const versions = data.versions ?? []

  return (
    <div className="stack">
      {problem ? <Banner>{problem}</Banner> : null}
      {stream.httpError ? <Banner>{stream.httpError}</Banner> : null}
      {teaching ? (
        <Banner tone="notice">
          学习已经开始。改标题与要点属于纠错，随时可做；增删或调整顺序只对尚未开始的节点生效。
        </Banner>
      ) : null}

      {profile.length > 0 ? (
        <div className="card">
          <h2>掌握度画像回顾</h2>
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: '30%' }}>子领域</th>
                <th style={{ width: '20%' }}>水平</th>
                <th>判定依据</th>
              </tr>
            </thead>
            <tbody>
              {profile.map((level) => (
                <tr key={level.topic}>
                  <td>{level.topic}</td>
                  <td>{LEVEL_LABELS[level.level] ?? level.level}</td>
                  <td className="small">{level.evidence || <span className="muted">（无）</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="card">
        <div className="row-between">
          <h2>
            学习路径 <span className="muted small">v{path.version}</span>
          </h2>
          <span className="muted small">
            共 {path.nodes.length} 个节点 · 当前第 {data.current_index} 个
            {versions.length > 1 ? ` · 历史版本 ${versions.length} 个` : ''}
          </span>
        </div>
        {path.rationale ? <Markdown text={path.rationale} /> : null}
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-sm"
            disabled={generating}
            onClick={() => {
              // 整条重生成会丢掉所有手动编辑（FR-305），必须二次确认
              if (!window.confirm('整条重新生成会丢失当前的手动编辑，旧版本会被保留。确定继续吗？')) return
              generate()
            }}
          >
            {generating ? '正在重新生成…' : '整条重新生成'}
          </button>
          {generating ? (
            <button type="button" className="btn btn-sm" onClick={stream.stop}>
              停止生成
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-sm"
            disabled={generating || busy}
            onClick={() => void edit(() => subjectsApi.addNode(subjectId))}
          >
            末尾加一个节点
          </button>
        </div>
        {generating ? <p className="muted small">正在根据画像重排节点…</p> : null}
      </div>

      <ol className="node-list">
        {path.nodes.map((node, index) => (
          <li key={node.id} className={`node-item${node.id === state.node?.id ? ' current' : ''}`}>
            {editingId === node.id ? (
              <NodeEditor
                node={node}
                busy={busy || regeneratingId === node.id}
                onCancel={() => setEditingId(null)}
                onSave={(fields) => {
                  void edit(() => subjectsApi.updateNode(subjectId, node.id, fields)).then(() => setEditingId(null))
                }}
              />
            ) : (
              <>
                <div className="node-item-head">
                  <span className="node-title">
                    {node.index}. {node.title}
                    {node.is_quick_review ? <span className="muted small">（快速复习）</span> : null}
                  </span>
                  <NodeStateBadge state={node.state} />
                </div>
                <p className="small muted">目标：{node.objective}</p>
                {node.key_points.length > 0 ? (
                  <ul className="small">
                    {node.key_points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                ) : null}
                <p className="small muted">
                  检查点 {node.checkpoints.length} 个
                  {node.prerequisites.length > 0 ? ` · 前置：${node.prerequisites.join('、')}` : ''}
                </p>
                {node.summary_note ? <p className="small">小结：{node.summary_note}</p> : null}

                <div className="btn-row">
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy || generating}
                    onClick={() => setEditingId(node.id)}
                  >
                    编辑
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy || generating || !canRestructure(node)}
                    title="让 AI 重新设计这一个节点；它的作答记录会保留"
                    onClick={() => {
                      const requirement = window.prompt('希望这个节点怎么调整？（可留空）', '') ?? ''
                      regenerateNode(node.id, requirement)
                    }}
                  >
                    {regeneratingId === node.id ? '重生成中…' : '重生成此节点'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy || generating || index === 0 || !canRestructure(node)}
                    onClick={() => void edit(() => subjectsApi.moveNode(subjectId, node.id, -1))}
                  >
                    上移
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy || generating || index === path.nodes.length - 1 || !canRestructure(node)}
                    onClick={() => void edit(() => subjectsApi.moveNode(subjectId, node.id, 1))}
                  >
                    下移
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy || generating || !canRestructure(node)}
                    onClick={() => void edit(() => subjectsApi.addNode(subjectId, node.id))}
                  >
                    在其后插入
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    disabled={busy || generating || !canRestructure(node) || path.nodes.length <= 1}
                    onClick={() => {
                      if (window.confirm(`确定删除「${node.title}」吗？`)) {
                        void edit(() => subjectsApi.deleteNode(subjectId, node.id))
                      }
                    }}
                  >
                    删除
                  </button>
                  {!canRestructure(node) ? <span className="muted small">已开始学习，不可增删移动</span> : null}
                </div>
              </>
            )}
          </li>
        ))}
      </ol>

      <div className="btn-row">
        <button
          type="button"
          className="btn btn-primary"
          disabled={confirming}
          onClick={() => {
            setConfirming(true)
            setProblem(null)
            void subjectsApi
              .confirmPath(subjectId)
              .then(() => {
                reloadState()
                navigate('../teach', { relative: 'path' })
              })
              .catch((cause: unknown) => setProblem(errorMessage(cause)))
              .finally(() => setConfirming(false))
          }}
        >
          {teaching ? '已在学习中，进入教学页' : '确认路径并开始学习'}
        </button>
        <span className="muted small">确认后即可开始逐节点学习；未确认前不会进入教学。</span>
      </div>
    </div>
  )
}
