import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { subjectsApi, streamPaths } from '../api/endpoints'
import type { SseEvent } from '../api/types'
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
 * 学习路径页（P5）：画像回顾 + 生成理由 + 节点清单 + 确认门禁。
 *
 * 「确认」是硬门禁（FR-306）：没确认就不能进入教学，因此这里既展示路径也承担放行职责。
 */
export function PathPage() {
  const { subjectId, state, reloadState } = useSubject()
  const location = useLocation()
  const navigate = useNavigate()
  const stream = useTurnStream()
  const [problem, setProblem] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)
  const autoStarted = useRef(false)

  const { data, error, loading, reload } = useResource(() => subjectsApi.path(subjectId), [subjectId])
  const path = data?.path ?? null
  const generating = stream.streaming
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

  // 从"跳过测评直接生成路径"或画像页过来时自动开始生成；用 ref 防止严格模式重复触发
  useEffect(() => {
    const wanted = (location.state as { generate?: boolean } | null)?.generate === true
    if (!wanted || autoStarted.current || path !== null || loading) return
    autoStarted.current = true
    generate()
    // generate 每次渲染都会变，这里只关心"是否要求自动生成"
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state, path, loading])

  // 刷新时保留现有界面（同上：换骨架屏会把刚生成的路径一起丢掉）
  if (data === null) {
    return error !== null ? <Banner onRetry={reload}>{error}</Banner> : <Spinner label="正在加载路径…" />
  }

  if (path === null) {
    const blocked = state.session.state === 'draft' || state.session.state === 'assessing'
    return (
      <div className="stack">
        {error ? <Banner onRetry={reload}>{error}</Banner> : null}
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

  const profile = data?.profile ?? []
  const versions = data?.versions ?? []

  return (
    <div className="stack">
      {error ? <Banner onRetry={reload}>{error}</Banner> : null}
      {problem ? <Banner>{problem}</Banner> : null}
      {stream.httpError ? <Banner>{stream.httpError}</Banner> : null}

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
            共 {path.nodes.length} 个节点 · 当前第 {data?.current_index ?? 1} 个
            {versions.length > 1 ? ` · 历史版本 ${versions.length} 个` : ''}
          </span>
        </div>
        {path.rationale ? <Markdown text={path.rationale} /> : null}
        <div className="btn-row">
          <button type="button" className="btn btn-sm" disabled={generating} onClick={generate}>
            {generating ? '正在重新生成…' : '重新生成路径'}
          </button>
          {generating ? (
            <button type="button" className="btn btn-sm" onClick={stream.stop}>
              停止生成
            </button>
          ) : null}
        </div>
        {generating ? <p className="muted small">正在根据画像重排节点…</p> : null}
      </div>

      <ol className="node-list">
        {path.nodes.map((node) => (
          <li
            key={node.id}
            className={`node-item${node.id === state.node?.id ? ' current' : ''}`}
          >
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
          {state.session.state === 'teaching' ? '已在学习中，进入教学页' : '确认路径并开始学习'}
        </button>
        <span className="muted small">确认后即可开始逐节点学习；未确认前不会进入教学。</span>
      </div>
    </div>
  )
}
