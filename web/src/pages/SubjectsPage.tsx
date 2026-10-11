import { useState } from 'react'
import { Link } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { subjectsApi } from '../api/endpoints'
import type { SubjectListItem } from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { SubjectStateBadge } from '../components/StatusBadge'
import { useResource } from '../hooks/useResource'

function formatTime(value: string | null): string {
  if (!value) {
    return '尚未学习'
  }
  const date = new Date(value.endsWith('Z') ? value : `${value}Z`)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString('zh-CN', { hour12: false })
}

function SubjectCard({ item, onChange }: { item: SubjectListItem; onChange: () => void }) {
  const [renaming, setRenaming] = useState(false)
  const [title, setTitle] = useState(item.title)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mastered = `${item.progress.mastered_nodes}/${item.progress.total_nodes}`

  async function act(action: () => Promise<unknown>) {
    setBusy(true)
    setError(null)
    try {
      await action()
      onChange()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className="card">
      <div className="row-between">
        {renaming ? (
          <form
            className="row-between"
            onSubmit={(event) => {
              event.preventDefault()
              void act(async () => {
                await subjectsApi.update(item.id, { title })
                setRenaming(false)
              })
            }}
          >
            <input value={title} onChange={(event) => setTitle(event.target.value)} aria-label="新的概念名称" />
            <div className="btn-row">
              <button type="submit" className="btn btn-sm btn-primary" disabled={busy}>
                保存
              </button>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => {
                  setRenaming(false)
                  setTitle(item.title)
                }}
              >
                取消
              </button>
            </div>
          </form>
        ) : (
          <Link to={`/subjects/${item.id}`} className="node-title">
            {item.title}
          </Link>
        )}
        <SubjectStateBadge state={item.status} />
      </div>

      <p className="muted small">
        进度 已掌握 {mastered} · 薄弱点 {item.progress.weak_points} · 最近学习 {formatTime(item.last_studied_at)}
      </p>
      {item.goal ? <p className="small">目标：{item.goal}</p> : null}
      {item.duplicate_title_hint ? <p className="small muted">已有同名主题（同名不冲突，可继续）</p> : null}

      {item.progress.total_nodes > 0 ? (
        <div className="progress-bar" aria-label={`掌握度 ${Math.round(item.progress.mastery_ratio * 100)}%`}>
          <span style={{ width: `${Math.round(item.progress.mastery_ratio * 100)}%` }} />
        </div>
      ) : null}

      {error ? <Banner>{error}</Banner> : null}

      <div className="btn-row">
        <Link className="btn btn-sm btn-primary" to={`/subjects/${item.id}`}>
          进入学习
        </Link>
        <button type="button" className="btn btn-sm" disabled={busy} onClick={() => setRenaming(true)}>
          重命名
        </button>
        <button
          type="button"
          className="btn btn-sm"
          disabled={busy}
          onClick={() => void act(() => subjectsApi.archive(item.id, !item.archived))}
        >
          {item.archived ? '取消归档' : '归档'}
        </button>
        <button
          type="button"
          className="btn btn-sm btn-danger"
          disabled={busy}
          onClick={() => {
            // 删除会连带清掉测评、路径、作答记录，必须二次确认（FR-104）
            if (window.confirm(`确定删除「${item.title}」吗？该主题的测评、路径与作答记录都会一并删除。`)) {
              void act(() => subjectsApi.remove(item.id))
            }
          }}
        >
          删除
        </button>
      </div>
    </article>
  )
}

export function SubjectsPage() {
  const [includeArchived, setIncludeArchived] = useState(false)
  const { data, error, loading, reload } = useResource(
    () => subjectsApi.list(includeArchived),
    [includeArchived],
  )
  const items: SubjectListItem[] = data?.items ?? []

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>学习主题</h1>
          <p className="muted">说出你想学的概念，AI 先摸清你的基础，再为你排出路径并逐节点讲透。</p>
        </div>
        <div className="btn-row">
          <label className="btn btn-sm" style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
            <input
              type="checkbox"
              style={{ width: 'auto' }}
              checked={includeArchived}
              onChange={(event) => setIncludeArchived(event.target.checked)}
            />
            显示已归档
          </label>
          <Link className="btn btn-primary" to="/new">
            新建学习主题
          </Link>
        </div>
      </div>

      {error ? (
        <Banner onRetry={reload}>{error}</Banner>
      ) : loading ? (
        <div className="grid-cards">
          <div className="skeleton" />
          <div className="skeleton" />
        </div>
      ) : items.length === 0 ? (
        <div className="card empty">
          <h2>还没有学习主题</h2>
          <p>从一个你一直想弄明白的概念开始，比如「贝叶斯定理」「TCP 拥塞控制」。</p>
          <Link className="btn btn-primary" to="/new">
            开始第一个学习主题
          </Link>
        </div>
      ) : (
        <div className="grid-cards">
          {items.map((item) => (
            <SubjectCard key={item.id} item={item} onChange={reload} />
          ))}
        </div>
      )}

      {loading && items.length > 0 ? <Spinner label="正在刷新…" /> : null}
    </div>
  )
}
