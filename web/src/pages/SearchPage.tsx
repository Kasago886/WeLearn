import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { searchApi } from '../api/endpoints'
import type { SearchResult } from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { VerdictBadge } from '../components/StatusBadge'

const KIND_LABELS: Record<string, string> = {
  assess: '知识基础确认',
  teach: '教学对话',
  review: '复习对话',
}

/**
 * 全局检索（FR-803）：笔记、作答、对话三处一起搜。
 *
 * 输入即搜（300ms 防抖）而不是按回车：这三类数据量在本机都很小，
 * 让用户多按一次回车没有换来任何好处。
 */
export function SearchPage() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<SearchResult | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const text = query.trim()
    if (!text) return
    let alive = true
    // 300ms 防抖放在 effect 里；setLoading 也放进回调，避免"改一个字就同步 setState"的级联渲染
    const timer = window.setTimeout(() => {
      if (!alive) return
      setLoading(true)
      void searchApi
        .search(text)
        .then((body) => {
          if (!alive) return
          setResult(body)
          setProblem(null)
        })
        .catch((cause: unknown) => {
          if (alive) setProblem(errorMessage(cause))
        })
        .finally(() => {
          if (alive) setLoading(false)
        })
    }, 300)
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [query])

  const activeQuery = query.trim()
  // 只展示与当前关键词匹配的结果：请求在途时旧结果留在屏幕上会让人以为搜错了
  const shown = result !== null && result.query === activeQuery ? result : null
  const showLoading = activeQuery !== '' && loading && shown === null

  return (
    <div className="stack" style={{ maxWidth: 980 }}>
      <div className="page-head">
        <div>
          <h1>全局检索</h1>
          <p className="muted">在节点笔记、作答记录与历史对话里找关键词。</p>
        </div>
      </div>

      <div className="card">
        <div className="field">
          <label htmlFor="search">关键词</label>
          <input
            id="search"
            autoFocus
            value={query}
            placeholder="例如：先验、分母、归一化"
            onChange={(event) => setQuery(event.target.value)}
          />
          <span className="hint">大小写不敏感；空关键词不会发起请求。</span>
        </div>
      </div>

      {problem ? <Banner>{problem}</Banner> : null}
      {showLoading ? <Spinner label="正在检索…" /> : null}

      {shown !== null && !showLoading ? (
        shown.total === 0 ? (
          <div className="card empty">
            <h2>没有找到「{shown.query}」</h2>
            <p>换个更短的关键词试试；笔记与对话里的原文才会被命中。</p>
          </div>
        ) : (
          <>
            <p className="muted small">共 {shown.total} 条结果</p>

            {shown.notes.length > 0 ? (
              <section className="card stack">
                <h2>笔记（{shown.notes.length}）</h2>
                {shown.notes.map((hit) => (
                  <article className="history-item" key={`${hit.node_id}-${hit.updated_at}`}>
                    <div className="row-between">
                      <Link to={`/subjects/${hit.subject_id}/archive`}>{hit.subject_title}</Link>
                      <span className="muted small">{hit.node_title}</span>
                    </div>
                    <p className="small">{hit.snippet}</p>
                  </article>
                ))}
              </section>
            ) : null}

            {shown.answers.length > 0 ? (
              <section className="card stack">
                <h2>作答记录（{shown.answers.length}）</h2>
                {shown.answers.map((hit, index) => (
                  <article className="history-item" key={`${hit.node_id}-${hit.created_at}-${index}`}>
                    <div className="row-between">
                      <span>
                        <Link to={`/subjects/${hit.subject_id}/archive`}>{hit.subject_title}</Link>
                        <span className="muted small"> · {hit.node_title}</span>
                      </span>
                      <span className="btn-row">
                        <span className="badge badge-node">{hit.source === 'review' ? '复习' : '教学'}</span>
                        <VerdictBadge verdict={hit.verdict} />
                      </span>
                    </div>
                    <p className="small">{hit.snippet}</p>
                  </article>
                ))}
              </section>
            ) : null}

            {shown.messages.length > 0 ? (
              <section className="card stack">
                <h2>历史对话（{shown.messages.length}）</h2>
                {shown.messages.map((hit, index) => (
                  <article className="history-item" key={`${hit.subject_id}-${hit.created_at}-${index}`}>
                    <div className="row-between">
                      <Link to={`/subjects/${hit.subject_id}/archive`}>{hit.subject_title}</Link>
                      <span className="muted small">
                        {KIND_LABELS[hit.conversation_kind] ?? hit.conversation_kind} ·{' '}
                        {hit.role === 'user' ? '我' : 'AI'}
                      </span>
                    </div>
                    <p className="small">{hit.snippet}</p>
                  </article>
                ))}
              </section>
            ) : null}
          </>
        )
      ) : null}
    </div>
  )
}
