import { NavLink, Outlet, useParams } from 'react-router-dom'

import { subjectsApi } from '../api/endpoints'
import { Banner, Spinner } from '../components/Banner'
import { SubjectStateBadge } from '../components/StatusBadge'
import { useResource } from '../hooks/useResource'
import { SubjectContext } from './subjectContext'

const STAGES = [
  { path: 'assessment', label: '知识基础确认' },
  { path: 'path', label: '学习路径' },
  { path: 'teach', label: '教学' },
]

/**
 * 主题工作区外壳：加载状态机快照，给各阶段页提供统一的标题、阶段导航与数据。
 *
 * 状态只在这里加载一次并向下传递，避免各阶段页各自请求"当前在哪一步"而给出不一致的结果。
 */
export function SubjectLayout() {
  const { subjectId = '' } = useParams()
  const { data, error, loading, reload } = useResource(() => subjectsApi.state(subjectId), [subjectId])

  // 刷新时**不能**把界面换回骨架屏：那会卸载整个 Outlet 子树，
  // 让正在流式输出的讲解、刚拿到的判定卡片全部丢失（每次回合结束都会刷新一次状态）。
  // 因此只有"从未加载成功过"时才显示加载/错误占位，已有数据就继续展示。
  if (data === null) {
    if (error !== null || loading) {
      return error !== null ? (
        <div className="stack">
          <Banner onRetry={reload}>{error}</Banner>
          <NavLink className="btn" to="/">
            返回主题列表
          </NavLink>
        </div>
      ) : (
        <Spinner label="正在加载主题…" />
      )
    }
    return <Spinner label="正在加载主题…" />
  }

  const { subject, session } = data

  return (
    <SubjectContext.Provider value={{ subjectId, state: data, reloadState: reload }}>
      <div className="stack">
        {error ? <Banner onRetry={reload}>{error}</Banner> : null}
        <div className="page-head">
          <div>
            <h1>
              {subject.title} <SubjectStateBadge state={subject.status} />
            </h1>
            <p className="muted small">
              {subject.goal ? `目标：${subject.goal}` : '未填写学习目标'}
              {subject.depth ? ` · 期望深度：${subject.depth}` : ''}
            </p>
          </div>
          <div className="btn-row">
            <NavLink className="btn btn-sm" to="/">
              返回列表
            </NavLink>
          </div>
        </div>

        <nav className="app-nav" aria-label="学习阶段">
          {STAGES.map((stage) => (
            <NavLink
              key={stage.path}
              to={stage.path}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
            >
              {stage.label}
            </NavLink>
          ))}
        </nav>

        <Outlet context={{ subjectId, state: data, reloadState: reload }} />

        <p className="muted small">
          当前状态：{session.state} · 下一步：{session.next_action}
          {session.pending_question ? ' · 有一个待回答的问题' : ''}
        </p>
      </div>
    </SubjectContext.Provider>
  )
}
