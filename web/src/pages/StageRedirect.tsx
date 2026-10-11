import { Navigate } from 'react-router-dom'

import { useSubject } from './subjectContext'

/**
 * 刷新/直接访问主题根路径时的分流（FR-410）。
 *
 * 跳到哪里完全由后端的 `next_action` 决定，前端不自己推断，
 * 否则前端与后端的状态判断一旦分叉，就会出现"页面显示的阶段与实际状态不符"。
 */
export function StageRedirect() {
  const { state } = useSubject()

  switch (state.session.next_action) {
    case 'assess':
      return <Navigate to="assessment" replace />
    case 'confirm_path':
      return <Navigate to="path" replace />
    default:
      return <Navigate to="teach" replace />
  }
}
