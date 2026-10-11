import { createContext, useContext } from 'react'

import type { SubjectStateView } from '../api/types'

export interface SubjectContextValue {
  subjectId: string
  state: SubjectStateView
  /** 任何会改变状态机的操作完成后都要调用它，保证页面与后端一致。 */
  reloadState: () => void
}

export const SubjectContext = createContext<SubjectContextValue | null>(null)

export function useSubject(): SubjectContextValue {
  const value = useContext(SubjectContext)
  if (value === null) {
    throw new Error('useSubject 必须在主题页面内使用')
  }
  return value
}
