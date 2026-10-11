/** 状态文案与格式化工具（与组件分开放，避免一个文件既导出组件又导出普通函数）。 */

import type { NodeState, SubjectState, Verdict } from '../api/types'

const SUBJECT_LABELS: Record<SubjectState, string> = {
  draft: '待测评',
  assessing: '知识基础确认中',
  path_review: '待确认路径',
  teaching: '学习中',
  paused: '已暂停',
  completed: '已完成',
}

const NODE_LABELS: Record<NodeState, string> = {
  not_started: '未开始',
  learning: '学习中',
  mastered: '已掌握',
  partial: '部分掌握',
  skipped: '已跳过',
  review_needed: '待复习',
}

const VERDICT_LABELS: Record<Verdict, string> = {
  pass: '通过',
  partial: '部分通过',
  fail: '未通过',
}

export function subjectStateLabel(state: SubjectState): string {
  return SUBJECT_LABELS[state] ?? state
}

export function nodeStateLabel(state: NodeState): string {
  return NODE_LABELS[state] ?? state
}

export function verdictLabel(verdict: Verdict): string {
  return VERDICT_LABELS[verdict] ?? verdict
}

export function formatDateTime(value: string | null, fallback = '尚未学习'): string {
  if (!value) {
    return fallback
  }
  // 后端统一按 UTC 存储、不带时区后缀，展示时按本地时区解释
  const date = new Date(value.endsWith('Z') ? value : `${value}Z`)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString('zh-CN', { hour12: false })
}
