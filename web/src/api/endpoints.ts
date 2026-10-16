/**
 * 接口路径与调用的集中处。
 *
 * 页面里不直接拼 URL：路径一旦改动只需要改这一个文件，也便于对照 docs/接口契约.md §6。
 */

import { api } from './client'
import type {
  ArchiveView,
  AssessmentView,
  CompletionReport,
  Health,
  LearningPath,
  ModelInfo,
  NodeHistory,
  PathView,
  Progress,
  ReviewOverview,
  ReviewTurn,
  Settings,
  StudyNote,
  Subject,
  SubjectListItem,
  SubjectStateView,
  TeachingView,
  TestConnectionResult,
  UsageSummary,
  WeakPointWithNode,
} from './types'

export interface SubjectCreatePayload {
  title: string
  goal?: string | null
  background?: string | null
  depth?: string | null
}

export const subjectsApi = {
  list: (includeArchived: boolean) =>
    api.get<{ items: SubjectListItem[] }>(`/api/subjects?include_archived=${includeArchived ? 'true' : 'false'}`),
  create: (payload: SubjectCreatePayload) =>
    api.post<{ subject: Subject; duplicate_title_hint: boolean }>('/api/subjects', payload),
  get: (id: string) => api.get<{ subject: Subject; progress: Progress }>(`/api/subjects/${id}`),
  update: (id: string, payload: Partial<SubjectCreatePayload>) =>
    api.patch<{ subject: Subject }>(`/api/subjects/${id}`, payload),
  archive: (id: string, archived: boolean) =>
    api.post<{ subject: Subject }>(`/api/subjects/${id}/archive`, { archived }),
  remove: (id: string) => api.del<void>(`/api/subjects/${id}`),
  state: (id: string) => api.get<SubjectStateView>(`/api/subjects/${id}/state`),
  weakPoints: (id: string) => api.get<{ items: WeakPointWithNode[] }>(`/api/subjects/${id}/weak-points`),
  review: (id: string) => api.get<ReviewOverview>(`/api/subjects/${id}/review`),
  startReview: (id: string, nodeId: string) =>
    api.post<ReviewTurn>(`/api/subjects/${id}/review/start`, { node_id: nodeId }),
  answerReview: (id: string, question: string) =>
    api.post<ReviewTurn>(`/api/subjects/${id}/review/answer`, { question }),
  nodeHistory: (id: string, nodeId: string) =>
    api.get<NodeHistory>(`/api/subjects/${id}/nodes/${nodeId}/history`),
  saveNote: (id: string, nodeId: string, content: string) =>
    api.put<{ note: StudyNote }>(`/api/subjects/${id}/nodes/${nodeId}/note`, { content }),
  archiveOverview: (id: string) => api.get<ArchiveView>(`/api/subjects/${id}/archive`),
  completionReport: (id: string) => api.get<CompletionReport>(`/api/subjects/${id}/report`),
  assessment: (id: string) => api.get<AssessmentView>(`/api/subjects/${id}/assessment`),
  path: (id: string) => api.get<PathView>(`/api/subjects/${id}/path`),
  confirmPath: (id: string) => api.post<{ state: string }>(`/api/subjects/${id}/path/confirm`),
  teaching: (id: string) => api.get<TeachingView>(`/api/subjects/${id}/teach/current`),
  pause: (id: string) => api.post<{ state: string }>(`/api/subjects/${id}/teach/pause`),
  resume: (id: string) => api.post<{ state: string }>(`/api/subjects/${id}/teach/resume`),

  // 路径编辑（FR-304、FR-305）
  updateNode: (id: string, nodeId: string, payload: Partial<NodeEditableFields>) =>
    api.patch<{ path: LearningPath }>(`/api/subjects/${id}/path/nodes/${nodeId}`, payload),
  addNode: (id: string, afterNodeId?: string | null) =>
    api.post<{ path: LearningPath }>(`/api/subjects/${id}/path/nodes`, {
      after_node_id: afterNodeId ?? null,
    }),
  deleteNode: (id: string, nodeId: string) =>
    api.del<{ path: LearningPath }>(`/api/subjects/${id}/path/nodes/${nodeId}`),
  moveNode: (id: string, nodeId: string, offset: -1 | 1) =>
    api.post<{ path: LearningPath }>(`/api/subjects/${id}/path/nodes/${nodeId}/move`, { offset }),
}

export interface NodeEditableFields {
  title: string
  objective: string
  key_points: string[]
  prerequisites: string[]
}

/** 流式端点（POST + SSE）：只提供路径，实际请求由 useTurnStream 发起。 */
export const streamPaths = {
  assessmentStart: (id: string) => `/api/subjects/${id}/assessment/start`,
  assessmentAnswer: (id: string) => `/api/subjects/${id}/assessment/answer`,
  pathGenerate: (id: string) => `/api/subjects/${id}/path/generate`,
  teachStart: (id: string) => `/api/subjects/${id}/teach/start`,
  teachAnswer: (id: string) => `/api/subjects/${id}/teach/answer`,
  teachDecision: (id: string) => `/api/subjects/${id}/teach/decision`,
  teachSkip: (id: string) => `/api/subjects/${id}/teach/skip`,
  teachHint: (id: string) => `/api/subjects/${id}/teach/hint`,
  teachAsk: (id: string) => `/api/subjects/${id}/teach/ask`,
  teachRegenerate: (id: string) => `/api/subjects/${id}/teach/regenerate`,
  pathRegenerateNode: (id: string, nodeId: string) =>
    `/api/subjects/${id}/path/nodes/${nodeId}/regenerate`,
  pathRegenerate: (id: string) => `/api/subjects/${id}/path/regenerate`,
}

export const assessmentApi = {
  skip: (id: string) => api.post<{ skipped: boolean }>(`/api/subjects/${id}/assessment/skip`),
  overrideLevel: (id: string, topic: string, level: string) =>
    api.post<{ regenerate_path: boolean }>(`/api/subjects/${id}/assessment/level`, { topic, level }),
}

export const settingsApi = {
  get: () => api.get<{ settings: Settings }>('/api/settings'),
  save: (payload: Record<string, unknown>) => api.put<{ settings: Settings }>('/api/settings', payload),
  saveApiKey: (apiKey: string) =>
    api.put<{ configured: boolean; source: string; masked: string | null }>('/api/settings/api-key', {
      api_key: apiKey,
    }),
  testConnection: (model?: string) =>
    api.post<TestConnectionResult>('/api/settings/test-connection', model ? { model } : {}),
  models: () => api.get<{ ok: boolean; message: string; items: ModelInfo[] }>('/api/models'),
  usage: () => api.get<UsageSummary>('/api/usage'),
  health: () => api.get<Health>('/api/health'),
}
