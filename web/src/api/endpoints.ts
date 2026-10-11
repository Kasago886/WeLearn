/**
 * 接口路径与调用的集中处。
 *
 * 页面里不直接拼 URL：路径一旦改动只需要改这一个文件，也便于对照 docs/接口契约.md §6。
 */

import { api } from './client'
import type {
  AssessmentView,
  Health,
  ModelInfo,
  PathView,
  Progress,
  Settings,
  Subject,
  SubjectListItem,
  SubjectStateView,
  TeachingView,
  TestConnectionResult,
  UsageSummary,
  WeakPoint,
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
  weakPoints: (id: string) => api.get<{ items: WeakPoint[] }>(`/api/subjects/${id}/weak-points`),
  assessment: (id: string) => api.get<AssessmentView>(`/api/subjects/${id}/assessment`),
  path: (id: string) => api.get<PathView>(`/api/subjects/${id}/path`),
  confirmPath: (id: string) => api.post<{ state: string }>(`/api/subjects/${id}/path/confirm`),
  teaching: (id: string) => api.get<TeachingView>(`/api/subjects/${id}/teach/current`),
  pause: (id: string) => api.post<{ state: string }>(`/api/subjects/${id}/teach/pause`),
  resume: (id: string) => api.post<{ state: string }>(`/api/subjects/${id}/teach/resume`),
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
