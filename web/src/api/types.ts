/**
 * 后端响应类型（对应 docs/接口契约.md v1.1）。
 *
 * 刻意手写而不是从 OpenAPI 生成：契约本身已经冻结成文档，手写能让字段名与契约逐条对照。
 */

export type SubjectState = 'draft' | 'assessing' | 'path_review' | 'teaching' | 'paused' | 'completed'

export type NodeState = 'not_started' | 'learning' | 'mastered' | 'partial' | 'skipped' | 'review_needed'

export type Verdict = 'pass' | 'partial' | 'fail'

export type MasteryLevel = 'none' | 'basic' | 'proficient'

export type NodeDecision = 'reteach' | 'show_answer' | 'advance'

/** 下一步该做什么；前端据此决定展示什么（FR-410）。 */
export type NextAction = 'assess' | 'confirm_path' | 'teach' | 'answer' | 'decide' | 'completed' | 'paused'

export interface Subject {
  id: string
  title: string
  goal: string | null
  background: string | null
  depth: string | null
  status: SubjectState
  archived: boolean
  created_at: string | null
  updated_at: string | null
  archived_at: string | null
}

export interface Progress {
  total_nodes: number
  mastered_nodes: number
  partial_nodes: number
  skipped_nodes: number
  weak_points: number
  total_tokens: number
  mastery_ratio: number
}

export interface SubjectListItem extends Subject {
  progress: Progress
  last_studied_at: string | null
  duplicate_title_hint: boolean
}

export interface Checkpoint {
  id: string
  order_index: number
  question: string
  expected_points: string[]
  reference_answer: string
}

export interface KnowledgeNode {
  id: string
  /** 从 1 开始的位置（与 node_state 事件同一套语义）。 */
  index: number
  title: string
  objective: string
  prerequisites: string[]
  key_points: string[]
  state: NodeState
  attempts: number
  is_quick_review: boolean
  summary_note: string | null
  mastered_at: string | null
  checkpoints: Checkpoint[]
}

export interface LearningPath {
  id: string
  version: number
  rationale: string
  model: string
  is_active: boolean
  created_at: string | null
  nodes: KnowledgeNode[]
}

export interface SessionSnapshot {
  state: SubjectState
  next_action: NextAction
  current_node_id: string | null
  pending_checkpoint_id: string | null
  pending_question: string | null
  node_attempts: number
  awaiting_decision: boolean
  node_taught: boolean
  updated_at: string | null
}

export interface ProfileLevel {
  topic: string
  level: MasteryLevel
  evidence: string
  user_overridden: boolean
}

export interface Assessment {
  id: string
  status: 'running' | 'completed' | 'skipped'
  turn_count: number
  max_turns: number
  turns_left: number
  topics: string[]
  summary: string
  suggested_start_node: string
  skipped: boolean
  completed_at: string | null
  levels: ProfileLevel[]
}

export interface AnswerRecord {
  id: number
  node_id: string
  question: string
  answer: string
  verdict: Verdict
  confidence: number | null
  feedback: string
  hit_points: string[]
  missing_points: string[]
  attempt_index: number
  created_at: string | null
}

export interface WeakPoint {
  id: number
  node_id: string
  reason: 'partial' | 'skipped' | 'show_answer' | 'advanced'
  fail_count: number
  last_attempt_at: string | null
  resolved: boolean
}

/** `GET /subjects/{id}/state`：刷新后唯一的恢复入口。 */
export interface SubjectStateView {
  session: SessionSnapshot
  subject: Subject
  node: KnowledgeNode | null
  path: LearningPath | null
  can_answer: boolean
  needs_decision: boolean
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AssessmentView {
  assessment: Assessment | null
  transcript: ChatMessage[]
  pending_question: string | null
}

export interface PathView {
  path: LearningPath | null
  profile: ProfileLevel[]
  current_index: number
  completed: boolean
  versions: { path_id: string; version: number; is_active: boolean }[]
}

export interface TeachingView {
  session: SessionSnapshot
  node: KnowledgeNode | null
  checkpoint: { id: string; question: string; order_index: number } | null
  node_index: number
  node_total: number
  can_answer: boolean
  needs_decision: boolean
  answers: AnswerRecord[]
}

export interface Settings {
  base_url: string
  model_default: string
  model_pro: string
  thinking_enabled: boolean
  reasoning_effort: 'low' | 'high' | 'max'
  max_tokens: number
  timeout_seconds: number
  max_retries: number
  assess_max_turns: number
  path_min_nodes: number
  path_max_nodes: number
  node_max_retries: number
  compress_threshold_rounds: number
  compress_round_count: number
  show_reasoning: boolean
  log_level: string
  data_dir: string
  read_only_fields: string[]
  api_key: { configured: boolean; source: string; masked: string | null }
}

export interface ModelInfo {
  id: string
  name: string
  context_window: number
  max_output_tokens: number
  effort_levels: string[]
  default_effort: string
}

export interface UsageSummary {
  calls: number
  prompt_tokens: number
  cached_prompt_tokens: number
  cache_miss_tokens: number
  completion_tokens: number
  reasoning_tokens: number
  total_tokens: number
  cache_hit_rate: number
}

export interface Health {
  status: 'ok' | 'degraded'
  problems: string[]
  storage: { ok: boolean; detail: string; data_dir: string; schema_version: number }
  model: {
    base_url: string
    default: string
    pro: string
    thinking_enabled: boolean
    reasoning_effort: string
    max_tokens: number
    timeout_seconds: number
  }
  api_key: { configured: boolean; source: string; masked: string | null }
  logging: { level: string; log_dir: string }
}

export interface TestConnectionResult {
  ok: boolean
  message: string
  category: string | null
  latency_ms: number | null
  model: string
}

// --------------------------------------------------------------------------- //
// SSE 事件（契约 §1）
// --------------------------------------------------------------------------- //

export type SseEventType =
  | 'token'
  | 'reasoning'
  | 'verdict'
  | 'node_state'
  | 'assessment'
  | 'path'
  | 'options'
  | 'error'
  | 'done'

export interface ErrorPayload {
  category: string
  message: string
  retryable: boolean
}

export interface VerdictPayload {
  verdict: Verdict
  confidence: number
  feedback: string
  hit_points: string[]
  missing_points: string[]
}

export interface NodeStatePayload {
  node_id: string
  state: NodeState
  index: number
  total: number
}

export interface AssessmentEventPayload {
  status: 'question' | 'profile' | 'skipped'
  topics: string[]
  turn_number: number
  total_turns: number
  current_topic: string | null
  profile: { levels: ProfileLevel[]; summary: string; suggested_start_node: string } | null
}

export interface PathPayload {
  path_id: string
  version: number
  rationale: string
  nodes: {
    id: string
    index: number
    title: string
    objective: string
    key_points: string[]
    state: NodeState
    is_quick_review: boolean
    checkpoints: { id: string; question: string }[]
  }[]
}

export interface OptionsPayload {
  kind: 'exhausted'
  choices: NodeDecision[]
  message: string
  directions: string[]
}

export interface SseEvent {
  type: SseEventType
  data: Record<string, unknown>
}
