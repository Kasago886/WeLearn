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
  /** 已到期的复习节点数（FR-506）。 */
  due_reviews?: number
  /** 有掌握节点久未触达或有到期排期，列表上标「建议复习」（FR-505）。 */
  needs_review?: boolean
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
  source: 'teach' | 'review'
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

export interface ReviewItem extends WeakPointWithNode {}

export interface ReviewTurn {
  text: string
  phase: 'question' | 'passed' | 'failed'
  mode: 'weak' | 'adjacent'
  node: KnowledgeNode | null
  verdict: Verdict | null
  confidence: number
  feedback: string
  hit_points: string[]
  missing_points: string[]
  promoted: boolean
  resolved: boolean
  fail_count: number
  suggestion: string
  node_index: number
  node_total: number
}

export interface ReviewOverview {
  items: ReviewItem[]
  open_count: number
  node_total: number
  mastered_count: number
  current: ReviewTurn | null
  /** 到期的已掌握节点（FR-506 的"当日复习清单"）。 */
  due: DueReview[]
  due_count: number
}

/** 到期复习项：只带展示需要的信息，不把整条路径塞进列表。 */
export interface DueReview {
  node_id: string
  node_title: string
  node_index: number
  node_total: number
  due_at: string | null
  interval_days: number
}

export interface NodeHistory {
  node_id: string
  items: AnswerRecord[]
  teach_count: number
  review_count: number
}

/** 节点笔记（FR-801）：`auto` 是模型自动小结，`user` 是手写覆盖。 */
export interface StudyNote {
  id: string
  node_id: string
  content: string
  source: 'auto' | 'user'
  updated_at: string | null
}

/** 薄弱点条目带上所属节点信息，便于在档案/复习页直接展示位置。 */
export interface WeakPointWithNode extends WeakPoint {
  node_title: string
  node_index?: number
  node_total?: number
}

export interface ArchiveNode extends KnowledgeNode {
  answer_count: number
  review_count: number
  note: StudyNote | null
}

export interface ArchiveView {
  subject: Subject
  path_id: string | null
  path_version: number | null
  nodes: ArchiveNode[]
  progress: Progress
  weak_points: WeakPointWithNode[]
}

export interface ReportEntry {
  id: string
  index: number
  title: string
  /** 仅「部分掌握」会带：是哪些原因让它停在这里（FR-417）。 */
  reasons?: string[]
}

export interface CompletionReport {
  subject: Subject
  state: SubjectState
  mastered: ReportEntry[]
  partial: ReportEntry[]
  skipped: ReportEntry[]
  weak_points: WeakPointWithNode[]
  usage: {
    calls: number
    total_tokens: number
    reasoning_tokens: number
    cached_prompt_tokens: number
    cache_hit_rate: number
    /** 累计模型调用耗时（毫秒），不含用户阅读时间（FR-417）。 */
    duration_ms: number
    /** 成本估算（FR-706）。 */
    cost?: CostSummary
  }
  started_at: string | null
  last_activity_at: string | null
}

/** 全局检索的结果（FR-803）。三类来源分开返回，便于各自展示与跳转。 */
export interface SearchNoteHit {
  subject_id: string
  subject_title: string
  node_id: string
  node_title: string
  snippet: string
  updated_at: string | null
}

export interface SearchAnswerHit {
  subject_id: string
  subject_title: string
  node_id: string
  node_title: string
  verdict: Verdict
  snippet: string
  created_at: string | null
  source: 'teach' | 'review'
}

export interface SearchMessageHit {
  subject_id: string
  subject_title: string
  conversation_kind: 'assess' | 'teach' | 'review'
  role: string
  snippet: string
  created_at: string | null
}

export interface SearchResult {
  query: string
  total: number
  notes: SearchNoteHit[]
  answers: SearchAnswerHit[]
  messages: SearchMessageHit[]
}

/** 导出结果：前端用 Blob 下载，不依赖后端设 Content-Disposition。 */
export interface ExportPayload {
  filename: string
  content: string
  format: 'markdown' | 'json'
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
  /** 最近一次教学动作；为空表示没有可「重新生成」的回复（FR-413）。 */
  last_action: string | null
}

// --------------------------------------------------------------------------- //
// 按用途的模型策略（FR-703）
// --------------------------------------------------------------------------- //

/** 可单独配置的调用用途（与后端 `LlmPurpose` 一一对应）。 */
export type LlmPurposeKey = 'assess' | 'plan' | 'teach' | 'judge' | 'compress'

export type ModelPick = 'inherit' | 'default' | 'pro'
export type ThinkingPick = 'inherit' | 'on' | 'off'
export type EffortPick = 'inherit' | 'low' | 'high' | 'max'

/** 单个用途的覆盖项；每一项都可以"继承全局设置"。 */
export interface PurposeOverride {
  model: ModelPick
  thinking: ThinkingPick
  reasoning_effort: EffortPick
}

/** 解析后的实际参数（只读展示，告诉用户"这一项真的会用什么"）。 */
export interface ModelChoice {
  model: string
  thinking: boolean
  reasoning_effort: string
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
  /** 久未复习的提醒阈值（FR-505），单位天。 */
  review_interval_days: number
  /** 单个日志文件的轮转阈值（MB）与归档保留天数（FR-907）。 */
  log_max_mb: number
  log_retention_days: number
  /** 工具调用开关（FR-714）；开启后历史思维链会回传，上下文与费用都会上升。 */
  tools_enabled: boolean
  /** 生效价表（元 / 百万 token，高峰价）；保存时整体替换。 */
  price_table: Record<string, Record<string, number>>
  /** 是否已设置访问口令（FR-903）；口令本身永不回传。 */
  access_token_enabled: boolean
  /** 采样温度（FR-718）：思考模式下官方会忽略它。 */
  temperature: number
  /** 核采样（FR-718）：仅思考模式下生效，且被钳制到 0.95~1.0。 */
  top_p: number
  /** 请求审计开关（FR-715）：开启后记录完整请求与响应，默认关闭。 */
  audit_enabled: boolean
  use_recommended_overrides: boolean
  purpose_overrides: Partial<Record<LlmPurposeKey, PurposeOverride>>
  /** 每个用途解析后的实际参数；只读，提交时忽略。 */
  model_plan?: Record<LlmPurposeKey, ModelChoice>
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

/** 用量聚合的一格（全局、某用途或某主题）。 */
export interface UsageBucket {
  calls: number
  prompt_tokens: number
  cached_prompt_tokens: number
  completion_tokens: number
  reasoning_tokens: number
  total_tokens: number
  cache_hit_rate: number
  duration_ms: number
}

export interface SubjectUsageBucket {
  subject_id: string
  title: string
  calls: number
  total_tokens: number
  cache_hit_rate: number
  duration_ms: number
}

export interface UsageSummary extends UsageBucket {
  cache_miss_tokens: number
  /** 按用途的用量（FR-705）；后端未提供时前端按空表展示。 */
  by_purpose?: Record<string, UsageBucket>
  /** 按主题的用量（取 token 前若干名）。 */
  by_subject?: SubjectUsageBucket[]
  /** 命中率异常时的排查提示；正常时为 null（FR-705、FR-720）。 */
  cache_hint?: string | null
  /** 成本估算（FR-706）。 */
  cost?: CostSummary
}

/** 费用估算：金额单位由 currency 决定（默认人民币元）。 */
export interface CostSummary {
  amount: number
  currency: string
  peak_tokens: number
  off_peak_tokens: number
  total_tokens: number
}

export interface Balance {
  ok: boolean
  currency: string | null
  total_balance: string | null
  granted_balance: string | null
  topped_up_balance: string | null
  message: string
  category?: string | null
}

export interface LogStats {
  log_dir: string
  files: { name: string; size: number }[]
  total_bytes: number
  max_bytes: number
  retention_days: number
}

export interface AuthVerify {
  ok: boolean
  required: boolean
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
  | 'hint'
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

/** 一级提示（FR-407）。`label` 是后端给的中文名，前端直接展示。 */
export interface HintPayload {
  level: number
  label: string
  question: string
}

/** 提示级别（1 方向性 / 2 关键要素 / 3 近似答案）。 */
export const HINT_LEVELS = [1, 2, 3] as const
export type HintLevel = (typeof HINT_LEVELS)[number]

export interface SseEvent {
  type: SseEventType
  data: Record<string, unknown>
}
