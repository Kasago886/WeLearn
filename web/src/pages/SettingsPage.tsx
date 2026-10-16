import { useEffect, useState } from 'react'

import { errorMessage } from '../api/client'
import { settingsApi } from '../api/endpoints'
import type {
  EffortPick,
  LlmPurposeKey,
  ModelInfo,
  ModelPick,
  PurposeOverride,
  Settings,
  TestConnectionResult,
  ThinkingPick,
  UsageSummary,
} from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { useResource } from '../hooks/useResource'

interface NumberFieldProps {
  id: string
  label: string
  hint?: string
  value: number
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  onChange: (value: number) => void
}

function NumberField({ id, label, hint, value, min, max, step, disabled, onChange }: NumberFieldProps) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  )
}

const PURPOSE_LABELS: Record<LlmPurposeKey, string> = {
  assess: '知识基础确认',
  plan: '路径规划',
  teach: '教学讲解',
  judge: '作答判定',
  compress: '上下文压缩',
}

const PURPOSE_KEYS: LlmPurposeKey[] = ['assess', 'plan', 'teach', 'judge', 'compress']

const INHERIT_OVERRIDE: PurposeOverride = {
  model: 'inherit',
  thinking: 'inherit',
  reasoning_effort: 'inherit',
}

const MODEL_PICK_LABELS: Record<ModelPick, string> = {
  inherit: '继承全局',
  default: '默认模型',
  pro: '深度模型',
}

const THINKING_PICK_LABELS: Record<ThinkingPick, string> = {
  inherit: '继承全局',
  on: '开启',
  off: '关闭',
}

const EFFORT_PICK_LABELS: Record<EffortPick, string> = {
  inherit: '继承全局',
  low: 'low',
  high: 'high',
  max: 'max',
}

function formatDuration(durationMs: number): string {
  if (!durationMs || durationMs <= 0) return '—'
  const seconds = Math.round(durationMs / 1000)
  if (seconds < 60) return `${seconds} 秒`
  return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`
}

/**
 * 按用途的模型与思考参数（FR-703）。
 *
 * 右侧「当前会实际使用」是后端解析出的值，而不是界面上的选择——只显示选择的话，
 * 用户看到"深度模型"就以为生效了，实际可能被推荐组合或继承规则盖掉。
 */
function PurposeStrategyTable({
  draft,
  busy,
  onPatch,
}: {
  draft: Settings
  busy: boolean
  onPatch: (changes: Partial<Settings>) => void
}) {
  const overrides = draft.purpose_overrides ?? {}

  function patchPurpose(key: LlmPurposeKey, changes: Partial<PurposeOverride>): void {
    onPatch({
      purpose_overrides: {
        ...overrides,
        [key]: { ...(overrides[key] ?? INHERIT_OVERRIDE), ...changes },
      },
    })
  }

  return (
    <table className="table">
      <thead>
        <tr>
          <th style={{ width: '16%' }}>用途</th>
          <th>模型</th>
          <th>思考</th>
          <th>思考档位</th>
          <th style={{ width: '30%' }}>当前会实际使用</th>
        </tr>
      </thead>
      <tbody>
        {PURPOSE_KEYS.map((key) => {
          const override = overrides[key] ?? INHERIT_OVERRIDE
          const effective = draft.model_plan?.[key]
          return (
            <tr key={key}>
              <td>{PURPOSE_LABELS[key]}</td>
              <td>
                <select
                  aria-label={`${PURPOSE_LABELS[key]}的模型`}
                  value={override.model}
                  disabled={busy}
                  onChange={(event) => patchPurpose(key, { model: event.target.value as ModelPick })}
                >
                  {(Object.keys(MODEL_PICK_LABELS) as ModelPick[]).map((pick) => (
                    <option key={pick} value={pick}>
                      {MODEL_PICK_LABELS[pick]}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <select
                  aria-label={`${PURPOSE_LABELS[key]}的思考开关`}
                  value={override.thinking}
                  disabled={busy}
                  onChange={(event) => patchPurpose(key, { thinking: event.target.value as ThinkingPick })}
                >
                  {(Object.keys(THINKING_PICK_LABELS) as ThinkingPick[]).map((pick) => (
                    <option key={pick} value={pick}>
                      {THINKING_PICK_LABELS[pick]}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <select
                  aria-label={`${PURPOSE_LABELS[key]}的思考档位`}
                  value={override.reasoning_effort}
                  disabled={busy || override.thinking === 'off'}
                  onChange={(event) =>
                    patchPurpose(key, { reasoning_effort: event.target.value as EffortPick })
                  }
                >
                  {(Object.keys(EFFORT_PICK_LABELS) as EffortPick[]).map((pick) => (
                    <option key={pick} value={pick}>
                      {EFFORT_PICK_LABELS[pick]}
                    </option>
                  ))}
                </select>
              </td>
              <td className="small muted">
                {effective
                  ? `${effective.model} · ${effective.thinking ? '思考开' : '思考关'} · ${effective.reasoning_effort}`
                  : '保存后可见'}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/** 设置页（P8）：模型与 Key、超时重试、各流程参数、数据目录与用量。 */
export function SettingsPage() {
  const [draftOverride, setDraftOverride] = useState<Settings | null>(null)
  const [saved, setSaved] = useState<string | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const [apiKey, setApiKey] = useState('')
  const [keyMessage, setKeyMessage] = useState<string | null>(null)
  const [testModel, setTestModel] = useState('')
  const [testResult, setTestResult] = useState<TestConnectionResult | null>(null)
  const [usage, setUsage] = useState<UsageSummary | null>(null)

  const { data, error, loading, reload } = useResource(() => settingsApi.get(), [])
  const models = useResource(() => settingsApi.models(), [])
  // 运维可见性（FR-909）：设置页直接呈现健康检查的降级原因，省得用户去翻日志
  const health = useResource(() => settingsApi.health(), [])

  // 表单草稿优先于服务端快照：用户在未保存时不会被后台刷新覆盖
  const draft: Settings | null = draftOverride ?? data?.settings ?? null
  const setDraft = setDraftOverride

  useEffect(() => {
    void settingsApi
      .usage()
      .then(setUsage)
      .catch(() => setUsage(null))
  }, [])

  if (loading) return <Spinner label="正在加载设置…" />
  if (error) return <Banner onRetry={reload}>{error}</Banner>
  if (!draft) return null

  const editable = draft.read_only_fields
  const modelOptions: ModelInfo[] = models.data?.items ?? []

  function patch(changes: Partial<Settings>): void {
    // 草稿的初始 state 是 null（此时界面上显示的是服务端快照），
    // 因此必须从"当前生效的快照"派生第一条草稿——只看 state 的话，
    // 首次编辑会被原样返回、静默丢弃：输入框里打完字立刻回弹，看着像没保存。
    setDraft((current) => {
      const base = current ?? data?.settings
      return base === undefined || base === null ? current : { ...base, ...changes }
    })
  }

  function save(): void {
    if (!draft) return
    setBusy(true)
    setProblem(null)
    setSaved(null)
    const payload = { ...draft }
    delete (payload as Partial<Settings>).data_dir
    delete (payload as Partial<Settings>).read_only_fields
    delete (payload as Partial<Settings>).api_key
    // model_plan 是后端解析出的只读视图，提交回去没有意义
    delete (payload as Partial<Settings>).model_plan

    void settingsApi
      .save(payload as unknown as Record<string, unknown>)
      .then((body) => {
        setDraft(body.settings)
        setSaved('设置已保存并立即生效。')
      })
      .catch((cause: unknown) => setProblem(errorMessage(cause)))
      .finally(() => setBusy(false))
  }

  return (
    <div className="stack" style={{ maxWidth: 980 }}>
      <div className="page-head">
        <div>
          <h1>设置</h1>
          <p className="muted">所有数据都保存在本机，设置改动保存后立即生效。</p>
        </div>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={save}>
          {busy ? '保存中…' : '保存设置'}
        </button>
      </div>

      {problem ? <Banner>{problem}</Banner> : null}
      {saved ? <Banner tone="notice">{saved}</Banner> : null}
      {health.data?.status === 'degraded' ? (
        <Banner tone="notice">
          自检发现问题：{health.data.problems.join('、')}（存储：{health.data.storage.ok ? '正常' : health.data.storage.detail}
          ；数据库版本 v{health.data.storage.schema_version}）
          <button type="button" className="btn btn-sm" onClick={health.reload}>
            重新自检
          </button>
        </Banner>
      ) : null}

      <div className="card">
        <h2>模型与 API Key</h2>
        <div className="form-grid">
          <div className="field">
            <label htmlFor="base_url">接口地址（base_url）</label>
            <input
              id="base_url"
              value={draft.base_url}
              onChange={(event) => patch({ base_url: event.target.value })}
            />
            <span className="hint">默认 https://api.deepseek.com，可保留子路径（如 /v1）。</span>
          </div>
          <div className="field">
            <label htmlFor="model_default">默认模型</label>
            <input
              id="model_default"
              value={draft.model_default}
              list="model-options"
              onChange={(event) => patch({ model_default: event.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="model_pro">深度模型</label>
            <input
              id="model_pro"
              value={draft.model_pro}
              list="model-options"
              onChange={(event) => patch({ model_pro: event.target.value })}
            />
          </div>
        </div>
        <datalist id="model-options">
          {modelOptions.map((model) => (
            <option key={model.id} value={model.id} />
          ))}
        </datalist>

        <div className="form-grid">
          <div className="field">
            <label htmlFor="thinking">思考模式</label>
            <select
              id="thinking"
              value={draft.thinking_enabled ? 'on' : 'off'}
              onChange={(event) => patch({ thinking_enabled: event.target.value === 'on' })}
            >
              <option value="on">开启（推理更深，速度更慢）</option>
              <option value="off">关闭（更快、更省）</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="effort">思考档位</label>
            <select
              id="effort"
              value={draft.reasoning_effort}
              disabled={!draft.thinking_enabled}
              onChange={(event) => patch({ reasoning_effort: event.target.value as Settings['reasoning_effort'] })}
            >
              <option value="low">low</option>
              <option value="high">high</option>
              <option value="max">max</option>
            </select>
            <span className="hint">仅在思考模式开启时下发。</span>
          </div>
          <div className="field">
            <label htmlFor="show_reasoning">展示思维链</label>
            <select
              id="show_reasoning"
              value={draft.show_reasoning ? 'on' : 'off'}
              onChange={(event) => patch({ show_reasoning: event.target.value === 'on' })}
            >
              <option value="on">展示（可折叠）</option>
              <option value="off">不展示</option>
            </select>
          </div>
        </div>

        <div className="field">
          <label htmlFor="api_key">API Key</label>
          <div className="row-between">
            <input
              id="api_key"
              type="password"
              value={apiKey}
              placeholder={draft.api_key.configured ? `当前：${draft.api_key.masked ?? '已配置'}` : '尚未配置'}
              onChange={(event) => setApiKey(event.target.value)}
            />
            <div className="btn-row">
              <button
                type="button"
                className="btn"
                disabled={busy || !apiKey.trim()}
                onClick={() => {
                  setBusy(true)
                  setKeyMessage(null)
                  void settingsApi
                    .saveApiKey(apiKey.trim())
                    .then((body) => {
                      // 后端只回掩码，明文永远不会回到前端
                      patch({ api_key: { ...draft.api_key, ...body } })
                      setApiKey('')
                      setKeyMessage('API Key 已保存（仅以掩码显示）。')
                    })
                    .catch((cause: unknown) => setKeyMessage(errorMessage(cause)))
                    .finally(() => setBusy(false))
                }}
              >
                保存 Key
              </button>
              <button
                type="button"
                className="btn btn-danger"
                disabled={busy || !draft.api_key.configured}
                onClick={() => {
                  if (!window.confirm('确定清除已保存的 API Key 吗？清除后将无法调用模型。')) return
                  setBusy(true)
                  void settingsApi
                    .saveApiKey('')
                    .then((body) => {
                      patch({ api_key: { ...draft.api_key, ...body } })
                      setKeyMessage('已清除。')
                    })
                    .catch((cause: unknown) => setKeyMessage(errorMessage(cause)))
                    .finally(() => setBusy(false))
                }}
              >
                清除
              </button>
            </div>
          </div>
          <span className="hint">
            来源：{draft.api_key.source === 'env' ? '环境变量（优先级高于设置页）' : draft.api_key.source || '未配置'}
            {draft.api_key.source === 'env' ? ' · 环境变量存在时设置页的 Key 不生效' : ''}
          </span>
        </div>

        {keyMessage ? <Banner tone="notice">{keyMessage}</Banner> : null}

        <div className="field">
          <label htmlFor="test_model">测试连接</label>
          <div className="row-between">
            <input
              id="test_model"
              value={testModel}
              list="model-options"
              placeholder={`留空则测试 ${draft.model_default}`}
              onChange={(event) => setTestModel(event.target.value)}
            />
            <div className="btn-row">
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() => {
                  setBusy(true)
                  setTestResult(null)
                  void settingsApi
                    .testConnection(testModel.trim() || undefined)
                    .then(setTestResult)
                    .catch((cause: unknown) => setProblem(errorMessage(cause)))
                    .finally(() => setBusy(false))
                }}
              >
                测试
              </button>
              <button type="button" className="btn" disabled={busy} onClick={() => models.reload()}>
                刷新模型列表
              </button>
            </div>
          </div>
          {testResult ? (
            <span className={testResult.ok ? 'hint' : 'hint'} style={{ color: testResult.ok ? undefined : '#b91c1c' }}>
              {testResult.ok ? '✓ ' : '✗ '}
              {testResult.message}
              {testResult.latency_ms !== null ? `（${testResult.latency_ms} ms）` : ''}
            </span>
          ) : null}
        </div>

        {modelOptions.length > 0 ? (
          <details>
            <summary>可用模型（{modelOptions.length} 个，来自官方 /models）</summary>
            <table className="table">
              <thead>
                <tr>
                  <th>模型</th>
                  <th>上下文窗口</th>
                  <th>最大输出</th>
                  <th>思考档位</th>
                </tr>
              </thead>
              <tbody>
                {modelOptions.map((model) => (
                  <tr key={model.id}>
                    <td>{model.id}</td>
                    <td>{model.context_window}</td>
                    <td>{model.max_output_tokens}</td>
                    <td>{model.effort_levels.join(' / ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">模型元数据来自接口，不写死在代码里，官方调整后无需改代码。</p>
          </details>
        ) : null}
      </div>

      <div className="card">
        <h2>按用途的模型与思考参数</h2>
        <div className="field">
          <label htmlFor="recommended" style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <input
              id="recommended"
              type="checkbox"
              style={{ width: 'auto' }}
              checked={draft.use_recommended_overrides}
              disabled={busy}
              onChange={(event) => patch({ use_recommended_overrides: event.target.checked })}
            />
            使用推荐组合
          </label>
          <span className="hint">
            推荐组合：路径规划与作答判定改用深度模型并开思考档 high，上下文压缩关闭思考（省时省钱）。
            下表的逐项设置会覆盖推荐组合。
          </span>
        </div>
        <PurposeStrategyTable draft={draft} busy={busy} onPatch={patch} />
        <p className="muted small">
          右列「当前会实际使用」由后端解析后返回，是判断覆盖是否生效的依据；界面上的选择本身不代表已生效。
        </p>
      </div>

      <div className="card">
        <h2>生成参数</h2>
        <div className="form-grid">
          <NumberField
            id="max_tokens"
            label="单次最大输出 token"
            value={draft.max_tokens}
            min={256}
            disabled={busy}
            onChange={(value) => patch({ max_tokens: value })}
          />
          <NumberField
            id="timeout_seconds"
            label="单次请求超时（秒）"
            hint="允许 30~600；规划与判定这类调用本身较慢"
            value={draft.timeout_seconds}
            min={30}
            max={600}
            disabled={busy}
            onChange={(value) => patch({ timeout_seconds: value })}
          />
          <NumberField
            id="max_retries"
            label="失败重试次数"
            hint="仅在实际调用失败时生效（0~5）"
            value={draft.max_retries}
            min={0}
            max={5}
            disabled={busy}
            onChange={(value) => patch({ max_retries: value })}
          />
          <NumberField
            id="temperature"
            label="采样温度 temperature"
            hint="0~2；思考模式下不生效"
            value={draft.temperature}
            min={0}
            max={2}
            step={0.1}
            disabled={busy}
            onChange={(value) => patch({ temperature: value })}
          />
          <NumberField
            id="top_p"
            label="核采样 top_p"
            hint="0.01~1；仅思考模式下生效，有效范围 0.95~1.0"
            value={draft.top_p}
            min={0.01}
            max={1}
            step={0.01}
            disabled={busy}
            onChange={(value) => patch({ top_p: value })}
          />
        </div>

        <div className="field">
          <strong className="small">参数生效性（官方约束，FR-718）</strong>
          <ul className="small muted">
            <li>
              temperature：
              {draft.thinking_enabled
                ? '当前思考模式开启——该参数不生效，官方会直接忽略它（保存不报错，但结果不受影响）。'
                : '当前思考模式关闭，该参数生效。'}
            </li>
            <li>
              top_p：
              {draft.thinking_enabled
                ? `仅在思考模式下生效，且官方把有效范围钳制到 0.95~1.0${
                    draft.top_p < 0.95 ? `；当前填的是 ${draft.top_p}，实际按 0.95 处理。` : '；当前取值在有效范围内。'
                  }`
                : '当前思考模式关闭——该参数不生效。'}
            </li>
            <li>presence_penalty / frequency_penalty：本项目不下发这两个参数，因此不存在"调了没用"的困惑。</li>
            <li>
              思考档位：
              {draft.thinking_enabled ? '思考开启时按上表下发。' : '当前思考模式关闭，档位不生效。'}
            </li>
          </ul>
        </div>

        <div className="field">
          <label htmlFor="audit" style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <input
              id="audit"
              type="checkbox"
              style={{ width: 'auto' }}
              checked={draft.audit_enabled}
              disabled={busy}
              onChange={(event) => patch({ audit_enabled: event.target.checked })}
            />
            记录完整请求与响应（排查问题用，默认关闭）
          </label>
          <span className="hint">
            开启后会把每次调用的完整提示词与模型响应写入日志，便于核对"到底发了什么"（FR-715）。
            API Key 始终脱敏；日志会包含你的学习内容，排查完请记得关掉。
          </span>
        </div>
      </div>

      <div className="card">
        <h2>学习流程</h2>
        <div className="form-grid">
          <NumberField
            id="assess_max_turns"
            label="测评轮次上限"
            hint="默认 6，允许 3~12"
            value={draft.assess_max_turns}
            min={3}
            max={12}
            disabled={busy}
            onChange={(value) => patch({ assess_max_turns: value })}
          />
          <NumberField
            id="path_min_nodes"
            label="路径最少节点数"
            value={draft.path_min_nodes}
            min={3}
            max={15}
            disabled={busy}
            onChange={(value) => patch({ path_min_nodes: value })}
          />
          <NumberField
            id="path_max_nodes"
            label="路径最多节点数"
            value={draft.path_max_nodes}
            min={3}
            max={15}
            disabled={busy}
            onChange={(value) => patch({ path_max_nodes: value })}
          />
          <NumberField
            id="node_max_retries"
            label="单节点自动重讲上限"
            hint="超过后由你选择重讲 / 看答案 / 跳过"
            value={draft.node_max_retries}
            min={0}
            max={5}
            disabled={busy}
            onChange={(value) => patch({ node_max_retries: value })}
          />
          <NumberField
            id="compress_threshold_rounds"
            label="压缩触发轮数"
            value={draft.compress_threshold_rounds}
            min={4}
            disabled={busy}
            onChange={(value) => patch({ compress_threshold_rounds: value })}
          />
          <NumberField
            id="compress_round_count"
            label="单次压缩轮数"
            value={draft.compress_round_count}
            min={1}
            disabled={busy}
            onChange={(value) => patch({ compress_round_count: value })}
          />
        </div>
      </div>

      <div className="card">
        <h2>运行与数据</h2>
        <dl className="detail-list">
          <dt>日志级别</dt>
          <dd>
            <select value={draft.log_level} onChange={(event) => patch({ log_level: event.target.value })}>
              {['DEBUG', 'INFO', 'WARNING', 'ERROR'].map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </dd>
          <dt>数据目录</dt>
          <dd>
            {draft.data_dir}
            {editable.includes('data_dir') ? <span className="muted">（只读，由环境变量决定）</span> : null}
          </dd>
          <dt>累计用量</dt>
          <dd>
            {usage === null
              ? '（读取失败）'
              : `${usage.calls} 次调用 · 输入 ${usage.prompt_tokens} / 输出 ${usage.completion_tokens} token · 推理 ${usage.reasoning_tokens} · 缓存命中率 ${(
                  usage.cache_hit_rate * 100
                ).toFixed(1)}% · 累计耗时 ${formatDuration(usage.duration_ms)}`}
          </dd>
        </dl>

        {usage?.cache_hint ? <Banner tone="notice">{usage.cache_hint}</Banner> : null}

        {usage && Object.keys(usage.by_purpose ?? {}).length > 0 ? (
          <details>
            <summary>按用途的用量（{Object.keys(usage.by_purpose ?? {}).length} 项）</summary>
            <table className="table">
              <thead>
                <tr>
                  <th>用途</th>
                  <th>调用</th>
                  <th>输入 token</th>
                  <th>输出 token</th>
                  <th>推理 token</th>
                  <th>命中率</th>
                  <th>耗时</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(usage.by_purpose ?? {}).map(([purpose, bucket]) => (
                  <tr key={purpose}>
                    <td>{PURPOSE_LABELS[purpose as LlmPurposeKey] ?? purpose}</td>
                    <td>{bucket.calls}</td>
                    <td>{bucket.prompt_tokens.toLocaleString('zh-CN')}</td>
                    <td>{bucket.completion_tokens.toLocaleString('zh-CN')}</td>
                    <td>{bucket.reasoning_tokens.toLocaleString('zh-CN')}</td>
                    <td>{(bucket.cache_hit_rate * 100).toFixed(1)}%</td>
                    <td>{formatDuration(bucket.duration_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ) : null}

        {usage && (usage.by_subject ?? []).length > 0 ? (
          <details>
            <summary>按主题的用量</summary>
            <table className="table">
              <thead>
                <tr>
                  <th>主题</th>
                  <th>调用</th>
                  <th>总 token</th>
                  <th>命中率</th>
                  <th>耗时</th>
                </tr>
              </thead>
              <tbody>
                {(usage.by_subject ?? []).map((item) => (
                  <tr key={item.subject_id}>
                    <td>{item.title}</td>
                    <td>{item.calls}</td>
                    <td>{item.total_tokens.toLocaleString('zh-CN')}</td>
                    <td>{(item.cache_hit_rate * 100).toFixed(1)}%</td>
                    <td>{formatDuration(item.duration_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ) : null}

        <p className="muted small">
          缓存命中率长期为 0 通常意味着对话前缀被改写（每次请求都无法命中缓存）；上下文压缩本身会重排前缀，
          压缩后出现一次"冷启动"是预期代价。可以把日志级别调成 DEBUG、并打开上面的请求审计来核对请求体。
        </p>
      </div>
    </div>
  )
}
