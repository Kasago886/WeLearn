import { useEffect, useState } from 'react'

import { errorMessage } from '../api/client'
import { settingsApi } from '../api/endpoints'
import type { ModelInfo, Settings, TestConnectionResult, UsageSummary } from '../api/types'
import { Banner, Spinner } from '../components/Banner'
import { useResource } from '../hooks/useResource'

interface NumberFieldProps {
  id: string
  label: string
  hint?: string
  value: number
  min?: number
  max?: number
  disabled?: boolean
  onChange: (value: number) => void
}

function NumberField({ id, label, hint, value, min, max, disabled, onChange }: NumberFieldProps) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
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
    setDraft((current) => (current === null ? current : { ...current, ...changes }))
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
              : `${usage.calls} 次调用 · 输入 ${usage.prompt_tokens} / 输出 ${usage.completion_tokens} token · 缓存命中率 ${(
                  usage.cache_hit_rate * 100
                ).toFixed(1)}%`}
          </dd>
        </dl>
        <p className="muted small">
          缓存命中率长期为 0 通常意味着对话前缀被改写（每次请求都无法命中缓存），可以从日志里核对请求体。
        </p>
      </div>
    </div>
  )
}
