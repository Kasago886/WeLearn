import { useEffect, useState } from 'react'

import { api, setAccessToken } from '../api/client'

/**
 * 可选访问口令的输入层（FR-903）。
 *
 * 只在客户端遇到 401 时才出现：没设口令的用户永远不会看到它，
 * 设了口令的人也不会被一个"你还没登录"的页面挡在门外——他们本来就只是想看自己的数据。
 */
export function AuthPrompt() {
  const [visible, setVisible] = useState(false)
  const [token, setToken] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    function open(): void {
      setVisible(true)
      setProblem(null)
    }
    window.addEventListener('welearn:auth-required', open)
    return () => window.removeEventListener('welearn:auth-required', open)
  }, [])

  if (!visible) return null

  async function submit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const value = token.trim()
    if (!value || busy) return
    setBusy(true)
    setProblem(null)
    // 先落到 localStorage，api.get 才会带上这个头
    setAccessToken(value)
    try {
      const result = await api.get<{ ok: boolean; required: boolean }>('/api/auth/verify')
      if (!result.ok) {
        setAccessToken(null)
        setProblem('口令不正确，请重新输入。')
        return
      }
      window.location.reload()
    } catch {
      setAccessToken(null)
      setProblem('无法验证口令，请确认后端服务已启动。')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="需要访问口令">
      <form className="card stack modal-card" onSubmit={(event) => void submit(event)}>
        <h2>需要访问口令</h2>
        <p className="muted small">
          这个 WeLearn 实例设置了访问口令。口令只用于拦住匿名的数据访问，输入一次后会记住。
        </p>
        <div className="field">
          <label htmlFor="access-token">访问口令</label>
          <input
            id="access-token"
            type="password"
            autoFocus
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="在设置页里设置的那个口令"
          />
        </div>
        {problem ? <p className="small" style={{ color: 'var(--danger)' }}>{problem}</p> : null}
        <div className="btn-row">
          <button type="submit" className="btn btn-primary" disabled={busy || !token.trim()}>
            {busy ? '正在验证…' : '进入'}
          </button>
        </div>
      </form>
    </div>
  )
}
