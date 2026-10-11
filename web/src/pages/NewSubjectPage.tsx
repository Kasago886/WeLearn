import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { errorMessage } from '../api/client'
import { assessmentApi, subjectsApi } from '../api/endpoints'
import { Banner } from '../components/Banner'

const DEPTH_OPTIONS = ['入门了解', '系统掌握', '深入原理']

/**
 * 新建主题（P2）。
 *
 * 「跳过测评」不是省略步骤，而是明确地按零基础处理（FR-205）：后端会落下"未测评"的痕迹，
 * 画像与路径都会带上这个前提。
 */
export function NewSubjectPage() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [goal, setGoal] = useState('')
  const [depth, setDepth] = useState('')
  const [background, setBackground] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function create(): Promise<string | null> {
    setBusy(true)
    setError(null)
    try {
      const created = await subjectsApi.create({
        title,
        goal: goal.trim() || null,
        depth: depth.trim() || null,
        background: background.trim() || null,
      })
      if (created.duplicate_title_hint) {
        // 同名不是错误（FR-107），继续流程，由主题页体现"已有同名"
        console.info('已存在同名主题，继续创建')
      }
      return created.subject.id
    } catch (cause) {
      setError(errorMessage(cause))
      return null
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack" style={{ maxWidth: 720 }}>
      <div className="page-head">
        <div>
          <h1>新建学习主题</h1>
          <p className="muted">先给这个主题起个名字，其余都可以留空。</p>
        </div>
        <Link className="btn" to="/">
          返回列表
        </Link>
      </div>

      {error ? <Banner>{error}</Banner> : null}

      <form
        className="card"
        onSubmit={(event) => {
          event.preventDefault()
          void (async () => {
            const id = await create()
            if (id) navigate(`/subjects/${id}/assessment`)
          })()
        }}
      >
        <div className="field">
          <label htmlFor="title">概念名称（必填）</label>
          <input
            id="title"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder="例如：贝叶斯定理"
            required
          />
          <span className="hint">写具体概念比写大领域更好教，例如「贝叶斯定理」优于「概率论」。</span>
        </div>

        <div className="form-grid">
          <div className="field">
            <label htmlFor="goal">学习目标（选填）</label>
            <input
              id="goal"
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="例如：能在面试里讲清楚"
            />
          </div>
          <div className="field">
            <label htmlFor="depth">期望深度（选填）</label>
            <input
              id="depth"
              value={depth}
              onChange={(event) => setDepth(event.target.value)}
              placeholder="例如：系统掌握"
              list="depth-options"
            />
            <datalist id="depth-options">
              {DEPTH_OPTIONS.map((option) => (
                <option key={option} value={option} />
              ))}
            </datalist>
          </div>
        </div>

        <div className="field">
          <label htmlFor="background">已有背景（选填）</label>
          <textarea
            id="background"
            value={background}
            onChange={(event) => setBackground(event.target.value)}
            placeholder="例如：学过基础概率，但说不清公式里的每一项"
          />
          <span className="hint">写得越具体，测评与讲解越贴合你的实际情况。</span>
        </div>

        <div className="btn-row">
          <button type="submit" className="btn btn-primary" disabled={busy || !title.trim()}>
            {busy ? '创建中…' : '创建并开始知识基础确认'}
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy || !title.trim()}
            onClick={() =>
              void (async () => {
                const id = await create()
                if (!id) return
                try {
                  await assessmentApi.skip(id)
                } catch (cause) {
                  setError(errorMessage(cause))
                  return
                }
                // 让路径页直接开始生成，省掉一次额外点击
                navigate(`/subjects/${id}/path`, { state: { generate: true } })
              })()
            }
          >
            跳过测评，直接生成路径
          </button>
        </div>
      </form>
    </div>
  )
}
