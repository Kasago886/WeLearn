import { useCallback, useEffect, useRef, useState } from 'react'

import { isAborted } from '../api/client'
import { streamTurn } from '../api/sse'
import type { SseEvent } from '../api/types'

export interface TurnStreamState {
  /** 本轮累计的正文（含后端用 on_delta 直接推送的本地文本）。 */
  text: string
  /** 本轮累计的思维链（FR-713）。 */
  reasoning: string
  streaming: boolean
  /** 被用户主动停止：已生成部分保留并标记为"未完成"（FR-412）。 */
  stopped: boolean
  /** 未进入流式就被拒的请求级错误（404/409 等）。 */
  httpError: string | null
}

const INITIAL: TurnStreamState = {
  text: '',
  reasoning: '',
  streaming: false,
  stopped: false,
  httpError: null,
}

/**
 * 流式增量合并到界面的间隔（NFR-101 要求 ≤100ms）。
 *
 * 模型按"碎片"吐字（实测一次回答 90+ 个正文碎片），每个碎片都 setState 会让
 * React 在一秒内重渲染上百次，长回答时明显掉帧。攒到下一个 60ms 再合并提交：
 * 视觉上仍是"逐字出现"，渲染次数却降一个数量级。
 */
const FLUSH_INTERVAL_MS = 60

/**
 * 一次流式回合的状态机。
 *
 * 同一时刻只允许一个回合在跑：新回合会先中断旧回合，避免两路增量交错写进同一段文本。
 */
export function useTurnStream() {
  const [state, setState] = useState<TurnStreamState>(INITIAL)
  const abortRef = useRef<AbortController | null>(null)
  // 尚未提交到 state 的增量。用 ref 而不是 state：写入频率很高，走 state 就失去节流的意义
  const pendingRef = useRef({ text: '', reasoning: '' })
  const timerRef = useRef<number | null>(null)

  const flush = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    const pending = pendingRef.current
    if (!pending.text && !pending.reasoning) return
    pendingRef.current = { text: '', reasoning: '' }
    setState((prev) => ({
      ...prev,
      text: prev.text + pending.text,
      reasoning: prev.reasoning + pending.reasoning,
    }))
  }, [])

  const schedule = useCallback(() => {
    if (timerRef.current !== null) return
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null
      flush()
    }, FLUSH_INTERVAL_MS)
  }, [flush])

  useEffect(
    () => () => {
      abortRef.current?.abort()
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
        timerRef.current = null
      }
    },
    [],
  )

  const run = useCallback(
    async (path: string, body?: unknown, onEvent?: (event: SseEvent) => void): Promise<void> => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      pendingRef.current = { text: '', reasoning: '' }
      setState({ ...INITIAL, streaming: true })

      try {
        await streamTurn(
          path,
          body,
          {
            onToken: (chunk) => {
              pendingRef.current.text += chunk
              schedule()
            },
            onReasoning: (chunk) => {
              pendingRef.current.reasoning += chunk
              schedule()
            },
            onEvent,
            onHttpError: (message) => setState((prev) => ({ ...prev, httpError: message })),
          },
          controller.signal,
        )
      } catch (error) {
        if (!isAborted(error)) {
          setState((prev) => ({ ...prev, httpError: '流式连接意外中断，请重试。' }))
        }
      } finally {
        // 收尾必须先冲刷：否则最后几十毫秒的正文会丢，用户看到的是被截断的回答
        flush()
        const aborted = controller.signal.aborted
        setState((prev) => ({ ...prev, streaming: false, stopped: aborted }))
        if (abortRef.current === controller) {
          abortRef.current = null
        }
      }
    },
    [flush, schedule],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    pendingRef.current = { text: '', reasoning: '' }
    setState(INITIAL)
  }, [])

  return { ...state, run, stop, reset }
}
