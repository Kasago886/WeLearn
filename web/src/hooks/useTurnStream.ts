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
 * 一次流式回合的状态机。
 *
 * 同一时刻只允许一个回合在跑：新回合会先中断旧回合，避免两路增量交错写进同一段文本。
 */
export function useTurnStream() {
  const [state, setState] = useState<TurnStreamState>(INITIAL)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  const run = useCallback(
    async (path: string, body?: unknown, onEvent?: (event: SseEvent) => void): Promise<void> => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      setState({ ...INITIAL, streaming: true })

      try {
        await streamTurn(
          path,
          body,
          {
            onToken: (chunk) => setState((prev) => ({ ...prev, text: prev.text + chunk })),
            onReasoning: (chunk) => setState((prev) => ({ ...prev, reasoning: prev.reasoning + chunk })),
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
        const aborted = controller.signal.aborted
        setState((prev) => ({ ...prev, streaming: false, stopped: aborted }))
        if (abortRef.current === controller) {
          abortRef.current = null
        }
      }
    },
    [],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setState(INITIAL)
  }, [])

  return { ...state, run, stop, reset }
}
