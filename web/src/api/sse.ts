/**
 * SSE 读取（FR-412：流式 + 停止生成）。
 *
 * **为什么不用 EventSource**：所有流式端点都是 POST（要带作答内容），
 * 而 EventSource 只支持 GET。这里改用 fetch + ReadableStream 手工解析帧，
 * 顺带拿到了 EventSource 没有的能力——用 AbortController 真正中断请求。
 */

import { ApiError, isAborted } from './client'
import type { SseEvent, SseEventType } from './types'

export interface TurnHandlers {
  onToken?: (text: string) => void
  onReasoning?: (text: string) => void
  onEvent?: (event: SseEvent) => void
  /** 后端在建立流派之前返回的非 2xx（例如主题不存在）：需要给出 404/409 提示。 */
  onHttpError?: (message: string, status: number) => void
}

const FRAME_SEPARATOR = '\n\n'

function parseFrame(raw: string): SseEvent | null {
  let type = ''
  let data = ''
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      type = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      data += line.slice('data:'.length).trim()
    }
  }
  if (!type) {
    return null
  }
  try {
    return { type: type as SseEventType, data: (data ? JSON.parse(data) : {}) as Record<string, unknown> }
  } catch {
    // 半截或畸形帧：丢弃而不是让整个页面崩掉
    return null
  }
}

function dispatch(frame: SseEvent, handlers: TurnHandlers): void {
  if (frame.type === 'token') {
    handlers.onToken?.(String(frame.data.text ?? ''))
  } else if (frame.type === 'reasoning') {
    handlers.onReasoning?.(String(frame.data.text ?? ''))
  }
  handlers.onEvent?.(frame)
}

/**
 * 发起一次流式回合。正常结束时返回；被中断时抛出 AbortError（调用方据此区分"停止"）。
 */
export async function streamTurn(
  path: string,
  body: unknown,
  handlers: TurnHandlers,
  signal: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    })
  } catch (error) {
    if (isAborted(error)) {
      throw error
    }
    handlers.onHttpError?.('无法连接到后端服务，请确认服务已启动（默认 127.0.0.1:8000）。', 0)
    return
  }

  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}），请稍后重试。`
    try {
      const payload = (await response.json()) as { detail?: unknown }
      if (typeof payload.detail === 'string' && payload.detail.trim()) {
        message = payload.detail
      }
    } catch {
      // 非 JSON 错误体：用兜底文案
    }
    handlers.onHttpError?.(message, response.status)
    return
  }

  if (!response.body) {
    handlers.onHttpError?.('后端没有返回流式响应体，请检查是否使用了支持 SSE 的环境。', response.status)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      buffer += decoder.decode(value, { stream: true })

      let index = buffer.indexOf(FRAME_SEPARATOR)
      while (index >= 0) {
        const raw = buffer.slice(0, index)
        buffer = buffer.slice(index + FRAME_SEPARATOR.length)
        const frame = parseFrame(raw)
        if (frame) {
          dispatch(frame, handlers)
        }
        index = buffer.indexOf(FRAME_SEPARATOR)
      }
    }
    const tail = parseFrame(buffer)
    if (tail) {
      dispatch(tail, handlers)
    }
  } finally {
    reader.releaseLock()
  }
}

export { ApiError }
