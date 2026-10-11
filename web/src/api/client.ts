/**
 * HTTP 客户端（NFR-116：失败必须给出"发生了什么 + 可以做什么"）。
 *
 * 约定：后端错误体统一为 `{detail: string}`，detail 本身就是面向用户的中文文案，
 * 因此这里直接透出，不做二次包装；网络层失败才使用本地文案。
 */

export class ApiError extends Error {
  readonly status: number
  readonly retryable: boolean

  constructor(message: string, status: number, retryable = false) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryable = retryable
  }
}

export class NetworkError extends Error {
  readonly retryable = true

  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

/** 把任意异常转成一句可以直接显示的文案。 */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError || error instanceof NetworkError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return '出现了未知问题，请稍后重试。'
}

export function isAborted(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string' && body.detail.trim()) {
      return body.detail
    }
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      // pydantic 校验错误：取第一条，避免把整个 JSON 丢给用户
      const first = body.detail[0] as { msg?: string }
      if (typeof first?.msg === 'string') {
        return `请求参数不合法：${first.msg}`
      }
    }
  } catch {
    // 响应体不是 JSON（例如 500 的纯文本）：走下面的兜底文案
  }
  return `请求失败（HTTP ${response.status}），请稍后重试。`
}

const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504])

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    })
  } catch (error) {
    if (isAborted(error)) {
      throw error
    }
    throw new NetworkError('无法连接到后端服务，请确认服务已启动（默认 127.0.0.1:8000）。')
  }

  if (!response.ok) {
    throw new ApiError(await readDetail(response), response.status, RETRYABLE_STATUS.has(response.status))
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>('GET', path, undefined, signal),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) => request<T>('POST', path, body, signal),
  put: <T>(path: string, body?: unknown, signal?: AbortSignal) => request<T>('PUT', path, body, signal),
  patch: <T>(path: string, body?: unknown, signal?: AbortSignal) => request<T>('PATCH', path, body, signal),
  del: <T>(path: string, signal?: AbortSignal) => request<T>('DELETE', path, undefined, signal),
}
