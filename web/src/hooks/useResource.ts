import { useCallback, useEffect, useRef, useState } from 'react'

import { errorMessage, isAborted } from '../api/client'

export interface Resource<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/**
 * 只读加载（GET）的通用形态。
 *
 * 刻意不做缓存与并发去重：本项目是单用户本地应用，请求都是毫秒级的本地查询，
 * 真正的成本在模型调用上，不值得为此引入数据层依赖。
 */
export function useResource<T>(loader: (signal: AbortSignal) => Promise<T>, deps: unknown[]): Resource<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)

  const loaderRef = useRef(loader)
  loaderRef.current = loader

  useEffect(() => {
    const controller = new AbortController()
    let alive = true

    setLoading(true)
    loaderRef
      .current(controller.signal)
      .then((value) => {
        if (!alive) return
        setData(value)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (!alive || isAborted(cause)) return
        setError(errorMessage(cause))
      })
      .finally(() => {
        if (alive) setLoading(false)
      })

    return () => {
      alive = false
      controller.abort()
    }
    // loader 每次渲染都会变，真正的依赖由调用方通过 deps 声明
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((value) => value + 1), [])

  return { data, error, loading, reload }
}
