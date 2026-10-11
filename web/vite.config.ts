import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 后端默认跑在 127.0.0.1:8000；开发期用代理把 /api 转过去，
// 这样前端代码里始终用同源相对路径，不区分开发与生产（NFR-101）。
const BACKEND = process.env.WELEARN_BACKEND ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  build: {
    // 单页应用、本机访问：Markdown 渲染 + KaTeX + 代码高亮本身就是这个体积，
    // 拆包带来的复杂度（懒加载骨架、Suspense 边界）没有对应的收益
    chunkSizeWarningLimit: 1024,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
        // SSE 不能被代理层缓冲，否则流式会被攒成一整块
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache, no-transform'
          })
        },
      },
    },
  },
})
