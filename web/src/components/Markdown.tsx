/**
 * 内容渲染（NFR-119）：Markdown + 代码高亮 + 数学公式，失败时降级为纯文本。
 *
 * 安全：**不启用 rehype-raw**，因此模型输出里的原始 HTML 不会被渲染成 DOM 节点，
 * 只作为文本出现（NFR-110 的第一道防线：渲染期不允许注入脚本）。
 */

import type { ComponentProps, ReactNode } from 'react'
import { Component } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github-dark.css'

interface BoundaryProps {
  text: string
  children: ReactNode
}

interface BoundaryState {
  failed: boolean
  /** 触发失败的那段文本：文本变了就自动重试渲染，而不必用 componentDidUpdate 改状态。 */
  failedText: string
}

/** 渲染失败（例如公式语法异常）时退回纯文本，而不是整页白屏。 */
class RenderBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { failed: false, failedText: '' }

  static getDerivedStateFromError(): Partial<BoundaryState> {
    return { failed: true }
  }

  static getDerivedStateFromProps(props: BoundaryProps, state: BoundaryState): Partial<BoundaryState> | null {
    if (state.failed && props.text !== state.failedText) {
      return { failed: false, failedText: '' }
    }
    return null
  }

  componentDidCatch(): void {
    this.setState({ failedText: this.props.text })
  }

  render(): ReactNode {
    if (this.state.failed) {
      return <pre className="plain-fallback">{this.props.text}</pre>
    }
    return this.props.children
  }
}

// 只注册常用语言：highlight.js 全量包会让首屏体积翻三倍，
// 而学习概念时几乎用不到冷门语言（NFR-119）
type MarkdownProps = ComponentProps<typeof ReactMarkdown>

const REMARK_PLUGINS: NonNullable<MarkdownProps['remarkPlugins']> = [remarkGfm, remarkMath]
const REHYPE_PLUGINS: NonNullable<MarkdownProps['rehypePlugins']> = [
  [rehypeKatex, {}],
  [rehypeHighlight, { detect: false }],
]

export function Markdown({ text }: { text: string }) {
  return (
    <RenderBoundary text={text}>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>
          {text}
        </ReactMarkdown>
      </div>
    </RenderBoundary>
  )
}
