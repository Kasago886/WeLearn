import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'

import { Markdown } from './Markdown'
import { ReasoningPanel } from './Banner'

export function Conversation({ children }: { children: ReactNode }) {
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  })

  return (
    <div className="conversation">
      {children}
      <div ref={endRef} />
    </div>
  )
}

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="row row-user">
      <div className="bubble bubble-user">
        <p>{text}</p>
      </div>
    </div>
  )
}

export function AssistantBubble({ children }: { children: ReactNode }) {
  return (
    <div className="row row-assistant">
      <div className="bubble bubble-assistant">{children}</div>
    </div>
  )
}

export function AssistantText({ text }: { text: string }) {
  return <AssistantBubble>{text.trim() ? <Markdown text={text} /> : <p className="muted">（无内容）</p>}</AssistantBubble>
}

/** 流式中的气泡：带光标，停止后标记为"未完成"（FR-412）。 */
export function StreamingBubble({
  text,
  reasoning,
  streaming,
  stopped,
  showReasoning,
}: {
  text: string
  reasoning: string
  streaming: boolean
  stopped: boolean
  showReasoning: boolean
}) {
  return (
    <AssistantBubble>
      {showReasoning ? <ReasoningPanel text={reasoning} open={streaming} /> : null}
      {text.trim() ? <Markdown text={text} /> : streaming ? <p className="muted">正在思考…</p> : null}
      {streaming ? <span className="caret" aria-label="正在生成" /> : null}
      {stopped && !streaming ? <p className="muted">（已停止生成，内容未完成）</p> : null}
    </AssistantBubble>
  )
}

export function TypingBubble({ label }: { label: string }) {
  return (
    <AssistantBubble>
      <p className="muted">
        {label}
        <span className="caret" aria-label="正在生成" />
      </p>
    </AssistantBubble>
  )
}
