import { useLayoutEffect, useRef } from 'react'
import type { ReactNode } from 'react'

import { Markdown } from './Markdown'
import { ReasoningPanel } from './Banner'

/** 距底部多少像素内仍算"在看最新内容"。超出这个范围就认为用户在翻看历史，不该把他拽回来。 */
const STICK_THRESHOLD_PX = 80

export function Conversation({ children }: { children: ReactNode }) {
  const boxRef = useRef<HTMLDivElement | null>(null)
  // 是否应该跟着新内容滚到底部。只由**用户的滚动**改变，不由内容变化改变。
  const stickRef = useRef(true)

  // 新内容出现时保持在底部。两处必须小心，否则会变成"每敲一个字整页往上跳"：
  //
  // 1. **直接设容器的 `scrollTop`，不要用 `scrollIntoView`**。后者会连同所有祖先滚动容器
  //    一起滚——包括整个窗口；而且 `block: 'end'` 是"把底边对齐到视口底边"，不是"必要时才滚"，
  //    所以它会把窗口往上拽，甚至把正在输入的框推出视口。容器自身的 `scrollTop` 只影响
  //    这一个盒子，绝不碰窗口（已实测：赋值前后 `window.scrollY` 完全不变）。
  // 2. **用"粘性标志"决定是否跟随，而不是每次去比高度差**。比高度差抓不住用户在两次渲染
  //    之间的滚动，而且盒子从"不可滚动"变成"可滚动"时 `scrollTop` 还是 0，会被误判成
  //    "用户滚远了"而拒绝跟随——明明那时用户就在底部。
  //
  // 用 `useLayoutEffect` 而不是 `useEffect`：滚动要在浏览器绘制前完成，否则会看到跳动。
  useLayoutEffect(() => {
    const box = boxRef.current
    if (box === null) return
    if (stickRef.current) {
      box.scrollTop = box.scrollHeight
    }
  })

  function handleScroll(): void {
    const box = boxRef.current
    if (box === null) return
    stickRef.current = box.scrollHeight - box.scrollTop - box.clientHeight <= STICK_THRESHOLD_PX
  }

  return (
    <div className="conversation" ref={boxRef} onScroll={handleScroll}>
      {children}
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
