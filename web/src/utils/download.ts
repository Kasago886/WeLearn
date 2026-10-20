/** 触发浏览器下载。用 Blob + 临时链接触发，避免后端必须设置 Content-Disposition。 */
export function downloadText(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // 立刻回收会让部分浏览器来不及开始下载，延后一拍更稳
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
