import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <div className="card empty">
      <h1>页面不存在</h1>
      <p>链接可能已经失效，回到主题列表继续吧。</p>
      <Link className="btn btn-primary" to="/">
        回到学习主题
      </Link>
    </div>
  )
}
