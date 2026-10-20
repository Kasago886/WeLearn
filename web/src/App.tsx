import { NavLink, Route, Routes } from 'react-router-dom'

import { AuthPrompt } from './components/AuthPrompt'
import { ArchivePage } from './pages/ArchivePage'
import { AssessmentPage } from './pages/AssessmentPage'
import { NewSubjectPage } from './pages/NewSubjectPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { PathPage } from './pages/PathPage'
import { ReviewPage } from './pages/ReviewPage'
import { SearchPage } from './pages/SearchPage'
import { SettingsPage } from './pages/SettingsPage'
import { StageRedirect } from './pages/StageRedirect'
import { SubjectLayout } from './pages/SubjectLayout'
import { SubjectsPage } from './pages/SubjectsPage'
import { TeachingPage } from './pages/TeachingPage'

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <NavLink to="/" className="brand">
          <strong>WeLearn</strong>
          <span className="muted">个性化概念学习</span>
        </NavLink>
        <nav className="app-nav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            学习主题
          </NavLink>
          <NavLink to="/search" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            检索
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            设置
          </NavLink>
        </nav>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<SubjectsPage />} />
          <Route path="/new" element={<NewSubjectPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/subjects/:subjectId" element={<SubjectLayout />}>
            {/* 刷新恢复：由状态机决定该落在哪一步（FR-410） */}
            <Route index element={<StageRedirect />} />
            <Route path="assessment" element={<AssessmentPage />} />
            <Route path="path" element={<PathPage />} />
            <Route path="teach" element={<TeachingPage />} />
            <Route path="review" element={<ReviewPage />} />
            <Route path="archive" element={<ArchivePage />} />
          </Route>
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </main>
      <AuthPrompt />
    </div>
  )
}
