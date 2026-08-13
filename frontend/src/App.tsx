import { Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { LogOut, User } from 'lucide-react';

import Dashboard from './pages/Dashboard';
import PracticeSetup from './pages/PracticeSetup';
import PracticeSession from './pages/PracticeSession';
import SessionReview from './pages/SessionReview';
import Reports from './pages/Reports';
import AdminDashboard from './pages/AdminDashboard';
import Certification from './pages/Certification';
import NotificationBell from './components/NotificationBell';

function App() {
  const [studentId, setStudentId] = useState<string | null>(localStorage.getItem('student_id'));
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (!studentId && location.pathname !== '/login') {
      navigate('/login');
    }
  }, [studentId, navigate, location.pathname]);

  const handleLogin = (id: string) => {
    localStorage.setItem('student_id', id);
    setStudentId(id);
    navigate('/');
  };

  const handleLogout = () => {
    localStorage.removeItem('student_id');
    setStudentId(null);
    navigate('/login');
  };

  if (!studentId) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="app-container slide-down">
      <nav className="navbar">
        <div className="nav-brand">SignApp</div>
        <div className="nav-links">
          <Link to="/" className={location.pathname === '/' ? 'active' : ''}>Dashboard</Link>
          <Link to="/practice" className={location.pathname.startsWith('/practice') ? 'active' : ''}>Practice</Link>
          <Link to="/certification" className={location.pathname === '/certification' ? 'active' : ''}>Certifications</Link>
          <Link to="/reports" className={location.pathname === '/reports' ? 'active' : ''}>Reports</Link>
          {(studentId === 'admin' || studentId === 'instructor') && (
            <Link to="/admin" className={location.pathname === '/admin' ? 'active' : ''}>Admin</Link>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
            <User size={18} />
            <span>{studentId}</span>
          </div>
          <NotificationBell studentId={studentId} />
          <button className="secondary" onClick={handleLogout} style={{ padding: '0.5rem', borderRadius: '50%' }}>
            <LogOut size={18} />
          </button>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard studentId={studentId} />} />
          <Route path="/practice" element={<PracticeSetup studentId={studentId} />} />
          <Route path="/practice/session/:id" element={<PracticeSession studentId={studentId} />} />
          <Route path="/practice/review/:id" element={<SessionReview studentId={studentId} />} />
          <Route path="/reports" element={<Reports studentId={studentId} />} />
          <Route path="/certification" element={<Certification studentId={studentId} />} />
          <Route path="/admin" element={<AdminDashboard />} />
        </Routes>
      </main>
    </div>
  );
}

function Login({ onLogin }: { onLogin: (id: string) => void }) {
  const [id, setId] = useState('');

  return (
    <div className="auth-container">
      <div className="card auth-card slide-down">
        <h2 className="card-title" style={{ textAlign: 'center', fontSize: '1.75rem', marginBottom: '2rem' }}>Welcome to SignApp</h2>
        <form onSubmit={(e) => { e.preventDefault(); if (id) onLogin(id); }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>Student ID</label>
            <input
              type="text"
              placeholder="e.g. learner_001"
              value={id}
              onChange={(e) => setId(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="primary" style={{ width: '100%' }}>Login</button>
        </form>
      </div>
    </div>
  );
}

export default App;
