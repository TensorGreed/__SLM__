import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ProjectListPage from './pages/ProjectListPage';
import ProjectDetailPage from './pages/ProjectDetailPage';
import SSOLoginPage from './pages/SSOLoginPage';
import ToastContainer from './components/shared/Toast';

function App() {
  const token = localStorage.getItem('slm_token');
  // For local dev without SSO configured, we can still fall through if we wanted,
  // but let's strictly protect the app if we are testing Phase 10.
  // In a real app we'd fetch `/api/auth/me` to verify the token validity.
  const isAuthenticated = !!token;

  return (
    <>
      <Routes>
        <Route path="/login" element={<SSOLoginPage />} />
        {isAuthenticated ? (
          <>
            <Route path="/" element={<ProjectListPage />} />
            <Route path="/project/:id" element={<ProjectDetailPage />} />
          </>
        ) : (
          <Route path="*" element={<Navigate to="/login" replace />} />
        )}
      </Routes>
      <ToastContainer />
    </>
  );
}

export default function AppWithRouter() {
  return (
    <BrowserRouter>
      <App />
    </BrowserRouter>
  );
}
