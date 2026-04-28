import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ProjectListPage from './pages/ProjectListPage';
import ProjectWorkspaceLayout from './pages/ProjectWorkspaceLayout';
import ProjectPipelinePage from './pages/ProjectPipelinePage';
import ProjectDomainPage from './pages/ProjectDomainPage';
import ProjectDomainPacksPage from './pages/ProjectDomainPacksPage';
import ProjectDomainProfilesPage from './pages/ProjectDomainProfilesPage';
import ProjectWorkflowPage from './pages/ProjectWorkflowPage';
import ProjectRecipesPage from './pages/ProjectRecipesPage';
import ProjectTrainingConfigPage from './pages/ProjectTrainingConfigPage';
import ProjectPlaygroundPage from './pages/ProjectPlaygroundPage';
import ProjectGuidePage from './pages/ProjectGuidePage';
import ProjectWizardPage from './pages/ProjectWizardPage';
import ProjectModelsPage from './pages/ProjectModelsPage';
import ProjectAdapterStudioPage from './pages/ProjectAdapterStudioPage';
import ProjectAutopilotPage from './pages/ProjectAutopilotPage';
import ProjectManifestPage from './pages/ProjectManifestPage';
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
            <Route path="/manifest/import" element={<ProjectManifestPage />} />
            <Route path="/project/:id" element={<ProjectWorkspaceLayout />}>
              <Route index element={<Navigate to="pipeline/data" replace />} />
              <Route path="guide" element={<ProjectGuidePage />} />
              <Route path="wizard" element={<ProjectWizardPage />} />
              <Route path="pipeline" element={<Navigate to="data" replace />} />
              <Route path="pipeline/:tabKey" element={<ProjectPipelinePage />} />
              <Route path="workflow" element={<ProjectWorkflowPage />} />
              <Route path="domain" element={<ProjectDomainPage />} />
              <Route path="domain/packs" element={<ProjectDomainPacksPage />} />
              <Route path="domain/profiles" element={<ProjectDomainProfilesPage />} />
              <Route path="recipes" element={<ProjectRecipesPage />} />
              <Route path="training-config" element={<ProjectTrainingConfigPage />} />
              <Route path="models" element={<ProjectModelsPage />} />
              <Route path="adapter-studio" element={<ProjectAdapterStudioPage />} />
              <Route path="autopilot" element={<ProjectAutopilotPage />} />
              <Route path="manifest" element={<ProjectManifestPage />} />
              <Route path="playground" element={<ProjectPlaygroundPage />} />
            </Route>
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
