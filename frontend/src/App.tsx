import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ProjectListPage from './pages/ProjectListPage';
import ProjectWorkspaceLayout from './pages/ProjectWorkspaceLayout';
import ProjectOverviewPage from './pages/ProjectOverviewPage';
import ProjectShipPage from './pages/ProjectShipPage';
import ProjectSettingsPage from './pages/ProjectSettingsPage';
import LegacyRouteRedirect from './components/layout/LegacyRouteRedirect';
import ProjectDataPage from './pages/ProjectDataPage';
import ProjectTrainPage from './pages/ProjectTrainPage';
import ProjectResultsPage from './pages/ProjectResultsPage';
import ProjectManifestPage from './pages/ProjectManifestPage';
import ProjectAnnotatePage from './pages/ProjectAnnotatePage';
import PipelinesPage from './pages/PipelinesPage';
import PipelineDetailPage from './pages/PipelineDetailPage';
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
            <Route path="/pipelines" element={<PipelinesPage />} />
            <Route path="/pipelines/:id" element={<PipelineDetailPage />} />
            <Route path="/manifest/import" element={<ProjectManifestPage />} />
            <Route path="/project/:id" element={<ProjectWorkspaceLayout />}>
              <Route index element={<ProjectOverviewPage />} />
              <Route path="data" element={<ProjectDataPage />} />
              <Route path="train" element={<ProjectTrainPage />} />
              <Route path="results" element={<ProjectResultsPage />} />
              <Route path="ship" element={<ProjectShipPage />} />
              <Route path="settings" element={<ProjectSettingsPage />} />
              {/* The long tail lives in Settings (Epic U phase U6). */}
              <Route path="workflow" element={<LegacyRouteRedirect />} />
              <Route path="domain" element={<LegacyRouteRedirect />} />
              <Route path="domain/packs" element={<LegacyRouteRedirect />} />
              <Route path="domain/profiles" element={<LegacyRouteRedirect />} />
              <Route path="autopilot" element={<LegacyRouteRedirect />} />
              <Route path="extensions" element={<LegacyRouteRedirect />} />
              {/* Retired URLs (Epic U phase U1): /guide, /wizard, /pipeline/<tab>. */}
              <Route path="guide" element={<LegacyRouteRedirect />} />
              <Route path="wizard" element={<LegacyRouteRedirect />} />
              <Route path="pipeline" element={<LegacyRouteRedirect />} />
              <Route path="pipeline/:tabKey" element={<LegacyRouteRedirect />} />
              {/* Folded into Data (Epic U phase U4); the labeler keeps its own page. */}
              <Route path="data-studio" element={<LegacyRouteRedirect />} />
              <Route path="annotate" element={<LegacyRouteRedirect />} />
              {/* Folded into Train (Epic U phase U2). */}
              <Route path="recipes" element={<LegacyRouteRedirect />} />
              <Route path="recipe-picker" element={<LegacyRouteRedirect />} />
              <Route path="training-config" element={<LegacyRouteRedirect />} />
              <Route path="models" element={<LegacyRouteRedirect />} />
              <Route path="adapter-studio" element={<LegacyRouteRedirect />} />
              {/* Folded into Results (Epic U phase U3). */}
              <Route path="eval/compare" element={<LegacyRouteRedirect />} />
              <Route path="playground" element={<LegacyRouteRedirect />} />
              {/* Folded into Ship (Epic U phase U5). */}
              <Route path="deployments" element={<LegacyRouteRedirect />} />
              <Route path="observability" element={<LegacyRouteRedirect />} />
              <Route path="manifest" element={<LegacyRouteRedirect />} />
              <Route path="annotate/:job_id" element={<ProjectAnnotatePage />} />
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
