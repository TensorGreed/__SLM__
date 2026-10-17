import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ProjectListPage from './pages/ProjectListPage';
import ProjectWorkspaceLayout from './pages/ProjectWorkspaceLayout';
import ProjectOverviewPage from './pages/ProjectOverviewPage';
import ProjectStagePage from './pages/ProjectStagePage';
import ProjectSettingsPage from './pages/ProjectSettingsPage';
import LegacyRouteRedirect from './components/layout/LegacyRouteRedirect';
import ProjectDataStudioPage from './pages/ProjectDataStudioPage';
import ProjectDomainPage from './pages/ProjectDomainPage';
import ProjectDomainPacksPage from './pages/ProjectDomainPacksPage';
import ProjectDomainProfilesPage from './pages/ProjectDomainProfilesPage';
import ProjectWorkflowPage from './pages/ProjectWorkflowPage';
import ProjectRecipesPage from './pages/ProjectRecipesPage';
import ProjectRecipePickerPage from './pages/ProjectRecipePickerPage';
import ProjectTrainingConfigPage from './pages/ProjectTrainingConfigPage';
import ProjectEvalComparePage from './pages/ProjectEvalComparePage';
import ProjectPlaygroundPage from './pages/ProjectPlaygroundPage';
import ProjectModelsPage from './pages/ProjectModelsPage';
import ProjectAdapterStudioPage from './pages/ProjectAdapterStudioPage';
import ProjectAutopilotPage from './pages/ProjectAutopilotPage';
import ProjectManifestPage from './pages/ProjectManifestPage';
import ProjectDeploymentsPage from './pages/ProjectDeploymentsPage';
import ProjectObservabilityPage from './pages/ProjectObservabilityPage';
import ProjectExtensionStudioPage from './pages/ProjectExtensionStudioPage';
import ProjectAnnotatePage from './pages/ProjectAnnotatePage';
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
              <Route index element={<ProjectOverviewPage />} />
              <Route path="data" element={<ProjectStagePage destination="data" />} />
              <Route path="train" element={<ProjectStagePage destination="train" />} />
              <Route path="results" element={<ProjectStagePage destination="results" />} />
              <Route path="ship" element={<ProjectStagePage destination="ship" />} />
              <Route path="settings" element={<ProjectSettingsPage />} />
              {/* Retired URLs (Epic U phase U1): /guide, /wizard, /pipeline/<tab>. */}
              <Route path="guide" element={<LegacyRouteRedirect />} />
              <Route path="wizard" element={<LegacyRouteRedirect />} />
              <Route path="pipeline" element={<LegacyRouteRedirect />} />
              <Route path="pipeline/:tabKey" element={<LegacyRouteRedirect />} />
              <Route path="data-studio" element={<ProjectDataStudioPage />} />
              <Route path="workflow" element={<ProjectWorkflowPage />} />
              <Route path="domain" element={<ProjectDomainPage />} />
              <Route path="domain/packs" element={<ProjectDomainPacksPage />} />
              <Route path="domain/profiles" element={<ProjectDomainProfilesPage />} />
              <Route path="recipes" element={<ProjectRecipesPage />} />
              <Route path="recipe-picker" element={<ProjectRecipePickerPage />} />
              <Route path="training-config" element={<ProjectTrainingConfigPage />} />
              <Route path="eval/compare" element={<ProjectEvalComparePage />} />
              <Route path="models" element={<ProjectModelsPage />} />
              <Route path="adapter-studio" element={<ProjectAdapterStudioPage />} />
              <Route path="autopilot" element={<ProjectAutopilotPage />} />
              <Route path="manifest" element={<ProjectManifestPage />} />
              <Route path="deployments" element={<ProjectDeploymentsPage />} />
              <Route path="observability" element={<ProjectObservabilityPage />} />
              <Route path="extensions" element={<ProjectExtensionStudioPage />} />
              <Route path="annotate" element={<ProjectAnnotatePage />} />
              <Route path="annotate/:job_id" element={<ProjectAnnotatePage />} />
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
