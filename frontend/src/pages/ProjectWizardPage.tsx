import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import api from '../api/client';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectWizardPage.css';

interface AutopilotPlan {
  profile?: string;
  title?: string;
  description?: string;
  config?: Record<string, unknown>;
  changes?: Array<{ field: string; from?: unknown; to?: unknown; reason?: string }>;
  estimated_vram_risk?: string;
  estimated_vram_score?: number;
  estimated_vram_note?: string | null;
  preflight?: AutopilotPreflight;
  estimate?: {
    estimated_seconds?: number;
    estimated_cost?: number;
    unit?: string;
    confidence_score?: number;
    metric_source?: string;
    provenance?: string;
    source?: string;
    note?: string | null;
    labels?: { speed: string; quality: string; cost: string };
  };
}

interface AutopilotIntentClarification {
  required?: boolean;
  confidence_band?: string;
  reason?: string | null;
  matched_keywords?: string[];
  questions?: string[];
  suggested_intent_examples?: string[];
  rewrite_suggestions?: AutopilotIntentRewriteSuggestion[];
}

interface AutopilotIntentRewriteSuggestion {
  id?: string;
  label?: string;
  rewritten_intent?: string;
  reason?: string;
  recommended?: boolean;
}

interface AutopilotAutoFix {
  id?: string;
  label?: string;
  description?: string;
  navigate_to?: string;
}

interface AutopilotDatasetReadiness {
  ready?: boolean;
  prepared_train_exists?: boolean;
  prepared_train_path?: string;
  prepared_row_count?: number;
  blockers?: string[];
  warnings?: string[];
  hints?: string[];
  auto_fixes?: AutopilotAutoFix[];
}

interface AutopilotLaunchGuardrails {
  can_run?: boolean;
  blockers?: string[];
  warnings?: string[];
  one_click_fix_available?: boolean;
}

interface AutopilotPreflight {
  ok?: boolean;
  errors?: string[];
  warnings?: string[];
  hints?: string[];
}

interface AutopilotTargetCompatibility {
  compatible?: boolean;
  reasons?: string[];
  warnings?: string[];
  target?: {
    id?: string;
    name?: string;
    description?: string;
  } | null;
  model_metadata?: {
    model_id?: string;
    parameters_billions?: number | null;
    parameters_source?: string;
    estimated_min_vram_gb?: number | null;
    estimated_ideal_vram_gb?: number | null;
    source?: string;
  } | null;
}

interface AutopilotIntentResolveResponse {
  project_id: number;
  intent: string;
  resolved_target_device?: string;
  plans?: AutopilotPlan[];
  recommended_profile?: string;
  intent_clarification?: AutopilotIntentClarification;
  dataset_readiness?: AutopilotDatasetReadiness;
  guardrails?: AutopilotLaunchGuardrails;
  target_compatibility?: AutopilotTargetCompatibility | null;
}

interface AutopilotDecisionLogEntry {
  step?: string;
  status?: string;
  summary?: string;
  changed?: boolean;
  safe?: boolean;
  blocker?: boolean;
  why?: string | null;
  fixes?: Array<{
    label?: string;
    description?: string;
    reason_code?: string;
    one_click_available?: boolean;
  }>;
}

interface AutopilotV2OrchestrationResponse {
  project_id: number;
  dry_run: boolean;
  strict_mode: boolean;
  intent: string;
  effective_target_profile_id: string;
  resolved_target_device: string;
  selected_profile?: string | null;
  guardrails?: AutopilotLaunchGuardrails & {
    reason_codes?: string[];
    unblock_actions?: Array<{
      label?: string;
      description?: string;
      reason_code?: string;
      one_click_available?: boolean;
    }>;
  };
  readiness?: {
    status?: string;
    checks?: Array<{
      name?: string;
      id?: string;
      message?: string;
      fix?: string;
      status?: string;
    }>;
  };
  decision_log?: AutopilotDecisionLogEntry[];
  repairs?: {
    intent_rewrite?: {
      applied?: boolean;
      original_intent?: string | null;
      rewritten_intent?: string | null;
      source?: string | null;
    };
    dataset_auto_prepare?: {
      attempted?: boolean;
      succeeded?: boolean;
      method?: string | null;
      error?: string | null;
      raw_documents?: {
        attempted?: boolean;
        accepted_before?: number;
        pending_before?: number;
        accepted_after?: number;
        processed_count?: number;
        failed_count?: number;
      } | null;
    } | null;
    target_fallback?: {
      applied?: boolean;
      from_target_profile_id?: string | null;
      to_target_profile_id?: string | null;
      reason?: string | null;
    };
    profile_autotune?: {
      applied?: boolean;
      from_profile?: string | null;
      to_profile?: string | null;
      reason?: string | null;
    };
  };
  plan_v2?: AutopilotIntentResolveResponse;
  experiment?: {
    id?: number;
    name?: string;
    status?: string;
    base_model?: string;
    training_mode?: string;
  } | null;
  started?: boolean;
  start_result?: Record<string, unknown> | null;
  start_error?: string | null;
}

interface ExperimentStatusResponse {
  experiment_id?: number;
  status?: string;
  final_train_loss?: number | null;
  final_eval_loss?: number | null;
  total_steps?: number | null;
  checkpoints?: Array<Record<string, unknown>>;
}

interface IngestionDocumentSummary {
  id?: number;
  status?: string;
}

type RemoteSourceTab = 'huggingface' | 'kaggle' | 'url';

interface RemoteImportQueueResponse {
  status?: string;
  report_path?: string;
  source_type?: string;
  identifier?: string;
  task_id?: string;
}

interface RemoteImportStatusResponse {
  status?: string;
  result?: {
    samples_ingested?: number;
    source_type?: string;
    identifier?: string;
  };
  result_visible_in_api_db?: boolean;
  warning?: string;
  error?: string;
}

type TargetDevice = 'mobile' | 'laptop' | 'server';
type PlanEstimateSource = 'measured' | 'estimated' | 'simulated';

const TARGET_PROFILE_DEVICE_MAP: Record<string, TargetDevice> = {
  mobile_cpu: 'mobile',
  browser_webgpu: 'mobile',
  edge_gpu: 'laptop',
  vllm_server: 'server',
};

const PLAN_ESTIMATE_BADGE_CLASS: Record<PlanEstimateSource, string> = {
  measured: 'badge-success',
  estimated: 'badge-warning',
  simulated: 'badge-warning',
};

const PLAN_ESTIMATE_BADGE_LABEL: Record<PlanEstimateSource, string> = {
  measured: 'Measured',
  estimated: 'Estimated',
  simulated: 'Simulated',
};

function normalizePlanEstimateSource(value: unknown): PlanEstimateSource | null {
  const token = String(value || '').trim().toLowerCase();
  if (!token) return null;
  if (token === 'measured') return 'measured';
  if (token === 'simulated') return 'simulated';
  return 'estimated';
}

function planEstimateSource(plan: AutopilotPlan): PlanEstimateSource {
  const estimate = plan.estimate;
  const explicit = normalizePlanEstimateSource(estimate?.metric_source)
    || normalizePlanEstimateSource(estimate?.provenance)
    || normalizePlanEstimateSource(estimate?.source);
  return explicit || 'estimated';
}

function formatPlanEstimateMinutes(seconds: unknown, source: PlanEstimateSource): string {
  const parsed = Number(seconds);
  if (!Number.isFinite(parsed) || parsed <= 0) return 'n/a';
  const minutes = Math.max(1, Math.round(parsed / 60));
  const prefix = source === 'measured' ? '' : '~';
  return `${prefix}${minutes}m`;
}

function formatPlanEstimateCost(cost: unknown, unit: unknown, source: PlanEstimateSource): string {
  const parsed = Number(cost);
  const unitLabel = String(unit || '').trim();
  if (!Number.isFinite(parsed)) return 'n/a';
  const prefix = source === 'measured' ? '' : '~';
  const numeric = Number.isInteger(parsed) ? String(parsed) : parsed.toFixed(2);
  return `${prefix}${numeric}${unitLabel ? ` ${unitLabel}` : ''}`;
}

function planEstimateHelpText(plan: AutopilotPlan, source: PlanEstimateSource): string {
  const estimate = plan.estimate || {};
  const note = String(estimate.note || '').trim();
  const confidence = Number(estimate.confidence_score);
  const confidenceText = Number.isFinite(confidence)
    ? ` Confidence: ${(confidence * 100).toFixed(0)}%.`
    : '';
  if (source === 'measured') {
    const base = 'Time/cost come from measured historical runs on comparable configs.';
    return `${base}${confidenceText}${note ? ` ${note}` : ''}`.trim();
  }
  if (source === 'simulated') {
    const base = 'Time/cost are simulated planning values, not measured from real runs.';
    return `${base}${confidenceText}${note ? ` ${note}` : ''}`.trim();
  }
  const base = 'Time/cost are heuristic estimates from dataset size and target profile, not measured runs.';
  return `${base}${confidenceText}${note ? ` ${note}` : ''}`.trim();
}

function decisionStatusClass(status: unknown): string {
  const token = String(status || '').trim().toLowerCase();
  if (token === 'ready' || token === 'completed' || token === 'applied' || token === 'active') return 'badge-success';
  if (token === 'blocked' || token === 'failed') return 'badge-error';
  if (token === 'skipped' || token === 'inactive') return 'badge-warning';
  return 'badge-info';
}

function decisionStepLabel(step: unknown): string {
  const token = String(step || '').trim();
  if (!token) return 'step';
  return token.replace(/_/g, ' ');
}

export default function ProjectWizardPage() {
  const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
  const navigate = useNavigate();

  const [currentStep, setCurrentStep] = useState(1);
  const [intentText, setIntentText] = useState('');
  const [runNameOverride, setRunNameOverride] = useState('');
  const [targetProfileId, setTargetProfileId] = useState('vllm_server');
  const [targetCatalog, setTargetCatalog] = useState<any[]>([]);
  const [targetLoading, setTargetLoading] = useState(false);
  const [targetSaving, setTargetSaving] = useState(false);
  const [targetError, setTargetError] = useState('');
  const [acknowledgeIntentClarification, setAcknowledgeIntentClarification] = useState(false);
  const [availableVramGb, setAvailableVramGb] = useState('8');
  const [baseModelOverride, setBaseModelOverride] = useState('');

  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState('');
  const [planResponse, setPlanResponse] = useState<AutopilotIntentResolveResponse | null>(null);
  const [planOrchestrationResponse, setPlanOrchestrationResponse] = useState<AutopilotV2OrchestrationResponse | null>(null);

  const [launchLoading, setLaunchLoading] = useState(false);
  const [launchError, setLaunchError] = useState('');
  const [launchResponse, setLaunchResponse] = useState<AutopilotV2OrchestrationResponse | null>(null);

  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState('');
  const [statusResponse, setStatusResponse] = useState<ExperimentStatusResponse | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [intakeLoading, setIntakeLoading] = useState(false);
  const [intakeError, setIntakeError] = useState('');
  const [intakeStatus, setIntakeStatus] = useState('');
  const [ingestionStats, setIngestionStats] = useState({ total: 0, accepted: 0, pending: 0 });
  const [remoteSource, setRemoteSource] = useState<RemoteSourceTab>('huggingface');
  const [remoteIdentifier, setRemoteIdentifier] = useState('');
  const [remoteSplit, setRemoteSplit] = useState('train');
  const [remoteConfigName, setRemoteConfigName] = useState('');
  const [remoteMaxSamples, setRemoteMaxSamples] = useState('');
  const [remoteHfToken, setRemoteHfToken] = useState('');
  const [remoteKaggleUsername, setRemoteKaggleUsername] = useState('');
  const [remoteKaggleKey, setRemoteKaggleKey] = useState('');
  const [remoteImportLoading, setRemoteImportLoading] = useState(false);
  const [remoteImportError, setRemoteImportError] = useState('');
  const [remoteImportStatus, setRemoteImportStatus] = useState('');
  const [remoteImportReportPath, setRemoteImportReportPath] = useState<string | null>(null);

  const [selectedIntentRewrite, setSelectedIntentRewrite] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('balanced');

  useEffect(() => {
    const fetchCatalog = async () => {
      setTargetLoading(true);
      try {
        const res = await api.get('/targets/catalog');
        setTargetCatalog(res.data || []);
      } catch (err: any) {
        setTargetError('Failed to load target catalog.');
      } finally {
        setTargetLoading(false);
      }
    };
    void fetchCatalog();
  }, []);

  const stepItems = useMemo(
    () => [
      { num: 1, label: 'Select Target' },
      { num: 2, label: 'Describe Goal' },
      { num: 3, label: 'Review Safe Plan' },
      { num: 4, label: 'Monitor Training' },
      { num: 5, label: 'Chat with Model' },
    ],
    [],
  );

  const parsedVram = Number.parseFloat(availableVramGb);
  const resolvedTargetDevice = useMemo<TargetDevice>(() => {
    const selectedTarget = targetCatalog.find((target) => String(target?.id || '') === targetProfileId);
    const catalogDevice = String(selectedTarget?.device_class || '').trim().toLowerCase();
    if (catalogDevice === 'mobile' || catalogDevice === 'laptop' || catalogDevice === 'server') {
      return catalogDevice;
    }
    return TARGET_PROFILE_DEVICE_MAP[targetProfileId] || 'laptop';
  }, [targetCatalog, targetProfileId]);
  const experimentId = Number(launchResponse?.experiment?.id || 0);
  const latestStatus = String(statusResponse?.status || launchResponse?.experiment?.status || '').toLowerCase();
  const clarificationRequired = Boolean(planResponse?.intent_clarification?.required);
  const hasSelectedRewrite = selectedIntentRewrite.trim().length >= 3;
  const resolvedGuardrails = planOrchestrationResponse?.guardrails || planResponse?.guardrails;
  const launchGuardrailsPass = resolvedGuardrails?.can_run !== false;
  const guardrailBlockers = Array.isArray(resolvedGuardrails?.blockers)
    ? resolvedGuardrails?.blockers || []
    : [];
  const canLaunchWithAutoPrep = !launchGuardrailsPass && guardrailBlockers.length > 0 && guardrailBlockers.every((blocker) => {
    const token = String(blocker || '').toLowerCase();
    return token.includes('dataset')
      || token.includes('prepared')
      || token.includes('ingest')
      || token.includes('split');
  });
  const canLaunchFromPlan = (launchGuardrailsPass || canLaunchWithAutoPrep)
    && (!clarificationRequired || acknowledgeIntentClarification || hasSelectedRewrite);

  const decisionLogRows = Array.isArray(planOrchestrationResponse?.decision_log)
    ? planOrchestrationResponse?.decision_log || []
    : [];
  const launchDecisionLog = Array.isArray(launchResponse?.decision_log)
    ? launchResponse?.decision_log || []
    : [];

  const fetchIngestionStats = useCallback(async () => {
    try {
      const res = await api.get<IngestionDocumentSummary[]>(
        `/projects/${projectId}/ingestion/documents`,
      );
      const docs = Array.isArray(res.data) ? res.data : [];
      let accepted = 0;
      let pending = 0;
      docs.forEach((doc) => {
        const token = String(doc?.status || '').trim().toLowerCase();
        if (token === 'accepted') accepted += 1;
        if (token === 'pending' || token === 'processing') pending += 1;
      });
      setIngestionStats({ total: docs.length, accepted, pending });
    } catch {
      setIngestionStats((prev) => prev);
    }
  }, [projectId]);

  const uploadAndProcessWizardFiles = async (files: FileList | File[]) => {
    const rows = Array.from(files || []);
    if (rows.length === 0) return;
    setIntakeLoading(true);
    setIntakeError('');
    setIntakeStatus('');
    try {
      const formData = new FormData();
      rows.forEach((file) => formData.append('files', file));
      const uploadRes = await api.post<{
        uploaded?: number;
        errors?: Array<{ filename?: string; error?: string }>;
        documents?: Array<{ id?: number }>;
      }>(
        `/projects/${projectId}/ingestion/upload-batch`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      );
      const uploaded = Number(uploadRes.data?.uploaded || 0);
      const documents = Array.isArray(uploadRes.data?.documents) ? uploadRes.data?.documents || [] : [];
      const processResponses = await Promise.all(
        documents.map(async (doc) => {
          const id = Number(doc?.id || 0);
          if (!Number.isFinite(id) || id <= 0) return false;
          try {
            await api.post(`/projects/${projectId}/ingestion/documents/${id}/process`);
            return true;
          } catch {
            return false;
          }
        }),
      );
      const processed = processResponses.filter(Boolean).length;
      const errorCount = Array.isArray(uploadRes.data?.errors) ? uploadRes.data?.errors?.length || 0 : 0;
      setIntakeStatus(
        `Uploaded ${uploaded} file(s), processed ${processed}.`
        + (errorCount > 0 ? ` ${errorCount} upload error(s).` : ''),
      );
    } catch (err: any) {
      setIntakeError(err?.response?.data?.detail || 'Failed to upload/process files.');
    } finally {
      setIntakeLoading(false);
      void fetchIngestionStats();
    }
  };

  const extractErrorMessage = useCallback((error: unknown): string => {
    if (typeof error === 'object' && error !== null) {
      const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      if (typeof detail === 'string' && detail.trim()) {
        return detail;
      }
      if (Array.isArray(detail)) {
        const joined = detail
          .map((item) => {
            if (typeof item === 'string') return item;
            if (typeof item === 'object' && item !== null) {
              const msg = (item as { msg?: unknown }).msg;
              if (typeof msg === 'string') return msg;
            }
            return '';
          })
          .filter((item) => item)
          .join('; ');
        if (joined) return joined;
      }
    }
    if (error instanceof Error && error.message.trim()) {
      return error.message;
    }
    return 'Operation failed';
  }, []);

  const queueRemoteImportFromWizard = async () => {
    const identifier = remoteIdentifier.trim();
    if (!identifier) {
      setRemoteImportError('Enter a dataset identifier or URL first.');
      return;
    }
    setRemoteImportLoading(true);
    setRemoteImportError('');
    setRemoteImportStatus('');

    let parsedMaxSamples: number | null = null;
    if (remoteMaxSamples.trim()) {
      const parsed = Number(remoteMaxSamples);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        setRemoteImportLoading(false);
        setRemoteImportError('Max samples must be a positive number or empty.');
        return;
      }
      parsedMaxSamples = Math.floor(parsed);
    }

    const payload: {
      source_type: RemoteSourceTab;
      identifier: string;
      split: string;
      config_name?: string | null;
      max_samples: number | null;
      use_saved_secrets: boolean;
      hf_token?: string;
      kaggle_username?: string;
      kaggle_key?: string;
    } = {
      source_type: remoteSource,
      identifier,
      split: remoteSource === 'huggingface' ? (remoteSplit.trim() || 'train') : 'train',
      config_name: remoteSource === 'huggingface' ? (remoteConfigName.trim() || null) : null,
      max_samples: parsedMaxSamples,
      use_saved_secrets: true,
    };
    if (remoteSource === 'huggingface' && remoteHfToken.trim()) {
      payload.hf_token = remoteHfToken.trim();
    }
    if (remoteSource === 'kaggle') {
      if (remoteKaggleUsername.trim()) payload.kaggle_username = remoteKaggleUsername.trim();
      if (remoteKaggleKey.trim()) payload.kaggle_key = remoteKaggleKey.trim();
    }

    try {
      const res = await api.post<RemoteImportQueueResponse>(
        `/projects/${projectId}/ingestion/import-remote/queue`,
        payload,
      );
      const reportPath = String(res.data?.report_path || '').trim();
      if (!reportPath) {
        setRemoteImportLoading(false);
        setRemoteImportError('Import queued but no report path was returned.');
        return;
      }
      setRemoteImportReportPath(reportPath);
      setRemoteImportStatus(`Import queued for ${remoteSource}. Waiting for completion...`);
    } catch (error) {
      setRemoteImportLoading(false);
      setRemoteImportError(`Import failed: ${extractErrorMessage(error)}`);
    }
  };

  const saveTargetAndContinue = async () => {
    setTargetError('');
    setTargetSaving(true);
    try {
      await api.put(`/projects/${projectId}`, {
        target_profile_id: targetProfileId,
      });
      setCurrentStep(2);
    } catch (err: any) {
      setTargetError('Failed to save target selection. Please try again.');
    } finally {
      setTargetSaving(false);
    }
  };

  useEffect(() => {
    if (currentStep !== 2) return;
    void fetchIngestionStats();
  }, [currentStep, fetchIngestionStats]);

  useEffect(() => {
    if (!remoteImportReportPath) return;
    const interval = window.setInterval(async () => {
      try {
        const statusRes = await api.get<RemoteImportStatusResponse>(
          `/projects/${projectId}/ingestion/imports/status`,
          { params: { report_path: remoteImportReportPath } },
        );
        const payload = statusRes.data || {};
        const state = String(payload.status || '').trim().toLowerCase();
        if (state === 'completed') {
          const ingested = Number(payload.result?.samples_ingested || 0);
          const source = String(payload.result?.source_type || remoteSource);
          const identifier = String(payload.result?.identifier || remoteIdentifier.trim());
          if (payload.result_visible_in_api_db === false) {
            setRemoteImportError(
              payload.warning
                || 'Import completed in worker but API cannot see the document. Verify API/worker DB settings.',
            );
          } else {
            setRemoteImportStatus(`Imported ${ingested} samples from ${source}:${identifier}.`);
          }
          setRemoteImportLoading(false);
          setRemoteImportReportPath(null);
          void fetchIngestionStats();
        } else if (state === 'failed') {
          setRemoteImportError(String(payload.error || 'Remote import failed.'));
          setRemoteImportLoading(false);
          setRemoteImportReportPath(null);
        }
      } catch (error) {
        setRemoteImportError(`Import polling failed: ${extractErrorMessage(error)}`);
        setRemoteImportLoading(false);
        setRemoteImportReportPath(null);
      }
    }, 2000);
    return () => window.clearInterval(interval);
  }, [
    extractErrorMessage,
    fetchIngestionStats,
    projectId,
    remoteIdentifier,
    remoteImportReportPath,
    remoteSource,
  ]);

  const resolveSafePlan = async (intentOverride?: string) => {
    const baseIntent = typeof intentOverride === 'string' ? intentOverride : intentText;
    const trimmedIntent = baseIntent.trim();
    if (trimmedIntent.length < 3) {
      setPlanError('Describe your goal in plain language (at least 3 characters).');
      return;
    }
    setPlanLoading(true);
    setPlanError('');
    setLaunchError('');
    setAcknowledgeIntentClarification(false);
    if (typeof intentOverride === 'string') {
      setIntentText(trimmedIntent);
      setSelectedIntentRewrite(trimmedIntent);
    } else {
      setSelectedIntentRewrite('');
    }
    try {
      const res = await api.post<AutopilotV2OrchestrationResponse>(
        `/projects/${projectId}/training/autopilot/v2/orchestrate`,
        {
          intent: trimmedIntent,
          target_profile_id: targetProfileId,
          target_device: resolvedTargetDevice,
          primary_language: 'english',
          available_vram_gb: Number.isFinite(parsedVram) && parsedVram > 0 ? parsedVram : undefined,
          base_model: baseModelOverride.trim() || undefined,
          auto_prepare_data: true,
          auto_apply_rewrite: true,
          intent_rewrite: hasSelectedRewrite ? selectedIntentRewrite.trim() : undefined,
          plan_profile: selectedProfile,
          dry_run: true,
        },
      );
      const payload = res.data || null;
      setPlanOrchestrationResponse(payload);
      const planPayload = payload?.plan_v2 || null;
      const mergedPlan: AutopilotIntentResolveResponse | null = planPayload
        ? {
          ...planPayload,
          project_id: Number(planPayload.project_id || payload?.project_id || projectId),
          intent: String(planPayload.intent || payload?.intent || trimmedIntent),
          resolved_target_device: String(
            planPayload.resolved_target_device
            || payload?.resolved_target_device
            || resolvedTargetDevice,
          ),
          guardrails: payload?.guardrails || planPayload.guardrails,
        }
        : null;
      setPlanResponse(mergedPlan);
      if (payload?.selected_profile) {
        setSelectedProfile(payload.selected_profile);
      } else if (mergedPlan?.recommended_profile) {
        setSelectedProfile(mergedPlan.recommended_profile);
      }
      setLaunchResponse(null);
      setCurrentStep(3);
    } catch (err: any) {
      setPlanOrchestrationResponse(null);
      setPlanResponse(null);
      setPlanError(err?.response?.data?.detail || 'Failed to resolve an autopilot plan.');
    } finally {
      setPlanLoading(false);
    }
  };

  const launchOneClickRun = async () => {
    const trimmedIntent = intentText.trim();
    if (trimmedIntent.length < 3) {
      setLaunchError('Describe your goal first.');
      return;
    }
    setCurrentStep(3);
    setLaunchLoading(true);
    setLaunchError('');
    try {
      const res = await api.post<AutopilotV2OrchestrationResponse>(
        `/projects/${projectId}/training/autopilot/v2/orchestrate/run`,
        {
          intent: trimmedIntent,
          target_profile_id: targetProfileId,
          target_device: resolvedTargetDevice,
          primary_language: 'english',
          available_vram_gb: Number.isFinite(parsedVram) && parsedVram > 0 ? parsedVram : undefined,
          base_model: baseModelOverride.trim() || undefined,
          auto_prepare_data: true,
          auto_apply_rewrite: true,
          intent_rewrite: hasSelectedRewrite ? selectedIntentRewrite.trim() : undefined,
          run_name: runNameOverride.trim() || undefined,
          plan_profile: selectedProfile,
          dry_run: false,
        },
      );
      const payload = res.data || null;
      setLaunchResponse(payload);
      setPlanOrchestrationResponse(payload);
      const planPayload = payload?.plan_v2 || null;
      if (planPayload) {
        setPlanResponse({
          ...planPayload,
          guardrails: payload?.guardrails || planPayload.guardrails,
          intent: String(planPayload.intent || payload?.intent || trimmedIntent),
          resolved_target_device: String(
            planPayload.resolved_target_device
            || payload?.resolved_target_device
            || resolvedTargetDevice,
          ),
        });
      }
      if (payload?.selected_profile) {
        setSelectedProfile(payload.selected_profile);
      }
      setCurrentStep(4);
      if (!payload?.started) {
        setLaunchError(String(payload?.start_error || 'Autopilot completed orchestration, but training did not start.'));
      }
    } catch (err: any) {
      setLaunchResponse(null);
      setLaunchError(err?.response?.data?.detail || 'Failed to launch one-click run.');
    } finally {
      setLaunchLoading(false);
    }
  };

  const refreshExperimentStatus = async () => {
    if (!Number.isFinite(experimentId) || experimentId <= 0) {
      return;
    }
    setStatusLoading(true);
    setStatusError('');
    try {
      const res = await api.get<ExperimentStatusResponse>(
        `/projects/${projectId}/training/experiments/${experimentId}/status`,
      );
      const payload = res.data || null;
      setStatusResponse(payload);
      const status = String(payload?.status || '').toLowerCase();
      if (status === 'completed') {
        setTrainingProgress(100);
        setCurrentStep(6);
      } else if (status === 'failed' || status === 'cancelled') {
        setTrainingProgress(100);
      } else if (status === 'running') {
        const checkpointCount = Array.isArray(payload?.checkpoints) ? payload.checkpoints.length : 0;
        const nextProgress = Math.min(95, 55 + (checkpointCount * 8));
        setTrainingProgress((prev) => Math.max(prev, nextProgress));
      } else if (status === 'pending') {
        setTrainingProgress((prev) => Math.max(prev, 20));
      }
    } catch (err: any) {
      setStatusError(err?.response?.data?.detail || 'Failed to refresh training status.');
    } finally {
      setStatusLoading(false);
    }
  };

  useEffect(() => {
    if (currentStep < 4) {
      return;
    }
    if (!Number.isFinite(experimentId) || experimentId <= 0) {
      return;
    }
    void refreshExperimentStatus();
    const interval = window.setInterval(() => {
      void refreshExperimentStatus();
    }, 4000);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep, experimentId, projectId]);

  return (
    <div className="wizard-page animate-fade-in">
      <div className="wizard-shell card">
        <div className="wizard-header">
          <div>
            <h2>Autopilot Wizard</h2>
            <p>Describe your goal in plain language. We pick safe defaults and launch training for you.</p>
          </div>
          <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/pipeline`)}>
            Advanced Mode
          </button>
        </div>

        <div className="wizard-stepper">
          {stepItems.map((item) => (
            <div key={item.num} className={`wizard-step-indicator ${currentStep >= item.num ? 'active' : ''}`}>
              <div className={`step-circle ${currentStep > item.num ? 'complete' : ''}`}>
                {currentStep > item.num ? '✓' : item.num}
              </div>
              <div className="step-label">{item.label}</div>
            </div>
          ))}
        </div>

        {currentStep === 1 && (
          <section className="wizard-section">
            <h3>Where will this model run?</h3>
            <p className="wizard-muted">Pick your target deployment environment to get optimized training recommendations.</p>
            {targetLoading ? (
              <div className="wizard-loading">Loading targets...</div>
            ) : targetError ? (
              <div className="wizard-error">{targetError}</div>
            ) : (
              <div className="target-grid">
                {targetCatalog.map((target) => (
                  <div
                    key={target.id}
                    className={`target-card ${targetProfileId === target.id ? 'selected' : ''}`}
                    onClick={() => setTargetProfileId(target.id)}
                  >
                    <h4>{target.name}</h4>
                    <p>{target.description}</p>
                    <div className="constraints">
                      {target.constraints.max_parameters_billions && (
                        <div>Max Size: {target.constraints.max_parameters_billions}B parameters</div>
                      )}
                      {target.constraints.min_vram_gb && (
                        <div>Min VRAM: {target.constraints.min_vram_gb}GB</div>
                      )}
                      {target.constraints.preferred_formats?.length > 0 && (
                        <div>Preferred: {target.constraints.preferred_formats.join(', ').toUpperCase()}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="wizard-actions wizard-actions-bottom">
              <button
                className="btn btn-primary"
                onClick={() => void saveTargetAndContinue()}
                disabled={targetSaving || targetLoading}
              >
                {targetSaving ? 'Saving Target...' : 'Next: Describe Goal'}
              </button>
            </div>
          </section>
        )}

        {currentStep === 2 && (
          <section className="wizard-section">
            <h3>What do you want your model to do?</h3>
            <p className="wizard-muted">
              Example:
              {' '}
              "I want a model that reads support tickets and drafts short answers."
            </p>
            <div className="wizard-upload-box wizard-data-intake">
              <div className="wizard-data-intake-copy">
                <strong>Dataset intake</strong>
                <span>Upload files now. Wizard will parse them, then one-click will auto-prepare train/val/test.</span>
                <span>
                  Raw docs: {ingestionStats.total} total, {ingestionStats.accepted} accepted, {ingestionStats.pending} pending
                </span>
              </div>
              <label className="btn btn-secondary" htmlFor="wizard-intake-upload">
                {intakeLoading ? 'Uploading...' : 'Upload Dataset Files'}
              </label>
              <input
                id="wizard-intake-upload"
                type="file"
                multiple
                disabled={intakeLoading}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    void uploadAndProcessWizardFiles(e.target.files);
                  }
                  e.target.value = '';
                }}
              />
              <button
                className="btn btn-secondary"
                onClick={() => navigate(`/project/${projectId}/pipeline/ingestion`)}
              >
                Remote/Advanced Intake
              </button>
              {intakeStatus && <span>{intakeStatus}</span>}
              {intakeError && <div className="wizard-error">{intakeError}</div>}
            </div>
            <div className="wizard-upload-box wizard-data-intake">
              <div className="wizard-data-intake-copy">
                <strong>Remote dataset import</strong>
                <span>Import directly from HuggingFace, Kaggle, or URL without leaving the wizard.</span>
              </div>
              <div className="wizard-remote-controls">
                <div className="wizard-remote-row">
                  <select
                    className="input"
                    value={remoteSource}
                    onChange={(e) => setRemoteSource(e.target.value as RemoteSourceTab)}
                    disabled={remoteImportLoading}
                  >
                    <option value="huggingface">HuggingFace</option>
                    <option value="kaggle">Kaggle</option>
                    <option value="url">URL</option>
                  </select>
                  <input
                    className="input"
                    value={remoteIdentifier}
                    onChange={(e) => setRemoteIdentifier(e.target.value)}
                    placeholder={
                      remoteSource === 'huggingface'
                        ? 'Dataset id (e.g. tatsu-lab/alpaca)'
                        : remoteSource === 'kaggle'
                          ? 'Dataset id (e.g. owner/dataset-name)'
                          : 'Direct file URL'
                    }
                    disabled={remoteImportLoading}
                  />
                </div>
                {remoteSource === 'huggingface' && (
                  <div className="wizard-remote-row">
                    <input
                      className="input"
                      value={remoteSplit}
                      onChange={(e) => setRemoteSplit(e.target.value)}
                      placeholder="Split (train)"
                      disabled={remoteImportLoading}
                    />
                    <input
                      className="input"
                      value={remoteConfigName}
                      onChange={(e) => setRemoteConfigName(e.target.value)}
                      placeholder="Config name (optional)"
                      disabled={remoteImportLoading}
                    />
                    <input
                      className="input"
                      value={remoteHfToken}
                      onChange={(e) => setRemoteHfToken(e.target.value)}
                      placeholder="HF token (optional)"
                      disabled={remoteImportLoading}
                    />
                  </div>
                )}
                {remoteSource === 'kaggle' && (
                  <div className="wizard-remote-row">
                    <input
                      className="input"
                      value={remoteKaggleUsername}
                      onChange={(e) => setRemoteKaggleUsername(e.target.value)}
                      placeholder="Kaggle username (optional)"
                      disabled={remoteImportLoading}
                    />
                    <input
                      className="input"
                      value={remoteKaggleKey}
                      onChange={(e) => setRemoteKaggleKey(e.target.value)}
                      placeholder="Kaggle key (optional)"
                      disabled={remoteImportLoading}
                    />
                  </div>
                )}
                <div className="wizard-remote-row">
                  <input
                    className="input"
                    value={remoteMaxSamples}
                    onChange={(e) => setRemoteMaxSamples(e.target.value)}
                    placeholder="Max samples (optional)"
                    disabled={remoteImportLoading}
                  />
                  <button
                    className="btn btn-secondary"
                    onClick={() => void queueRemoteImportFromWizard()}
                    disabled={remoteImportLoading}
                  >
                    {remoteImportLoading ? 'Importing...' : 'Import Remote Dataset'}
                  </button>
                </div>
              </div>
              {remoteImportStatus && <span>{remoteImportStatus}</span>}
              {remoteImportError && <div className="wizard-error">{remoteImportError}</div>}
            </div>
            <label className="form-label" htmlFor="wizard-intent-input">Plain-language goal</label>
            <textarea
              id="wizard-intent-input"
              className="input"
              value={intentText}
              onChange={(e) => {
                setIntentText(e.target.value);
                setSelectedIntentRewrite('');
              }}
              placeholder="Describe your use case in one or two sentences..."
              rows={4}
            />
            <div className="wizard-param-row">
              <label className="form-label">Available VRAM (optional)</label>
              <input
                className="input"
                value={availableVramGb}
                onChange={(e) => setAvailableVramGb(e.target.value)}
                placeholder="8"
              />
            </div>
            <div className="wizard-param-row">
              <label className="form-label">Base model (optional override)</label>
              <input
                className="input"
                value={baseModelOverride}
                onChange={(e) => setBaseModelOverride(e.target.value)}
                placeholder="Qwen/Qwen2.5-1.5B-Instruct"
              />
            </div>
            <div className="wizard-param-row">
              <label className="form-label">Run name (optional override)</label>
              <input
                className="input"
                value={runNameOverride}
                onChange={(e) => setRunNameOverride(e.target.value)}
                placeholder="Autopilot - Support Q&A Assistant"
              />
            </div>
            {planError && <div className="wizard-error">{planError}</div>}
            <div className="wizard-actions wizard-actions-bottom">
              <button className="btn btn-secondary" onClick={() => setCurrentStep(1)}>
                Back
              </button>
              <button className="btn btn-primary" onClick={() => void resolveSafePlan()} disabled={planLoading}>
                {planLoading ? 'Building Safe Plan...' : 'Build Safe Plan'}
              </button>
            </div>
          </section>
        )}

        {currentStep === 3 && (
          <section className="wizard-section">
            <h3>Choose your path</h3>
            <p className="wizard-muted">We've prepared 3 ways to reach your goal. Pick one to launch.</p>
            <div className="wizard-panel">
              <div className="wizard-intent-summary">
                <strong>Intent:</strong> {intentText.trim()}
                {baseModelOverride.trim() && (
                  <div>
                    <strong>Base model override:</strong> {baseModelOverride.trim()}
                  </div>
                )}
                {hasSelectedRewrite && (
                  <div className="wizard-rewrite-badge">
                    <span>Rewritten for clarity:</span> {selectedIntentRewrite.trim()}
                  </div>
                )}
              </div>

              {planOrchestrationResponse && (
                <div className={`wizard-upload-box ${planOrchestrationResponse.strict_mode ? 'wizard-warning' : ''}`}>
                  <div>
                    <strong>Autopilot v2 mode:</strong>
                    {' '}
                    {planOrchestrationResponse.strict_mode ? 'Strict mode enabled' : 'Standard mode'}
                  </div>
                  <div>
                    Effective target profile:
                    {' '}
                    {planOrchestrationResponse.effective_target_profile_id}
                  </div>
                </div>
              )}

              {planOrchestrationResponse?.readiness && (
                <div className="wizard-upload-box">
                  <div>
                    <strong>Runtime readiness:</strong>
                    {' '}
                    {String(planOrchestrationResponse.readiness.status || 'unknown').toUpperCase()}
                  </div>
                  {Array.isArray(planOrchestrationResponse.readiness.checks)
                    && planOrchestrationResponse.readiness.checks.length > 0 && (
                      <ul className="wizard-filter-list">
                        {planOrchestrationResponse.readiness.checks.slice(0, 4).map((check) => (
                          <li key={String(check.id || check.name || 'readiness-check')}>
                            {check.name || check.id || 'check'}
                            {check.fix ? `: ${check.fix}` : check.message ? `: ${check.message}` : ''}
                          </li>
                        ))}
                      </ul>
                    )}
                </div>
              )}

              <div className="autopilot-plan-grid">
                {(planResponse?.plans || []).map((plan) => {
                  const estimateSource = planEstimateSource(plan);
                  const estimateHelp = planEstimateHelpText(plan, estimateSource);
                  return (
                    <div
                      key={plan.profile}
                      className={`autopilot-plan-card ${selectedProfile === plan.profile ? 'selected' : ''}`}
                      onClick={() => setSelectedProfile(plan.profile || 'balanced')}
                    >
                      <div className="plan-header">
                        <h4>{plan.title}</h4>
                        <div className="plan-header-badges">
                          {plan.profile === planResponse?.recommended_profile && (
                            <span className="badge badge-success">Recommended</span>
                          )}
                          <span
                            className={`badge ${PLAN_ESTIMATE_BADGE_CLASS[estimateSource]} plan-provenance-badge`}
                            title={estimateHelp}
                          >
                            {PLAN_ESTIMATE_BADGE_LABEL[estimateSource]}
                          </span>
                        </div>
                      </div>
                      <p className="plan-description">{plan.description}</p>
                      <div className="plan-estimate">
                        <div>Time: {formatPlanEstimateMinutes(plan.estimate?.estimated_seconds, estimateSource)}</div>
                        <div>Cost: {formatPlanEstimateCost(plan.estimate?.estimated_cost, plan.estimate?.unit, estimateSource)}</div>
                      </div>
                      <div className="plan-estimate-note" title={estimateHelp}>{estimateHelp}</div>
                      <div className="plan-labels">
                        <span className="label-badge speed">{plan.estimate?.labels?.speed} Speed</span>
                        <span className="label-badge quality">{plan.estimate?.labels?.quality} Quality</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {planResponse?.dataset_readiness && (
                <div className={`wizard-upload-box ${planResponse.dataset_readiness.ready ? '' : 'wizard-error'}`}>
                  <div>
                    <strong>Dataset readiness:</strong>
                    {' '}
                    {planResponse.dataset_readiness.ready ? 'READY' : 'BLOCKED'}
                  </div>
                  <div>
                    Prepared rows:
                    {' '}
                    {Number(planResponse.dataset_readiness.prepared_row_count || 0)}
                  </div>
                  {Array.isArray(planResponse.dataset_readiness.blockers)
                    && planResponse.dataset_readiness.blockers.length > 0 && (
                      <div className="wizard-blockers">
                        <strong>Blockers:</strong>
                        <ul className="wizard-filter-list">
                          {planResponse.dataset_readiness.blockers.map((b) => <li key={b}>{b}</li>)}
                        </ul>
                      </div>
                    )}
                </div>
              )}

              {planResponse?.target_compatibility && (
                <div
                  className={`wizard-upload-box ${
                    planResponse.target_compatibility.compatible === false ? 'wizard-error' : 'wizard-warning'
                  }`}
                >
                  <div>
                    <strong>Target compatibility:</strong>
                    {' '}
                    {planResponse.target_compatibility.compatible === false ? 'BLOCKED' : 'OK'}
                  </div>
                  <div>
                    Target:
                    {' '}
                    {String(
                      planResponse.target_compatibility.target?.name
                      || planResponse.target_compatibility.target?.id
                      || targetProfileId,
                    )}
                  </div>
                  {typeof planResponse.target_compatibility.model_metadata?.parameters_billions === 'number' && (
                    <div>
                      Model size:
                      {' '}
                      {planResponse.target_compatibility.model_metadata.parameters_billions}
                      B
                    </div>
                  )}
                  {typeof planResponse.target_compatibility.model_metadata?.estimated_min_vram_gb === 'number' && (
                    <div>
                      Estimated min VRAM:
                      {' '}
                      {planResponse.target_compatibility.model_metadata.estimated_min_vram_gb}
                      GB
                    </div>
                  )}
                  {Array.isArray(planResponse.target_compatibility.reasons)
                    && planResponse.target_compatibility.reasons.length > 0 && (
                      <div className="wizard-blockers">
                        <strong>Blockers:</strong>
                        <ul className="wizard-filter-list">
                          {planResponse.target_compatibility.reasons.map((reason) => <li key={reason}>{reason}</li>)}
                        </ul>
                      </div>
                    )}
                  {Array.isArray(planResponse.target_compatibility.warnings)
                    && planResponse.target_compatibility.warnings.length > 0 && (
                      <div className="wizard-blockers">
                        <strong>Warnings:</strong>
                        <ul className="wizard-filter-list">
                          {planResponse.target_compatibility.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                        </ul>
                      </div>
                    )}
                </div>
              )}

              {planResponse?.intent_clarification?.required && (
                <div className="wizard-upload-box wizard-warning">
                  <div>
                    <strong>Clarification recommended</strong>
                    {' '}
                    ({planResponse.intent_clarification.confidence_band || 'low'} confidence)
                  </div>
                  {Array.isArray(planResponse.intent_clarification.rewrite_suggestions)
                    && planResponse.intent_clarification.rewrite_suggestions.length > 0 && (
                      <div className="wizard-rewrites-inline">
                        <strong>Try these rewrites:</strong>
                        <div className="wizard-actions">
                          {planResponse.intent_clarification.rewrite_suggestions.slice(0, 2).map((suggestion) => (
                            <button
                              key={suggestion.id}
                              className="btn btn-secondary btn-sm"
                              onClick={() => void resolveSafePlan(suggestion.rewritten_intent)}
                            >
                              {suggestion.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  <label className="form-label form-label-inline">
                    <input
                      type="checkbox"
                      checked={acknowledgeIntentClarification}
                      onChange={(e) => setAcknowledgeIntentClarification(e.target.checked)}
                    />
                    I reviewed this and still want to launch with this intent
                  </label>
                </div>
              )}

              {Array.isArray(resolvedGuardrails?.warnings)
                && resolvedGuardrails?.warnings?.length > 0 && (
                  <div className="wizard-upload-box wizard-warning">
                    <strong>Launch warnings</strong>
                    <ul className="wizard-filter-list">
                      {(resolvedGuardrails?.warnings || []).slice(0, 5).map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              {canLaunchWithAutoPrep && (
                <div className="wizard-upload-box wizard-warning">
                  <strong>Auto-fix on launch</strong>
                  <span>Dataset blockers were detected. One-click will auto-process and auto-prepare data before training.</span>
                </div>
              )}

              {Array.isArray(planResponse?.dataset_readiness?.auto_fixes)
                && planResponse?.dataset_readiness?.auto_fixes?.length > 0 && (
                  <div className="wizard-panel wizard-fix-panel">
                    <strong>Suggested fixes</strong>
                    <div className="wizard-actions wizard-actions-bottom">
                      {planResponse?.dataset_readiness?.auto_fixes?.slice(0, 3).map((fix) => (
                        <button
                          key={String(fix.id || fix.label || 'fix')}
                          className={`btn ${fix.navigate_to ? 'btn-primary' : 'btn-secondary'}`}
                          onClick={() => {
                            const path = String(fix.navigate_to || '').trim();
                            if (path) navigate(path);
                          }}
                        >
                          {fix.label || 'Open Fix'}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

              {decisionLogRows.length > 0 && (
                <div className="wizard-upload-box">
                  <strong>Autopilot decision log</strong>
                  <div className="wizard-decision-log">
                    {decisionLogRows.map((entry, idx) => (
                      <div key={`${entry.step || 'step'}-${idx}`} className="wizard-decision-entry">
                        <div className="wizard-decision-header">
                          <span>{decisionStepLabel(entry.step)}</span>
                          <span className={`badge ${decisionStatusClass(entry.status)}`}>
                            {String(entry.status || 'unknown').toUpperCase()}
                          </span>
                        </div>
                        <div>{entry.summary || 'No summary available.'}</div>
                        {entry.why && (
                          <div className="wizard-muted">Why: {entry.why}</div>
                        )}
                        {Array.isArray(entry.fixes) && entry.fixes.length > 0 && (
                          <ul className="wizard-filter-list">
                            {entry.fixes.slice(0, 2).map((fix, fixIdx) => (
                              <li key={`${entry.step || idx}-fix-${fixIdx}`}>
                                {fix.label || 'Fix'}{fix.description ? `: ${fix.description}` : ''}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div className="wizard-actions wizard-actions-bottom">
              <button className="btn btn-secondary" onClick={() => setCurrentStep(2)}>Back</button>
              <button
                className="btn btn-primary"
                onClick={() => void launchOneClickRun()}
                disabled={launchLoading || planLoading || !canLaunchFromPlan}
              >
                {launchLoading ? 'Launching...' : 'One-Click Run'}
              </button>
            </div>
            {!canLaunchFromPlan && (
              <div className="wizard-error-summary">
                Please resolve blockers before launching.
              </div>
            )}
          </section>
        )}

        {currentStep === 4 && (
          <section className="wizard-section">
            <h3>Launch Result</h3>
            <p className="wizard-muted">Autopilot v2 orchestration completed and attempted training launch.</p>
            <div className="wizard-panel">
              <div>
                <strong>Experiment:</strong>
                {' '}
                {launchResponse?.experiment?.name || '-'}
                {' '}
                (#{launchResponse?.experiment?.id || '-'})
              </div>
              <div><strong>Status:</strong> {launchResponse?.experiment?.status || '-'}</div>
              <div><strong>Started:</strong> {launchResponse?.started ? 'yes' : 'no'}</div>
              <div>
                <strong>Effective target:</strong>
                {' '}
                {launchResponse?.effective_target_profile_id || targetProfileId}
              </div>
              {launchResponse?.repairs?.intent_rewrite?.applied && (
                <div>
                  <strong>Applied intent rewrite:</strong>
                  {' '}
                  {String(launchResponse?.repairs?.intent_rewrite?.rewritten_intent || '')}
                </div>
              )}
              {(launchError || launchResponse?.start_error) && (
                <div className="wizard-error">
                  {launchError || launchResponse?.start_error}
                </div>
              )}
              {launchResponse?.repairs?.dataset_auto_prepare?.attempted && (
                <div>
                  <strong>Auto data prep:</strong>
                  {' '}
                  {launchResponse.repairs.dataset_auto_prepare.succeeded ? 'applied' : 'not applied'}
                  {launchResponse.repairs.dataset_auto_prepare.method
                    ? ` (${launchResponse.repairs.dataset_auto_prepare.method})`
                    : ''}
                  {launchResponse.repairs.dataset_auto_prepare.raw_documents && (
                    <span>
                      {` · docs processed ${Number(launchResponse.repairs.dataset_auto_prepare.raw_documents.processed_count || 0)} (accepted: ${Number(launchResponse.repairs.dataset_auto_prepare.raw_documents.accepted_after || 0)})`}
                    </span>
                  )}
                </div>
              )}
              {launchResponse?.repairs?.target_fallback?.applied && (
                <div>
                  <strong>Target fallback:</strong>
                  {' '}
                  {launchResponse.repairs.target_fallback.from_target_profile_id}
                  {' -> '}
                  {launchResponse.repairs.target_fallback.to_target_profile_id}
                </div>
              )}
              {launchResponse?.repairs?.profile_autotune?.applied && (
                <div>
                  <strong>Profile auto-tune:</strong>
                  {' '}
                  {launchResponse.repairs.profile_autotune.from_profile}
                  {' -> '}
                  {launchResponse.repairs.profile_autotune.to_profile}
                </div>
              )}
              {launchDecisionLog.length > 0 && (
                <div className="wizard-decision-log" style={{ marginTop: 10 }}>
                  {launchDecisionLog.map((entry, idx) => (
                    <div key={`${entry.step || 'step'}-${idx}`} className="wizard-decision-entry">
                      <div className="wizard-decision-header">
                        <span>{decisionStepLabel(entry.step)}</span>
                        <span className={`badge ${decisionStatusClass(entry.status)}`}>
                          {String(entry.status || 'unknown').toUpperCase()}
                        </span>
                      </div>
                      <div>{entry.summary || 'No summary available.'}</div>
                      {entry.why && <div className="wizard-muted">Why: {entry.why}</div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="wizard-actions wizard-actions-bottom">
              <button className="btn btn-secondary" onClick={() => setCurrentStep(3)}>Back</button>
              <button
                className="btn btn-primary"
                onClick={() => setCurrentStep(5)}
                disabled={!launchResponse?.experiment?.id}
              >
                Monitor Training
              </button>
            </div>
          </section>
        )}

        {currentStep === 5 && (
          <section className="wizard-section">
            <h3>Training Progress</h3>
            <p className="wizard-muted">
              Experiment
              {' '}
              #{launchResponse?.experiment?.id || '-'}
              {' '}
              is
              {' '}
              {latestStatus || 'starting'}.
            </p>
            <div className="wizard-progress">
              <div className="wizard-progress-fill" style={{ width: `${trainingProgress}%` }} />
            </div>
            <div className="wizard-progress-label">{trainingProgress}% complete</div>
            {statusError && <div className="wizard-error">{statusError}</div>}
            <div className="wizard-actions wizard-actions-bottom">
              <button className="btn btn-secondary" onClick={() => void refreshExperimentStatus()} disabled={statusLoading}>
                {statusLoading ? 'Refreshing...' : 'Refresh Status'}
              </button>
              <button className="btn btn-primary" onClick={() => navigate(`/project/${projectId}/training`)}>
                Open Training Panel
              </button>
            </div>
          </section>
        )}

        {currentStep === 6 && (
          <section className="wizard-section">
            <h3>Model Ready</h3>
            <p className="wizard-muted">Training completed. You can now test and export your model.</p>
            <div className="wizard-actions">
              <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/playground`)}>
                Chat with Model
              </button>
              <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/pipeline/export`)}>
                Export Model
              </button>
              <button className="btn btn-primary" onClick={() => navigate(`/project/${projectId}/training`)}>
                Fine-Tune Settings
              </button>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
