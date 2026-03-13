import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import api from '../api/client';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectWizardPage.css';

interface AutopilotPlan {
  preset_id?: string;
  preset_label?: string;
  preset_description?: string;
  task_profile?: string;
  confidence?: number;
  matched_keywords?: string[];
  run_name_suggestion?: string;
  user_friendly_plan?: string[];
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
  can_one_click_run?: boolean;
  blockers?: string[];
  warnings?: string[];
}

interface AutopilotModelRecommendation {
  model_id?: string;
  match_score?: number;
  match_reasons?: string[];
}

interface AutopilotPreflight {
  ok?: boolean;
  errors?: string[];
  warnings?: string[];
  hints?: string[];
}

interface AutopilotIntentResolveResponse {
  project_id: number;
  plan?: AutopilotPlan;
  intent_clarification?: AutopilotIntentClarification;
  dataset_readiness?: AutopilotDatasetReadiness;
  launch_guardrails?: AutopilotLaunchGuardrails;
  safe_training_config?: Record<string, unknown>;
  model_recommendation?: AutopilotModelRecommendation | null;
  preflight?: AutopilotPreflight;
}

interface AutopilotOneClickRunResponse extends AutopilotIntentResolveResponse {
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
  applied_intent_rewrite?: {
    applied?: boolean;
    original_intent?: string | null;
    rewritten_intent?: string | null;
    source?: string | null;
  } | null;
}

interface ExperimentStatusResponse {
  experiment_id?: number;
  status?: string;
  final_train_loss?: number | null;
  final_eval_loss?: number | null;
  total_steps?: number | null;
  checkpoints?: Array<Record<string, unknown>>;
}

export default function ProjectWizardPage() {
  const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
  const navigate = useNavigate();

  const [currentStep, setCurrentStep] = useState(1);
  const [intentText, setIntentText] = useState('');
  const [targetDevice, setTargetDevice] = useState<'mobile' | 'laptop' | 'server'>('laptop');
  const [availableVramGb, setAvailableVramGb] = useState('8');
  const [runNameOverride, setRunNameOverride] = useState('');
  const [acknowledgeIntentClarification, setAcknowledgeIntentClarification] = useState(false);
  const [selectedIntentRewrite, setSelectedIntentRewrite] = useState('');

  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState('');
  const [planResponse, setPlanResponse] = useState<AutopilotIntentResolveResponse | null>(null);

  const [launchLoading, setLaunchLoading] = useState(false);
  const [launchError, setLaunchError] = useState('');
  const [launchResponse, setLaunchResponse] = useState<AutopilotOneClickRunResponse | null>(null);

  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState('');
  const [statusResponse, setStatusResponse] = useState<ExperimentStatusResponse | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);

  const stepItems = useMemo(
    () => [
      { num: 1, label: 'Describe Goal' },
      { num: 2, label: 'Review Safe Plan' },
      { num: 3, label: 'One-Click Launch' },
      { num: 4, label: 'Monitor Training' },
      { num: 5, label: 'Chat with Model' },
    ],
    [],
  );

  const parsedVram = Number.parseFloat(availableVramGb);
  const experimentId = Number(launchResponse?.experiment?.id || 0);
  const latestStatus = String(statusResponse?.status || launchResponse?.experiment?.status || '').toLowerCase();
  const clarificationRequired = Boolean(planResponse?.intent_clarification?.required);
  const hasSelectedRewrite = selectedIntentRewrite.trim().length >= 3;
  const launchGuardrailsPass = planResponse?.launch_guardrails?.can_one_click_run !== false;
  const canLaunchFromPlan = launchGuardrailsPass
    && (!clarificationRequired || acknowledgeIntentClarification || hasSelectedRewrite);

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
      const res = await api.post<AutopilotIntentResolveResponse>(
        `/projects/${projectId}/training/autopilot/intent-resolve`,
        {
          intent: trimmedIntent,
          target_device: targetDevice,
          primary_language: 'english',
          available_vram_gb: Number.isFinite(parsedVram) && parsedVram > 0 ? parsedVram : undefined,
        },
      );
      setPlanResponse(res.data || null);
      setCurrentStep(2);
    } catch (err: any) {
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
      const res = await api.post<AutopilotOneClickRunResponse>(
        `/projects/${projectId}/training/autopilot/one-click-run`,
        {
          intent: trimmedIntent,
          target_device: targetDevice,
          primary_language: 'english',
          available_vram_gb: Number.isFinite(parsedVram) && parsedVram > 0 ? parsedVram : undefined,
          auto_apply_rewrite: true,
          intent_rewrite: hasSelectedRewrite ? selectedIntentRewrite.trim() : undefined,
          run_name: runNameOverride.trim() || undefined,
        },
      );
      const payload = res.data || null;
      setLaunchResponse(payload);
      if (payload?.started) {
        setCurrentStep(4);
      } else {
        setLaunchError(String(payload?.start_error || 'Autopilot created a run, but could not start training.'));
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
        setCurrentStep(5);
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
            <h3>What do you want your model to do?</h3>
            <p className="wizard-muted">
              Example:
              {' '}
              "I want a model that reads support tickets and drafts short answers."
            </p>
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
              <label className="form-label">Target hardware</label>
              <select
                className="input"
                value={targetDevice}
                onChange={(e) => setTargetDevice(e.target.value as 'mobile' | 'laptop' | 'server')}
              >
                <option value="mobile">Mobile / Edge</option>
                <option value="laptop">Laptop / Single GPU</option>
                <option value="server">Server GPU</option>
              </select>
            </div>
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
              <button className="btn btn-primary" onClick={() => void resolveSafePlan()} disabled={planLoading}>
                {planLoading ? 'Building Safe Plan...' : 'Build Safe Plan'}
              </button>
            </div>
          </section>
        )}

        {currentStep === 2 && (
          <section className="wizard-section">
            <h3>Safe plan is ready</h3>
            <p className="wizard-muted">Review, then launch with one click.</p>
            <div className="wizard-panel">
              <div><strong>Intent:</strong> {intentText.trim()}</div>
              {hasSelectedRewrite && (
                <div>
                  <strong>Selected rewrite:</strong>
                  {' '}
                  {selectedIntentRewrite.trim()}
                </div>
              )}
              <div><strong>Preset:</strong> {planResponse?.plan?.preset_label || '-'}</div>
              <div><strong>Why this preset:</strong> {planResponse?.plan?.preset_description || '-'}</div>
              <div><strong>Task profile:</strong> {planResponse?.plan?.task_profile || '-'}</div>
              <div>
                <strong>Confidence:</strong>
                {' '}
                {Number.isFinite(Number(planResponse?.plan?.confidence))
                  ? `${Math.round(Number(planResponse?.plan?.confidence) * 100)}%`
                  : 'n/a'}
              </div>
              <div>
                <strong>Recommended model:</strong>
                {' '}
                {planResponse?.model_recommendation?.model_id || '-'}
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
                      <div>
                        Blockers:
                        {' '}
                        {planResponse.dataset_readiness.blockers.slice(0, 2).join(' | ')}
                      </div>
                    )}
                  {Array.isArray(planResponse.dataset_readiness.warnings)
                    && planResponse.dataset_readiness.warnings.length > 0 && (
                      <div>
                        Warnings:
                        {' '}
                        {planResponse.dataset_readiness.warnings.slice(0, 2).join(' | ')}
                      </div>
                    )}
                </div>
              )}
              {planResponse?.intent_clarification?.required && (
                <div className="wizard-upload-box wizard-error">
                  <div>
                    <strong>Clarification recommended</strong>
                    {' '}
                    ({planResponse.intent_clarification.confidence_band || 'low'} confidence)
                  </div>
                  {planResponse.intent_clarification.reason && (
                    <div>{planResponse.intent_clarification.reason}</div>
                  )}
                  {Array.isArray(planResponse.intent_clarification.questions)
                    && planResponse.intent_clarification.questions.length > 0 && (
                      <ul className="wizard-filter-list">
                        {planResponse.intent_clarification.questions.slice(0, 3).map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    )}
                  {Array.isArray(planResponse.intent_clarification.suggested_intent_examples)
                    && planResponse.intent_clarification.suggested_intent_examples.length > 0 && (
                      <details>
                        <summary>See better intent examples</summary>
                        <ul className="wizard-filter-list">
                          {planResponse.intent_clarification.suggested_intent_examples.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </details>
                    )}
                  {Array.isArray(planResponse.intent_clarification.rewrite_suggestions)
                    && planResponse.intent_clarification.rewrite_suggestions.length > 0 && (
                      <div className="wizard-panel">
                        <strong>Try one-click rewrites</strong>
                        <div className="wizard-actions wizard-actions-bottom">
                          {planResponse.intent_clarification.rewrite_suggestions.slice(0, 3).map((suggestion) => {
                            const rewritten = String(suggestion?.rewritten_intent || '').trim();
                            if (!rewritten) {
                              return null;
                            }
                            return (
                              <button
                                key={String(suggestion?.id || rewritten)}
                                className="btn btn-secondary"
                                disabled={planLoading}
                                onClick={() => {
                                  setSelectedIntentRewrite(rewritten);
                                  void resolveSafePlan(rewritten);
                                }}
                              >
                                {suggestion?.label || 'Use and Rebuild Plan'}
                              </button>
                            );
                          })}
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
              {Array.isArray(planResponse?.dataset_readiness?.auto_fixes)
                && planResponse?.dataset_readiness?.auto_fixes?.length > 0 && (
                  <div className="wizard-panel">
                    <strong>Suggested fixes</strong>
                    <div className="wizard-actions wizard-actions-bottom">
                      {planResponse?.dataset_readiness?.auto_fixes?.slice(0, 3).map((fix) => (
                        <button
                          key={String(fix.id || fix.label || 'fix')}
                          className="btn btn-secondary"
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
              {Array.isArray(planResponse?.plan?.user_friendly_plan) && planResponse?.plan?.user_friendly_plan?.length > 0 && (
                <ul className="wizard-filter-list">
                  {planResponse?.plan?.user_friendly_plan?.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              )}
              {planResponse?.preflight && (
                <div className={`wizard-upload-box ${planResponse.preflight.ok ? '' : 'wizard-error'}`}>
                  <div>
                    <strong>Preflight:</strong>
                    {' '}
                    {planResponse.preflight.ok ? 'PASS' : 'BLOCKED'}
                  </div>
                  {!planResponse.preflight.ok && Array.isArray(planResponse.preflight.errors) && planResponse.preflight.errors.length > 0 && (
                    <div>{planResponse.preflight.errors.slice(0, 2).join(' | ')}</div>
                  )}
                </div>
              )}
              <details>
                <summary>Show safe config JSON</summary>
                <pre>{JSON.stringify(planResponse?.safe_training_config || {}, null, 2)}</pre>
              </details>
            </div>
            <div className="wizard-actions wizard-actions-bottom">
              <button className="btn btn-secondary" onClick={() => setCurrentStep(1)}>Back</button>
              <button
                className="btn btn-primary"
                onClick={() => void launchOneClickRun()}
                disabled={launchLoading || planLoading || !canLaunchFromPlan}
              >
                {launchLoading ? 'Launching...' : 'One-Click Run'}
              </button>
            </div>
            {(!canLaunchFromPlan
              || (Array.isArray(planResponse?.launch_guardrails?.warnings) && planResponse?.launch_guardrails?.warnings?.length > 0)) && (
              <div className="wizard-muted">
                {Array.isArray(planResponse?.launch_guardrails?.blockers) && planResponse?.launch_guardrails?.blockers?.length > 0
                  ? `Blocked: ${planResponse?.launch_guardrails?.blockers?.slice(0, 2).join(' | ')}`
                  : ''}
                {Array.isArray(planResponse?.launch_guardrails?.warnings) && planResponse?.launch_guardrails?.warnings?.length > 0
                  ? `${Array.isArray(planResponse?.launch_guardrails?.blockers) && planResponse?.launch_guardrails?.blockers?.length > 0 ? ' • ' : ''}Warnings: ${planResponse?.launch_guardrails?.warnings?.slice(0, 2).join(' | ')}`
                  : ''}
              </div>
            )}
          </section>
        )}

        {currentStep === 3 && (
          <section className="wizard-section">
            <h3>Launch Result</h3>
            <p className="wizard-muted">We created your experiment and attempted to start training.</p>
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
              {launchResponse?.applied_intent_rewrite?.applied && (
                <div>
                  <strong>Applied intent rewrite:</strong>
                  {' '}
                  {String(launchResponse?.applied_intent_rewrite?.rewritten_intent || '')}
                </div>
              )}
              {(launchError || launchResponse?.start_error) && (
                <div className="wizard-error">
                  {launchError || launchResponse?.start_error}
                </div>
              )}
            </div>
            <div className="wizard-actions wizard-actions-bottom">
              <button className="btn btn-secondary" onClick={() => setCurrentStep(2)}>Back</button>
              <button
                className="btn btn-primary"
                onClick={() => setCurrentStep(4)}
                disabled={!launchResponse?.experiment?.id}
              >
                Monitor Training
              </button>
            </div>
          </section>
        )}

        {currentStep === 4 && (
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

        {currentStep === 5 && (
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
