/**
 * Validation & Planning (Power tools): effective-config preview, capability
 * preflight (adapter / model / runtime / media contracts) and preflight plan
 * suggestions. Presentational — TrainingPanel owns the requests.
 */

import DatasetFitCard from './DatasetFitCard';
import type {
  TrainingEffectiveConfigResponse,
  TrainingPreflightReport,
  TrainingPreflightPlanSuggestion,
  TrainingPreflightPlanReport,
} from './trainingPanelTypes';
import type { ModelSelectionSummary, PreflightContractDetails } from './trainingPanelUtils';
import { describeWarmStartReason } from './trainingPanelUtils';

interface TrainingValidationSectionProps {
  projectId: number;
  effectivePreview: TrainingEffectiveConfigResponse | null;
  effectivePreviewLoading: boolean;
  effectivePreviewError: string;
  preflightPreview: TrainingPreflightReport | null;
  preflightPreviewLoading: boolean;
  preflightPreviewError: string;
  preflightPlan: TrainingPreflightPlanReport | null;
  preflightPlanLoading: boolean;
  preflightPlanError: string;
  preferredPlanProfile: string;
  preflightContractDetails: PreflightContractDetails;
  modelSelectionSummary: ModelSelectionSummary;
  onPreviewEffectiveConfig: () => void;
  onRunPreflightPreview: () => void;
  onRunPreflightPlan: () => void;
  onApplyPlanSuggestion: (suggestion: TrainingPreflightPlanSuggestion) => void;
}

export default function TrainingValidationSection({
  projectId,
  effectivePreview,
  effectivePreviewLoading,
  effectivePreviewError,
  preflightPreview,
  preflightPreviewLoading,
  preflightPreviewError,
  preflightPlan,
  preflightPlanLoading,
  preflightPlanError,
  preferredPlanProfile,
  preflightContractDetails,
  modelSelectionSummary,
  onPreviewEffectiveConfig,
  onRunPreflightPreview,
  onRunPreflightPlan,
  onApplyPlanSuggestion,
}: TrainingValidationSectionProps) {
  return (
      <details className="training-collapsible">
        <summary>
          <span>Validation & Planning</span>
          <small>Preflight, effective config, and suggested plans</small>
        </summary>
        <div className="training-collapsible__content">
          <div className="form-group form-group--spaced">
            <div className="form-inline-actions">
              <button
                className="btn btn-secondary"
                onClick={() => onPreviewEffectiveConfig()}
                disabled={effectivePreviewLoading}
              >
                {effectivePreviewLoading ? 'Resolving...' : 'Preview Effective Config'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => onRunPreflightPreview()}
                disabled={preflightPreviewLoading}
              >
                {preflightPreviewLoading ? 'Checking...' : 'Run Capability Preflight'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => onRunPreflightPlan()}
                disabled={preflightPlanLoading}
              >
                {preflightPlanLoading ? 'Planning...' : 'Run Preflight Plan'}
              </button>
            </div>
          </div>
          {effectivePreviewError && (
            <div className="training-alert training-alert--error">
              {effectivePreviewError}
            </div>
          )}
          {preflightPreviewError && (
            <div className="training-alert training-alert--error">
              {preflightPreviewError}
            </div>
          )}
          {preflightPlanError && (
            <div className="training-alert training-alert--error">
              {preflightPlanError}
            </div>
          )}
          {effectivePreview && (
            <div className="resolved-defaults-panel">
              <div className="resolved-defaults-panel__title">Effective Config Preview (Pre-create)</div>
              <div className="resolved-defaults-panel__kv">
                <span>Applied Pack</span>
                <strong>
                  {effectivePreview.domain_pack_applied
                    ? `${effectivePreview.domain_pack_applied} (${effectivePreview.domain_pack_source || 'unknown'})`
                    : 'none'}
                </strong>
              </div>
              <div className="resolved-defaults-panel__kv">
                <span>Applied Profile</span>
                <strong>
                  {effectivePreview.domain_profile_applied
                    ? `${effectivePreview.domain_profile_applied} (${effectivePreview.domain_profile_source || 'unknown'})`
                    : 'none'}
                </strong>
              </div>
              <div className="resolved-defaults-panel__kv">
                <span>Resolved Training Mode</span>
                <strong>{effectivePreview.resolved_training_mode || 'sft'}</strong>
              </div>
              {effectivePreview.warm_start && (
                <div className="resolved-defaults-panel__kv">
                  <span>Starting Weights</span>
                  <strong>
                    {effectivePreview.warm_start.source === 'checkpoint'
                      ? effectivePreview.warm_start.checkpoint_name ||
                        effectivePreview.warm_start.manifest?.display_name ||
                        'warm start'
                      : 'base model (cold start)'}
                    <span className="resolved-defaults-panel__warm-start-reason">
                      {describeWarmStartReason(effectivePreview.warm_start.reason)}
                    </span>
                  </strong>
                </div>
              )}
              <div className="resolved-defaults-panel__kv">
                <span>Runtime Fields Applied</span>
                <strong>
                  {effectivePreview.profile_defaults_applied && effectivePreview.profile_defaults_applied.length > 0
                    ? effectivePreview.profile_defaults_applied.join(', ')
                    : 'none'}
                </strong>
              </div>
              <div className="resolved-defaults-panel__grid">
                <div>
                  <div className="resolved-defaults-panel__subtitle">Resolved Training Config</div>
                  <pre className="resolved-defaults-panel__json">
                    {JSON.stringify(effectivePreview.resolved_training_config || {}, null, 2)}
                  </pre>
                </div>
                <div>
                  <div className="resolved-defaults-panel__subtitle">Runtime Training Defaults</div>
                  <pre className="resolved-defaults-panel__json">
                    {JSON.stringify(effectivePreview.profile_training_defaults || {}, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
          {preflightPreview && (
            <div
              className={`training-preflight-panel ${preflightPreview.ok ? 'training-preflight-panel--ok' : 'training-preflight-panel--error'
                }`}
            >
              <div className="training-preflight-panel__title-row">
                <strong>Capability Preflight</strong>
                <span className={`badge ${preflightPreview.ok ? 'badge-success' : 'badge-error'}`}>
                  {preflightPreview.ok ? 'PASS' : 'BLOCKED'}
                </span>
              </div>
              {(() => {
                const ds = (preflightPreview.capability_summary as Record<string, unknown> | undefined)?.dataset as
                  | Record<string, unknown>
                  | undefined;
                const contract = ds?.contract as Record<string, unknown> | undefined;
                if (!contract || typeof contract !== 'object') return null;
                const sampled = Number((contract as { sampled_rows?: number }).sampled_rows || 0);
                const errs = (contract as { errors?: string[] }).errors || [];
                // Only render when the contract actually ran (sampled_rows > 0) and
                // either flagged an error or has a coverage worth surfacing.
                if (sampled <= 0 && (!errs || errs.length === 0)) return null;
                return (
                  <DatasetFitCard
                    contract={contract as Parameters<typeof DatasetFitCard>[0]['contract']}
                    projectId={projectId}
                  />
                );
              })()}
              {preflightPreview.errors.length > 0 && (
                <div className="training-preflight-panel__section">
                  <div className="training-preflight-panel__section-title">Blocking Issues</div>
                  <ul className="training-preflight-panel__list">
                    {preflightPreview.errors.map((item, idx) => (
                      <li key={`preflight-error-${idx}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
              {preflightPreview.warnings.length > 0 && (
                <div className="training-preflight-panel__section">
                  <div className="training-preflight-panel__section-title">Warnings</div>
                  <ul className="training-preflight-panel__list">
                    {preflightPreview.warnings.map((item, idx) => (
                      <li key={`preflight-warning-${idx}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
              {Array.isArray(preflightPreview.hints) && preflightPreview.hints.length > 0 && (
                <div className="training-preflight-panel__section">
                  <div className="training-preflight-panel__section-title">Fix Hints</div>
                  <ul className="training-preflight-panel__list">
                    {preflightPreview.hints.map((item, idx) => (
                      <li key={`preflight-hint-${idx}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
              {modelSelectionSummary.hasAny && (
                <div className="training-preflight-panel__section">
                  <div className="training-preflight-panel__section-title">Model Selection Snapshot</div>
                  <div className="training-preflight-contract-grid">
                    <div className="training-preflight-contract-card">
                      <div className="training-preflight-contract-card__title">Recommendation Winner</div>
                      <div className="training-preflight-contract-card__row">
                        <span>Model</span>
                        <strong>{modelSelectionSummary.recommendationWinnerId || 'not available'}</strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Match Score</span>
                        <strong>
                          {Number.isFinite(Number(modelSelectionSummary.recommendationWinnerScore))
                            ? Number(modelSelectionSummary.recommendationWinnerScore).toFixed(2)
                            : 'n/a'}
                        </strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Adaptive Bias</span>
                        <strong>
                          {Number.isFinite(Number(modelSelectionSummary.recommendationWinnerAdaptiveBias))
                            ? Number(modelSelectionSummary.recommendationWinnerAdaptiveBias).toFixed(2)
                            : 'n/a'}
                        </strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Top Reason</span>
                        <strong>{modelSelectionSummary.recommendationWinnerReason || 'n/a'}</strong>
                      </div>
                    </div>
                    <div className="training-preflight-contract-card">
                      <div className="training-preflight-contract-card__title">Benchmark Winner</div>
                      <div className="training-preflight-contract-card__row">
                        <span>Model</span>
                        <strong>{modelSelectionSummary.benchmarkWinnerId || 'not available'}</strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Quality</span>
                        <strong>
                          {Number.isFinite(Number(modelSelectionSummary.benchmarkWinnerAccuracy))
                            ? `${Number(modelSelectionSummary.benchmarkWinnerAccuracy).toFixed(1)}%`
                            : 'n/a'}
                        </strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Latency</span>
                        <strong>
                          {Number.isFinite(Number(modelSelectionSummary.benchmarkWinnerLatencyMs))
                            ? `${Number(modelSelectionSummary.benchmarkWinnerLatencyMs).toFixed(1)} ms`
                            : 'n/a'}
                        </strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Throughput</span>
                        <strong>
                          {Number.isFinite(Number(modelSelectionSummary.benchmarkWinnerThroughputTps))
                            ? `${Number(modelSelectionSummary.benchmarkWinnerThroughputTps).toFixed(1)} t/s`
                            : 'n/a'}
                        </strong>
                      </div>
                      <div className="training-preflight-contract-card__row">
                        <span>Source</span>
                        <strong>
                          {modelSelectionSummary.benchmarkWinnerMode || 'real_sampled'}
                          {modelSelectionSummary.benchmarkWinnerSource
                            ? ` • ${modelSelectionSummary.benchmarkWinnerSource}`
                            : ''}
                        </strong>
                      </div>
                    </div>
                  </div>
                  <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                    {modelSelectionSummary.winnerAlignmentLabel}
                    {modelSelectionSummary.activeModelId
                      ? ` Active model (${modelSelectionSummary.activeModelId}) ${modelSelectionSummary.activeModelLabel}.`
                      : ''}
                  </div>
                </div>
              )}
              <div className="training-preflight-panel__section">
                <div className="training-preflight-panel__section-title">Capability Contract Diagnostics</div>
                <div className="training-preflight-contract-grid">
                  <div className="training-preflight-contract-card">
                    <div className="training-preflight-contract-card__title">Task + Trainer</div>
                    <div className="training-preflight-contract-card__row">
                      <span>Task Type</span>
                      <strong>{preflightContractDetails.taskType}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Training Mode</span>
                      <strong>{preflightContractDetails.trainingMode}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Trainer Backend</span>
                      <strong>{preflightContractDetails.trainerBackend}</strong>
                    </div>
                  </div>
                  <div className="training-preflight-contract-card">
                    <div className="training-preflight-contract-card__title">Runtime Contract</div>
                    <div className="training-preflight-contract-card__row">
                      <span>Runtime</span>
                      <strong>{preflightContractDetails.runtimeId}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Backend</span>
                      <strong>{preflightContractDetails.runtimeBackend}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Modalities</span>
                      <strong>
                        {preflightContractDetails.runtimeSupportedModalities.length > 0
                          ? preflightContractDetails.runtimeSupportedModalities.join(', ')
                          : 'text'}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Metadata</span>
                      <strong>
                        {preflightContractDetails.runtimeModalitiesDeclared === true
                          ? 'Declared'
                          : preflightContractDetails.runtimeModalitiesDeclared === false
                            ? 'Fallback assumption'
                            : 'Unknown'}
                        {preflightContractDetails.runtimeKnown === false ? ' • runtime unresolved' : ''}
                      </strong>
                    </div>
                  </div>
                  <div className="training-preflight-contract-card">
                    <div className="training-preflight-contract-card__title">Adapter Contract</div>
                    <div className="training-preflight-contract-card__row">
                      <span>Adapter</span>
                      <strong>{preflightContractDetails.adapterId}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Adapter Source</span>
                      <strong>{preflightContractDetails.adapterSource}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Task Profile</span>
                      <strong>
                        {preflightContractDetails.adapterTaskProfile}
                        {preflightContractDetails.adapterTaskProfileSource
                          ? ` (${preflightContractDetails.adapterTaskProfileSource})`
                          : ''}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Resolved Modality</span>
                      <strong>{preflightContractDetails.adapterModality}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Declared Profiles</span>
                      <strong>
                        {preflightContractDetails.adapterDeclaredProfiles.length > 0
                          ? preflightContractDetails.adapterDeclaredProfiles.join(', ')
                          : 'n/a'}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Preferred Tasks</span>
                      <strong>
                        {preflightContractDetails.adapterPreferredTasks.length > 0
                          ? preflightContractDetails.adapterPreferredTasks.join(', ')
                          : 'n/a'}
                      </strong>
                    </div>
                  </div>
                  <div className="training-preflight-contract-card">
                    <div className="training-preflight-contract-card__title">Model Compatibility</div>
                    <div className="training-preflight-contract-card__row">
                      <span>Model</span>
                      <strong>{preflightContractDetails.modelId}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Family</span>
                      <strong>{preflightContractDetails.modelFamily}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Architecture</span>
                      <strong>{preflightContractDetails.modelArchitecture}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Gate Status</span>
                      <strong>
                        {preflightContractDetails.modelGateOk === true
                          ? 'Pass'
                          : preflightContractDetails.modelGateOk === false
                            ? 'Blocked'
                            : 'Unknown'}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Metadata Source</span>
                      <strong>{preflightContractDetails.modelIntrospectionSource}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Supported Architectures</span>
                      <strong>
                        {preflightContractDetails.modelSupportedArchitectures.length > 0
                          ? preflightContractDetails.modelSupportedArchitectures.join(', ')
                          : 'n/a'}
                      </strong>
                    </div>
                    {preflightContractDetails.modelGateErrors.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--error">
                        {preflightContractDetails.modelGateErrors.join(' | ')}
                      </div>
                    )}
                    {preflightContractDetails.modelGateHints.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                        {preflightContractDetails.modelGateHints.join(' | ')}
                      </div>
                    )}
                  </div>
                  <div className="training-preflight-contract-card">
                    <div className="training-preflight-contract-card__title-row">
                      <div className="training-preflight-contract-card__title">Model + Dataset Modality</div>
                      <span
                        className={`badge ${preflightContractDetails.modelModalityStatus === 'blocked'
                          ? 'badge-error'
                          : preflightContractDetails.modelModalityStatus === 'warning'
                            ? 'badge-warning'
                            : preflightContractDetails.modelModalityStatus === 'pass'
                              ? 'badge-success'
                              : 'badge-info'
                          }`}
                      >
                        {preflightContractDetails.modelModalityStatus === 'blocked'
                          ? 'BLOCKED'
                          : preflightContractDetails.modelModalityStatus === 'warning'
                            ? 'WARNING'
                            : preflightContractDetails.modelModalityStatus === 'pass'
                              ? 'PASS'
                              : 'UNKNOWN'}
                      </span>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Architecture</span>
                      <strong>{preflightContractDetails.modelModalityArchitecture}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Adapter Modality</span>
                      <strong>{preflightContractDetails.modelModalityAdapterModality}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Supported Modalities</span>
                      <strong>{preflightContractDetails.modelModalitySupportedModalities.join(', ')}</strong>
                    </div>
                    {preflightContractDetails.modelModalityErrors.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--error">
                        {preflightContractDetails.modelModalityErrors.join(' | ')}
                      </div>
                    )}
                    {preflightContractDetails.modelModalityWarnings.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--warning">
                        {preflightContractDetails.modelModalityWarnings.join(' | ')}
                      </div>
                    )}
                    {preflightContractDetails.modelModalityHints.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                        {preflightContractDetails.modelModalityHints.join(' | ')}
                      </div>
                    )}
                  </div>
                  <div className="training-preflight-contract-card">
                    <div className="training-preflight-contract-card__title-row">
                      <div className="training-preflight-contract-card__title">Media Asset Contract</div>
                      <span
                        className={`badge ${preflightContractDetails.mediaContractStatus === 'blocked'
                          ? 'badge-error'
                          : preflightContractDetails.mediaContractStatus === 'warning'
                            ? 'badge-warning'
                            : preflightContractDetails.mediaContractStatus === 'pass'
                              ? 'badge-success'
                              : 'badge-info'
                          }`}
                      >
                        {preflightContractDetails.mediaContractStatus === 'blocked'
                          ? 'BLOCKED'
                          : preflightContractDetails.mediaContractStatus === 'warning'
                            ? 'WARNING'
                            : preflightContractDetails.mediaContractStatus === 'pass'
                              ? 'PASS'
                              : 'UNKNOWN'}
                      </span>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Expected Modality</span>
                      <strong>{preflightContractDetails.mediaContractExpectedModality}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Strict Require Media</span>
                      <strong>
                        {preflightContractDetails.mediaContractRequireMedia === true
                          ? 'enabled'
                          : preflightContractDetails.mediaContractRequireMedia === false
                            ? 'disabled'
                            : 'unknown'}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Media Rows</span>
                      <strong>
                        {preflightContractDetails.mediaContractMediaRows}
                        {' / '}
                        {preflightContractDetails.mediaContractSampledRows}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Image / Audio Rows</span>
                      <strong>
                        {preflightContractDetails.mediaContractImageRows}
                        {' / '}
                        {preflightContractDetails.mediaContractAudioRows}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Mixed Rows</span>
                      <strong>{preflightContractDetails.mediaContractMixedRows}</strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Missing Local Refs</span>
                      <strong>
                        {preflightContractDetails.mediaContractMissingLocalImages}
                        {' image, '}
                        {preflightContractDetails.mediaContractMissingLocalAudios}
                        {' audio'}
                      </strong>
                    </div>
                    <div className="training-preflight-contract-card__row">
                      <span>Remote URL Refs</span>
                      <strong>
                        {preflightContractDetails.mediaContractRemoteImageRefs}
                        {' image, '}
                        {preflightContractDetails.mediaContractRemoteAudioRefs}
                        {' audio'}
                      </strong>
                    </div>
                    {preflightContractDetails.mediaContractErrors.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--error">
                        {preflightContractDetails.mediaContractErrors.join(' | ')}
                      </div>
                    )}
                    {preflightContractDetails.mediaContractWarnings.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--warning">
                        {preflightContractDetails.mediaContractWarnings.join(' | ')}
                      </div>
                    )}
                    {preflightContractDetails.mediaContractHints.length > 0 && (
                      <div className="training-preflight-contract-card__notice training-preflight-contract-card__notice--hint">
                        {preflightContractDetails.mediaContractHints.join(' | ')}
                      </div>
                    )}
                  </div>
                </div>
                <details className="training-preflight-panel__details">
                  <summary>Raw capability summary JSON</summary>
                  <pre className="resolved-defaults-panel__json">
                    {JSON.stringify(preflightContractDetails.rawCapabilitySummary || {}, null, 2)}
                  </pre>
                </details>
              </div>
            </div>
          )}
          {preflightPlan && Array.isArray(preflightPlan.suggestions) && preflightPlan.suggestions.length > 0 && (
            <div className="training-plan-panel">
              <div className="training-plan-panel__head">
                <div>
                  <strong>Preflight Plan Suggestions</strong>
                  <div className="training-plan-panel__hint">
                    Recommended: {preflightPlan.recommended_profile || 'balanced'}
                  </div>
                </div>
              </div>
              <div className="training-plan-panel__grid">
                {preflightPlan.suggestions.map((suggestion) => {
                  const isRecommended =
                    String(suggestion.profile || '').trim() === String(preflightPlan.recommended_profile || '').trim();
                  const isPreferred = String(suggestion.profile || '').trim() === preferredPlanProfile;
                  return (
                    <div
                      key={`training-plan-${suggestion.profile}`}
                      className={`training-plan-card ${suggestion.preflight?.ok ? 'training-plan-card--ok' : 'training-plan-card--error'
                        } ${isRecommended ? 'training-plan-card--recommended' : ''}`}
                    >
                      <div className="training-plan-card__head">
                        <strong>{suggestion.title || suggestion.profile}</strong>
                        <div className="training-plan-card__badges">
                          {isPreferred && <span className="badge badge-info">Preferred</span>}
                          {isRecommended && <span className="badge badge-success">Recommended</span>}
                          <span className={`badge ${suggestion.preflight?.ok ? 'badge-success' : 'badge-error'}`}>
                            {suggestion.preflight?.ok ? 'PASS' : 'BLOCKED'}
                          </span>
                        </div>
                      </div>
                      <div className="training-plan-card__meta">
                        VRAM risk: <strong>{String(suggestion.estimated_vram_risk || 'unknown')}</strong>
                        {Number.isFinite(Number(suggestion.estimated_vram_score))
                          ? ` (score ${Number(suggestion.estimated_vram_score)})`
                          : ''}
                      </div>
                      {suggestion.estimated_vram_note && (
                        <div className="training-plan-card__meta">{suggestion.estimated_vram_note}</div>
                      )}
                      <div className="training-plan-card__desc">
                        {suggestion.description}
                      </div>
                      {Array.isArray(suggestion.changes) && suggestion.changes.length > 0 && (
                        <div className="training-plan-card__changes">
                          {suggestion.changes.slice(0, 8).map((change, idx) => (
                            <div key={`plan-change-${suggestion.profile}-${idx}`}>
                              <code>{change.field}</code>: {String(change.from)} → {String(change.to)}
                              {change.reason ? ` (${change.reason})` : ''}
                            </div>
                          ))}
                          {suggestion.changes.length > 8 && (
                            <div>+{suggestion.changes.length - 8} more changes</div>
                          )}
                        </div>
                      )}
                      {Array.isArray(suggestion.preflight?.errors) && suggestion.preflight.errors.length > 0 && (
                        <div className="training-plan-card__errors">
                          {suggestion.preflight.errors.slice(0, 3).join(' | ')}
                        </div>
                      )}
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => onApplyPlanSuggestion(suggestion)}
                      >
                        Apply Suggested Config
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </details>
  );
}
