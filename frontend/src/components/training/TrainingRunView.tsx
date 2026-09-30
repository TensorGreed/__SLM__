/**
 * One training run in focus: status + cancel, warnings, live signals, loss
 * curve, why-this-plan + checkpoints, observability + vibe-check telemetry
 * and the worker log. Presentational — TrainingPanel owns the run stream.
 */

import type { ErrorEnvelope } from '../../api/errors';
import CheckpointsPanel from './CheckpointsPanel';
import ErrorPanel from '../shared/ErrorPanel';
import LossCurvePanel from './LossCurvePanel';
import WarmStartDeltaChart from './WarmStartDeltaChart';
import WhyThisPlanPanel from './WhyThisPlanPanel';
import { TerminalConsole } from '../shared/TerminalConsole';
import './LossCurvePanel.css';
import './TrainingPanel.css';
import type {
  Experiment,
  TrainingMetric,
  TrainingObservabilitySummary,
  VibeCheckSnapshot,
  VibeCheckTimelineResponse,
} from './trainingPanelTypes';
import { asRecord, statusColor } from './trainingPanelUtils';

interface TrainingRunViewProps {
  projectId: number;
  activeExperiment: Experiment;
  metrics: TrainingMetric[];
  taskState: string;
  trainingError: ErrorEnvelope | null;
  trainingWarnings: string[];
  trainingLogs: string[];
  observabilitySummary: TrainingObservabilitySummary | null;
  observabilityRecentCount: number;
  observabilityError: string;
  observabilityLoading: boolean;
  vibeTimeline: VibeCheckSnapshot[];
  vibeSelectedIndex: number;
  vibeConfig: VibeCheckTimelineResponse['config'] | null;
  vibeError: string;
  vibeLoading: boolean;
  onBack: () => void;
  onCancel: (experimentId: number) => void;
  onDismissError: () => void;
  onRefreshExperiments: () => void;
  onRefreshObservability: (experimentId: number) => void;
  onRefreshVibe: (experimentId: number) => void;
  onSelectVibe: (index: number) => void;
}

export default function TrainingRunView({
  projectId,
  activeExperiment,
  metrics,
  taskState,
  trainingError,
  trainingWarnings,
  trainingLogs,
  observabilitySummary,
  observabilityRecentCount,
  observabilityError,
  observabilityLoading,
  vibeTimeline,
  vibeSelectedIndex,
  vibeConfig,
  vibeError,
  vibeLoading,
  onBack,
  onCancel,
  onDismissError,
  onRefreshExperiments,
  onRefreshObservability,
  onRefreshVibe,
  onSelectVibe,
}: TrainingRunViewProps) {
  const latestMetric = metrics[metrics.length - 1] || {};
  const totalEpochs = Number(activeExperiment.config?.num_epochs || 3);
  const epochValue = typeof latestMetric.epoch === 'number' ? Number(latestMetric.epoch) : null;
  const currentEpoch = epochValue !== null ? epochValue.toFixed(2) : '—';
  const currentStep = typeof latestMetric.step === 'number' ? Number(latestMetric.step) : null;
  const completedEpochs = epochValue !== null ? Math.floor(epochValue) : 0;
  let epochState = 'Waiting for first metric...';
  if (epochValue !== null) {
    if (completedEpochs >= totalEpochs) {
      epochState = `All ${totalEpochs} epochs completed`;
    } else {
      epochState = `Epoch ${Math.min(totalEpochs, completedEpochs + 1)} running`;
    }
  }
  const currentTrainLoss = latestMetric.train_loss !== undefined ? latestMetric.train_loss : null;
  const currentEvalLoss = latestMetric.eval_loss !== undefined ? latestMetric.eval_loss : null;
  const anomalyRate = Number(observabilitySummary?.gradient_anomaly_rate || 0);
  const hallucinationRate = Number(observabilitySummary?.hallucination_signal_rate || 0);
  const topLayers = Array.isArray(observabilitySummary?.top_layers)
    ? observabilitySummary?.top_layers || []
    : [];
  const topTokens = Array.isArray(observabilitySummary?.top_attention_tokens)
    ? observabilitySummary?.top_attention_tokens || []
    : [];
  const safeVibeIndex = Math.min(
    Math.max(vibeSelectedIndex, 0),
    Math.max(0, vibeTimeline.length - 1),
  );
  const selectedVibeSnapshot = vibeTimeline.length > 0 ? vibeTimeline[safeVibeIndex] : null;

  return (
    <div className="animate-fade-in training-panel-stack">
      <div className="card">
        <button
          className="btn btn-secondary btn-sm training-back-btn"
          onClick={() => {
            onBack();
          }}
        >
          ← Back to Experiments
        </button>

        <div className="training-active-head">
          <div>
            <h3 className="training-active-title">
              {activeExperiment.name}
              <span
                className="training-experiment-id"
                title="Experiment ID — use this for SQL queries / log greps / artifact paths"
              >
                #{activeExperiment.id}
              </span>
            </h3>
            <div className="training-active-meta">
              {activeExperiment.base_model} • {activeExperiment.training_mode}
            </div>
            {activeExperiment.domain_pack_applied && (
              <div className="training-active-submeta">
                Pack: {activeExperiment.domain_pack_applied}
                {activeExperiment.domain_pack_source ? ` (${activeExperiment.domain_pack_source})` : ''}
              </div>
            )}
            {activeExperiment.domain_profile_applied && (
              <div className="training-active-submeta">
                Profile: {activeExperiment.domain_profile_applied}
                {activeExperiment.domain_profile_source ? ` (${activeExperiment.domain_profile_source})` : ''}
              </div>
            )}
            {taskState && (
              <div className="training-active-submeta">
                Worker task state: {taskState}
              </div>
            )}
          </div>
          <div className="training-inline-actions">
            {activeExperiment.status === 'running' && (
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => onCancel(activeExperiment.id)}
              >
                Cancel
              </button>
            )}
            <span className={`badge ${statusColor(activeExperiment.status)} training-status-badge`}>
              {activeExperiment.status.toUpperCase()}
            </span>
          </div>
        </div>

        {trainingError && (
          <ErrorPanel
            envelope={trainingError}
            onDismiss={onDismissError}
            testIdPrefix="training-error"
          />
        )}
        {trainingWarnings.length > 0 && (
          <div className="training-alert training-alert--warning">
            Preflight warnings: {trainingWarnings.join(' | ')}
          </div>
        )}

        <div className="metrics-grid">
          <div className="metric-box box-blue">
            <span className="mb-label">Current Epoch</span>
            <span className="mb-value">{currentEpoch} / {totalEpochs}</span>
            <div className="metric-subtext">
              {epochState}
              {currentStep !== null ? ` • step ${currentStep}` : ''}
            </div>
          </div>
          <div className="metric-box box-green">
            <span className="mb-label">Training Loss</span>
            <span className="mb-value">{currentTrainLoss !== null ? Number(currentTrainLoss).toFixed(4) : '--'}</span>
          </div>
          <div className="metric-box box-purple">
            <span className="mb-label">Eval Loss</span>
            <span className="mb-value">{currentEvalLoss !== null ? Number(currentEvalLoss).toFixed(4) : '--'}</span>
          </div>
        </div>

        {/* V2 ML-native viz — train vs eval loss overlay with the
            best-eval marker + the overfitting region shaded when eval
            starts climbing back from its minimum. Mounts for every
            run that has any loss metrics streamed in. */}
        <LossCurvePanel metrics={metrics} />

        {/* Track 1, Epic B/C — warm-started runs: what your rows added on top
            of the pre-tuned base (delta vs the warm-start's starting loss). */}
        <WarmStartDeltaChart
          metrics={metrics}
          warmStart={asRecord(activeExperiment.config?.['_warm_start']) as {
            source?: string;
            checkpoint_name?: string | null;
            reason?: string;
          }}
        />

        {/* P20 — Why-this-plan + checkpoints (Wave D backend exposure). */}
        <WhyThisPlanPanel projectId={projectId} experiment={activeExperiment} />
        <CheckpointsPanel
          projectId={projectId}
          experiment={activeExperiment}
          onLifecycleChange={() => {
            onRefreshExperiments();
          }}
        />

        <div className="training-observability-panel">
          <div className="training-observability-panel__head">
            <h4>Observability Telemetry</h4>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => onRefreshObservability(activeExperiment.id)}
              disabled={observabilityLoading}
            >
              {observabilityLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          {observabilityError && (
            <div className="training-alert training-alert--warning training-alert--tight">{observabilityError}</div>
          )}
          <div className="training-observability-panel__stats">
            <span className="badge badge-info">Events {observabilitySummary?.event_count ?? 0}</span>
            <span>Recent payloads: {observabilityRecentCount}</span>
            <span>Gradient anomaly rate: {(anomalyRate * 100).toFixed(1)}%</span>
            <span>Hallucination signal rate: {(hallucinationRate * 100).toFixed(1)}%</span>
            <span>Step range: {observabilitySummary?.step_min ?? '—'} - {observabilitySummary?.step_max ?? '—'}</span>
          </div>
          <div className="training-observability-panel__grid">
            <div>
              <div className="training-observability-panel__subtitle">Top Layers</div>
              <div className="training-observability-panel__list">
                {topLayers.length === 0 ? (
                  <div className="training-observability-panel__muted">No gradient snapshots yet.</div>
                ) : (
                  topLayers.slice(0, 6).map((layer, idx) => (
                    <div key={`obs-layer-${idx}`} className="training-observability-panel__row">
                      <span>{layer.layer || 'layer'}</span>
                      <strong>avg {Number(layer.avg_grad_norm || 0).toFixed(4)}</strong>
                      <span>max {Number(layer.max_grad_norm || 0).toFixed(4)}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div>
              <div className="training-observability-panel__subtitle">Top Attention Tokens</div>
              <div className="training-observability-panel__list">
                {topTokens.length === 0 ? (
                  <div className="training-observability-panel__muted">No attention probe samples yet.</div>
                ) : (
                  topTokens.slice(0, 8).map((token, idx) => (
                    <div key={`obs-token-${idx}`} className="training-observability-panel__row">
                      <span>{token.token || 'token'}</span>
                      <strong>{token.count ?? 0}</strong>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="training-vibe-panel">
          <div className="training-vibe-panel__head">
            <h4>Vibe Check Timeline</h4>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => onRefreshVibe(activeExperiment.id)}
              disabled={vibeLoading}
            >
              {vibeLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          {vibeError && (
            <div className="training-alert training-alert--warning training-alert--tight">{vibeError}</div>
          )}
          <div className="training-vibe-panel__meta">
            <span>Snapshots: {vibeTimeline.length}</span>
            <span>Interval: {vibeConfig?.interval_steps ?? 50} steps</span>
            <span>Provider: {vibeConfig?.provider || 'mock'}</span>
            <span>Prompts: {Array.isArray(vibeConfig?.prompts) ? vibeConfig?.prompts.length : 0}</span>
          </div>
          {selectedVibeSnapshot ? (
            <div className="training-vibe-panel__body">
              <div className="training-vibe-panel__slider">
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, vibeTimeline.length - 1)}
                  value={safeVibeIndex}
                  onChange={(e) => onSelectVibe(Number(e.target.value || 0))}
                />
                <div className="training-vibe-panel__snapshot-meta">
                  <span>Step {selectedVibeSnapshot.step ?? '—'}</span>
                  <span>Epoch {selectedVibeSnapshot.epoch ?? '—'}</span>
                  <span>
                    Progress{' '}
                    {typeof selectedVibeSnapshot.progress === 'number'
                      ? `${(selectedVibeSnapshot.progress * 100).toFixed(1)}%`
                      : '—'}
                  </span>
                </div>
              </div>
              <div className="training-vibe-grid">
                {(Array.isArray(selectedVibeSnapshot.outputs) ? selectedVibeSnapshot.outputs : []).map((item, idx) => (
                  <article className="training-vibe-card" key={`vibe-${selectedVibeSnapshot.step || 0}-${idx}`}>
                    <div className="training-vibe-card__prompt">{item.prompt || 'Prompt'}</div>
                    <div className="training-vibe-card__reply">{item.reply || 'No reply generated.'}</div>
                  </article>
                ))}
              </div>
            </div>
          ) : (
            <div className="training-vibe-panel__empty">
              No vibe snapshots yet. Snapshots appear every configured interval during training.
            </div>
          )}
        </div>

        <TerminalConsole logs={trainingLogs} height="320px" />
      </div>
    </div>
  );
}
