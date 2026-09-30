/**
 * Experiment list: per-run status, live loss sparkline + kill switch for
 * in-flight runs, open / start / reset / delete, compare selection and
 * bulk-archive of failed runs.
 */

import type { Job as JobShape } from '../../api/jobs';
import EmptyState from '../shared/EmptyState';
import ExperimentClassifierHeadBadge from './ExperimentClassifierHeadBadge';
import ExperimentLiveSignals from './ExperimentLiveSignals';
import type {
  Experiment,
} from './trainingPanelTypes';
import { statusColor } from './trainingPanelUtils';

interface TrainingRunsListProps {
  /** Copy tweak: runs are created from Training Config, not this list. */
  hideCreateControls: boolean;
  experiments: Experiment[];
  jobByExperimentId: Map<number, JobShape>;
  selectedForCompare: number[];
  handleBulkArchiveFailed: () => Promise<void>;
  handleDeleteExperiment: (exp: Experiment) => Promise<void>;
  handleResetExperiment: (exp: Experiment) => Promise<void>;
  setPendingStartExperiment: (exp: Experiment) => void;
  setShowCompare: (show: boolean) => void;
  toggleCompareSelection: (expId: number) => void;
  viewDashboard: (exp: Experiment) => void;
}

export default function TrainingRunsList({
  hideCreateControls,
  experiments,
  jobByExperimentId,
  selectedForCompare,
  handleBulkArchiveFailed,
  handleDeleteExperiment,
  handleResetExperiment,
  setPendingStartExperiment,
  setShowCompare,
  toggleCompareSelection,
  viewDashboard,
}: TrainingRunsListProps) {
  return (
      experiments.length === 0 ? (
        <EmptyState
          title="No experiments yet"
          description={
            hideCreateControls
              ? 'No runs yet. Open Training Config to pick a recipe + base model, then launch your first experiment.'
              : 'Create a training experiment to fine-tune your first model. The Autopilot Planner is the fastest path — type a plain-English brief and it picks the recipe.'
          }
          docsHref="http://localhost:3001/docs/workflows/training"
        />
      ) : (
        <div className="training-experiment-list">
          {experiments.filter((e) => e.status === 'failed').length >= 2 && (
            <div
              data-testid="bulk-archive-failed-banner"
              style={{
                padding: 'var(--space-md)',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 'var(--space-md)',
                marginBottom: 'var(--space-md)',
              }}
            >
              <span>
                You have <strong>
                  {experiments.filter((e) => e.status === 'failed').length}
                </strong> failed experiment(s). Archiving resets each one
                back to PENDING and renames its output directory to
                <code> .bak.&lt;timestamp&gt;</code> so a fresh start won't
                inherit stale checkpoints.
              </span>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleBulkArchiveFailed}
                data-testid="bulk-archive-failed-button"
              >
                📦 Archive all failed
              </button>
            </div>
          )}
          {selectedForCompare.length > 1 && (
            <div className="training-compare-bar">
              <button className="btn btn-primary" onClick={() => setShowCompare(true)}>
                Compare Selected ({selectedForCompare.length})
              </button>
            </div>
          )}
          {experiments.map((exp) => (
            <div
              key={exp.id}
              className="training-experiment-item"
            >
              <div className="training-experiment-main">
                <input
                  type="checkbox"
                  checked={selectedForCompare.includes(exp.id)}
                  onChange={() => toggleCompareSelection(exp.id)}
                  className="training-checkbox"
                />
                <div>
                  <div className="training-experiment-name">
                    {exp.name}
                    <span
                      className="training-experiment-id"
                      title="Experiment ID — use this for SQL queries / log greps / artifact paths"
                    >
                      #{exp.id}
                    </span>
                  </div>
                  <div className="training-experiment-meta">
                    {exp.base_model} • {exp.training_mode}
                  </div>
                  {exp.domain_pack_applied && (
                    <div className="training-experiment-submeta">
                      Pack: {exp.domain_pack_applied}
                      {exp.domain_pack_source ? ` (${exp.domain_pack_source})` : ''}
                    </div>
                  )}
                  {exp.domain_profile_applied && (
                    <div className="training-experiment-submeta">
                      Profile: {exp.domain_profile_applied}
                      {exp.domain_profile_source ? ` (${exp.domain_profile_source})` : ''}
                    </div>
                  )}
                </div>
              </div>
              <div className="training-experiment-actions">
                <span className={`badge ${statusColor(exp.status)}`}>{exp.status}</span>
                <ExperimentClassifierHeadBadge
                  taskType={exp.config?.task_type}
                  status={exp.status}
                />
                <ExperimentLiveSignals
                  job={jobByExperimentId.get(exp.id)}
                />
                {exp.status === 'pending' && (
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => setPendingStartExperiment(exp)}
                  >
                    Start
                  </button>
                )}
                {exp.status === 'failed' && (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleResetExperiment(exp)}
                    data-testid={`experiment-reset-${exp.id}`}
                    title="Reset to PENDING + archive stale output dir + clear stale checkpoints"
                  >
                    🔄 Reset
                  </button>
                )}
                <button className="btn btn-secondary btn-sm" onClick={() => viewDashboard(exp)}>
                  Dashboard
                </button>
                {exp.status !== 'running' && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => handleDeleteExperiment(exp)}
                    data-testid={`experiment-delete-${exp.id}`}
                    title="Permanently delete this experiment + its output directory"
                  >
                    🗑
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )
  );
}
