/**
 * Quality gate reporter showing pass/fail decisions with missing-metric detection.
 *
 * Multi-seed variance (Quality-Lift phase 1, slice 3): when a gate's
 * actual value came from an aggregate EvalResult (seed-group rollup),
 * the row renders ``mean ± std (n=N)`` instead of the bare scalar and
 * gains an expand toggle that lists each contributing seed run. The
 * lower-bound variance policy on the backend (gate passes only if
 * ``mean − std >= threshold`` for gte, ``mean + std <= threshold`` for
 * lte) surfaces a dedicated ``variance_below_threshold`` /
 * ``variance_above_threshold`` reason when the mean would've cleared
 * the bar but the variance squeezed it under — the row's status badge
 * calls that out explicitly so the user isn't surprised.
 */

import React, { useEffect, useState } from 'react';
import api from '../../api/client';
import { Term } from '../shared/Term';
import './ScorecardPanel.css';

interface PerSeedEntry {
  experiment_id: number | null;
  seed_value: number | null;
  eval_result_id: number | null;
  pass_rate: number | null;
}

interface GateCheck {
  gate_id: string;
  metric_id: string;
  operator: string;
  threshold: number;
  required: boolean;
  actual: number | null;
  passed: boolean;
  reason?: string;
  // Multi-seed variance (slice 3) — only present on aggregate rows.
  actual_std?: number;
  actual_min?: number;
  actual_max?: number;
  actual_n?: number;
  gate_value?: number;
  variance_policy?: 'lower_bound' | 'mean' | 'scalar';
  per_seed?: PerSeedEntry[];
  seed_group_id?: string | null;
}

interface GateReport {
  passed: boolean;
  checks: GateCheck[];
  missing_required_metrics: string[];
  failed_gate_ids: string[];
}

interface Scorecard {
  experiment_id: number;
  is_ship: boolean;
  decision: 'SHIP' | 'NO-SHIP';
  reasons: string[];
  failed_gates: string[];
  missing_metrics: string[];
  gate_report: GateReport;
}

interface ScorecardPanelProps {
  projectId: number;
  experimentId: number;
}

const isAggregateCheck = (gate: GateCheck): boolean =>
  typeof gate.actual_n === 'number' && gate.actual_n > 1 && typeof gate.actual_std === 'number';

const formatStatusLabel = (gate: GateCheck, notMeasured: boolean): string => {
  if (notMeasured) {
    return gate.required ? '⚠️ Not measured' : '⏭️ Skipped';
  }
  if (gate.passed) {
    return '✅ Pass';
  }
  if (
    gate.reason === 'variance_below_threshold' ||
    gate.reason === 'variance_above_threshold'
  ) {
    // Honest-metrics surfacing: mean cleared the gate but the spread
    // squeezed the lower / upper bound past the threshold.
    return gate.required ? '❌ Fail (variance)' : '⚠️ Variance';
  }
  return gate.required ? '❌ Fail' : '⚠️ Low';
};

const ScorecardPanel: React.FC<ScorecardPanelProps> = ({ projectId, experimentId }) => {
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedGate, setExpandedGate] = useState<string | null>(null);

  useEffect(() => {
    const fetchScorecard = async () => {
      try {
        setLoading(true);
        const response = await api.get<Scorecard>(`/projects/${projectId}/evaluation/scorecard/${experimentId}`);
        setScorecard(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to load scorecard');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchScorecard();
  }, [projectId, experimentId]);

  if (loading) return <div className="scorecard-loading">Loading scorecard...</div>;
  if (error) return <div className="scorecard-error">{error}</div>;
  if (!scorecard) return null;

  const { is_ship, decision, reasons, gate_report } = scorecard;

  return (
    <div className={`scorecard-container ${is_ship ? 'ship' : 'no-ship'}`}>
      <div className="scorecard-header">
        <div className="decision-badge">{decision}</div>
        <h2>Experiment Scorecard</h2>
      </div>

      {reasons.length > 0 && (
        <div className="scorecard-reasons">
          <h3>Blockers:</h3>
          <ul>
            {reasons.map((reason, idx) => (
              <li key={idx} className="blocker-item">{reason}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="gates-grid">
        <div className="gates-section">
          <h3>Quality <Term id="gate" plural /></h3>
          <table className="gates-table">
            <thead>
              <tr>
                <th><Term id="gate" /> ID</th>
                <th>Metric</th>
                <th>Target</th>
                <th>Actual</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {gate_report.checks.map((gate) => {
                const notMeasured = gate.actual === null;
                const aggregate = isAggregateCheck(gate);
                const rowClass = notMeasured
                  ? (gate.required ? 'failed' : 'warn')
                  : (gate.passed ? 'passed' : gate.required ? 'failed' : 'warn');
                const statusLabel = formatStatusLabel(gate, notMeasured);
                const isExpanded = expandedGate === gate.gate_id;
                const canExpand = aggregate && (gate.per_seed?.length ?? 0) > 0;
                const policyHint =
                  gate.variance_policy === 'lower_bound'
                    ? `gate value = mean ${gate.operator === 'gte' ? '−' : '+'} std`
                    : gate.variance_policy === 'mean'
                    ? 'gate value = mean (variance ignored)'
                    : undefined;
                return (
                  <React.Fragment key={gate.gate_id}>
                    <tr
                      className={`${rowClass}${canExpand ? ' scorecard-row--expandable' : ''}`}
                      onClick={canExpand ? () => setExpandedGate(isExpanded ? null : gate.gate_id) : undefined}
                    >
                      <td>{gate.gate_id}</td>
                      <td>{gate.metric_id}</td>
                      <td>{gate.operator} {gate.threshold}</td>
                      <td>
                        {notMeasured ? (
                          'N/A'
                        ) : aggregate ? (
                          <span
                            className="scorecard-metric-block"
                            title={policyHint}
                          >
                            <span className="scorecard-metric-mean">{gate.actual!.toFixed(3)}</span>
                            <span className="scorecard-metric-std"> ± {gate.actual_std!.toFixed(3)}</span>
                            <span className="scorecard-metric-n"> (n={gate.actual_n})</span>
                            {gate.variance_policy === 'lower_bound' && typeof gate.gate_value === 'number' && (
                              <span className="scorecard-metric-gate-value" title="value compared against threshold">
                                {' → '}{gate.gate_value.toFixed(3)}
                              </span>
                            )}
                          </span>
                        ) : (
                          gate.actual!.toFixed(4)
                        )}
                      </td>
                      <td>
                        {statusLabel}
                        {canExpand && (
                          <span className="scorecard-expand-caret" aria-label={isExpanded ? 'collapse' : 'expand'}>
                            {' '}{isExpanded ? '▾' : '▸'}
                          </span>
                        )}
                      </td>
                    </tr>
                    {canExpand && isExpanded && (
                      <tr className="scorecard-drilldown-row">
                        <td colSpan={5}>
                          <div className="scorecard-drilldown">
                            <div className="scorecard-drilldown-header">
                              Per-seed runs (group <code>{gate.seed_group_id?.slice(0, 8) ?? '—'}</code>)
                              {' · '}
                              <span className="scorecard-drilldown-stat">
                                min {gate.actual_min?.toFixed(3)} · max {gate.actual_max?.toFixed(3)}
                              </span>
                            </div>
                            <table className="scorecard-drilldown-table">
                              <thead>
                                <tr>
                                  <th>Seed</th>
                                  <th>Experiment</th>
                                  <th>EvalResult</th>
                                  <th>Pass rate</th>
                                </tr>
                              </thead>
                              <tbody>
                                {gate.per_seed!.map((entry, idx) => (
                                  <tr key={`${gate.gate_id}-seed-${entry.seed_value ?? idx}`}>
                                    <td>{entry.seed_value ?? '—'}</td>
                                    <td>{entry.experiment_id ?? '—'}</td>
                                    <td>{entry.eval_result_id ?? '—'}</td>
                                    <td>
                                      {entry.pass_rate != null
                                        ? entry.pass_rate.toFixed(3)
                                        : '—'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {(gate.reason === 'variance_below_threshold' ||
                              gate.reason === 'variance_above_threshold') && (
                              <div className="scorecard-drilldown-note">
                                Mean {gate.actual!.toFixed(3)} {gate.operator === 'gte' ? '≥' : '≤'} threshold
                                {' '}{gate.threshold}, but lower-bound policy applies the std and the gate value
                                {' '}{gate.gate_value?.toFixed(3)} fails. Drop std (more seeds) or relax the gate.
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        {gate_report.missing_required_metrics.length > 0 && (
          <div className="missing-metrics-section">
            <h3>Missing Metrics</h3>
            <div className="missing-metrics-list">
              {gate_report.missing_required_metrics.map(m => (
                <div key={m} className="missing-metric-tag">{m}</div>
              ))}
            </div>
            <p className="missing-diagnostics">
              Some required metrics are missing from evaluation results. Run full evaluation suite to clear.
            </p>
          </div>
        )}
      </div>

      <div className="scorecard-footer">
        <p>Decisions are deterministic and reproducible based on project gate policy.</p>
      </div>
    </div>
  );
};

export default ScorecardPanel;
