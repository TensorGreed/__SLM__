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

// Quality-Lift phase 2 slice 3 — per-slice gate response fields.
interface PerSliceValue {
  slice_id: string;
  value: number;
  gate_value: number | null;
  support: number;
  passes: boolean;
  below_min_support: boolean;
  std?: number;
  n?: number;
}

// Quality-Lift phase 5 slice 3 — behavioral test failure shape that
// the backend's _attach_behavioral_details enricher merges into gate
// responses whose metric_id matches behavioral.*. Mirrors the
// runner's failed_examples shape — original_input vs perturbed_input
// for INV/DIR, input vs expected vs predicted for MFT.
interface BehavioralFailedExample {
  // INV / DIR fields
  original_input?: string;
  perturbed_input?: string;
  perturbation_name?: string;
  original_label?: string | null;
  perturbed_label?: string;
  expectation_kind?: string;
  target?: string | string[] | null;
  // MFT fields
  input?: string;
  expected_label?: string;
  predicted_label?: string;
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
  // Multi-seed variance (phase 1 slice 3) — only present on aggregate rows.
  actual_std?: number;
  actual_min?: number;
  actual_max?: number;
  actual_n?: number;
  gate_value?: number;
  variance_policy?: 'lower_bound' | 'mean' | 'scalar';
  per_seed?: PerSeedEntry[];
  seed_group_id?: string | null;
  // Phase 2 slice 3 — single-slice gate names a specific slice;
  // worst-slice gates carry the worst slice id + the full per-slice
  // breakdown so the drill-down can show every slice's verdict.
  slice_name?: string;
  worst_slice_id?: string | null;
  worst_slice_support?: number | null;
  per_slice_values?: PerSliceValue[];
  min_slice_support?: number;
  // Phase 5 slice 3 — behavioral test enrichment. Present only on
  // gates whose metric_id matches ``behavioral.<test_id>.<metric>``;
  // emitted by the backend's _attach_behavioral_details helper.
  behavioral_test_id?: string;
  behavioral_kind?: 'INV' | 'DIR' | 'MFT';
  behavioral_failed_examples?: BehavioralFailedExample[];
  behavioral_passed?: number;
  behavioral_total?: number;
  behavioral_capped_at_budget?: number;
  // Phase 6 slice 3 — Per-slice behavioral gate enrichment. Set when
  // the gate targets ``behavioral.<test_id>.per_slice.<slice_id>.<metric>``;
  // failed_examples already filtered to the slice bucket by the
  // backend (slice 2 work). The frontend just needs to render the
  // slice id alongside the test id and visually nest the row under
  // its parent test.
  behavioral_slice_id?: string;
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

const isWorstSliceGate = (gate: GateCheck): boolean =>
  gate.operator === 'worst_slice_gte' || gate.operator === 'worst_slice_lte';

const isSingleSliceGate = (gate: GateCheck): boolean =>
  typeof gate.slice_name === 'string' && gate.slice_name.length > 0;

const isBehavioralGate = (gate: GateCheck): boolean =>
  typeof gate.behavioral_test_id === 'string' && gate.behavioral_test_id.length > 0;

// Phase 14 — the optional probe gate (phase 13) grades against the
// platform-authored, held-out probe pack rather than the user's gold
// set. Style it distinctly so it reads as an *independent* ruler, not
// just another gold-set gate. Detected by its fixed metric/gate id.
const isProbeGate = (gate: GateCheck): boolean =>
  gate.metric_id === 'probe_pass_rate' || gate.gate_id === 'min_probe_pass_rate';

const BEHAVIORAL_KIND_COPY: Record<'INV' | 'DIR' | 'MFT', { label: string; explainer: string }> = {
  INV: {
    label: 'INV',
    explainer: 'Invariance — perturbed inputs must yield the same prediction.',
  },
  DIR: {
    label: 'DIR',
    explainer: 'Directional — perturbed inputs must change prediction in a stated way.',
  },
  MFT: {
    label: 'MFT',
    explainer: 'Minimum functionality — hand-authored canonicals that must always pass.',
  },
};

const formatStatusLabel = (gate: GateCheck, notMeasured: boolean): string => {
  if (notMeasured) {
    return gate.required ? '⚠️ Not measured' : '⏭️ Skipped';
  }
  if (gate.passed) {
    return '✅ Pass';
  }
  if (
    gate.reason === 'worst_slice_below_threshold' ||
    gate.reason === 'worst_slice_above_threshold'
  ) {
    // Phase 2 slice 3 — one or more slices dragged the worst below
    // the gate. Name it directly in the status so the user can act
    // without expanding the drill-down.
    const sliceTag = gate.worst_slice_id ? ` (${gate.worst_slice_id})` : '';
    return gate.required ? `❌ Worst slice fails${sliceTag}` : `⚠️ Worst slice${sliceTag}`;
  }
  if (
    gate.reason === 'variance_below_threshold' ||
    gate.reason === 'variance_above_threshold'
  ) {
    // Honest-metrics surfacing: mean cleared the gate but the spread
    // squeezed the lower / upper bound past the threshold.
    return gate.required ? '❌ Fail (variance)' : '⚠️ Variance';
  }
  if (gate.reason === 'no_eligible_slices_required' || gate.reason === 'no_eligible_slices_optional') {
    return gate.required ? '⚠️ No eligible slices' : '⏭️ No eligible slices';
  }
  return gate.required ? '❌ Fail' : '⚠️ Low';
};

const formatMetricLabel = (gate: GateCheck): string => {
  if (isSingleSliceGate(gate)) {
    return `${gate.metric_id} on ${gate.slice_name}`;
  }
  if (isWorstSliceGate(gate)) {
    return `worst-slice ${gate.metric_id}`;
  }
  return gate.metric_id;
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
              {gate_report.checks.filter((g) => !isBehavioralGate(g)).map((gate) => {
                const notMeasured = gate.actual === null;
                const aggregate = isAggregateCheck(gate);
                const worstSlice = isWorstSliceGate(gate);
                const sliceValues = gate.per_slice_values ?? [];
                const rowClass = notMeasured
                  ? (gate.required ? 'failed' : 'warn')
                  : (gate.passed ? 'passed' : gate.required ? 'failed' : 'warn');
                const probeGate = isProbeGate(gate);
                const statusLabel = formatStatusLabel(gate, notMeasured);
                const isExpanded = expandedGate === gate.gate_id;
                // Multi-seed (phase 1) drill-down OR worst-slice
                // (phase 2 slice 3) drill-down — either can expand.
                const canExpand =
                  (aggregate && (gate.per_seed?.length ?? 0) > 0) ||
                  (worstSlice && sliceValues.length > 0);
                const policyHint =
                  gate.variance_policy === 'lower_bound'
                    ? `gate value = mean ${gate.operator === 'gte' || gate.operator === 'worst_slice_gte' ? '−' : '+'} std`
                    : gate.variance_policy === 'mean'
                    ? 'gate value = mean (variance ignored)'
                    : undefined;
                return (
                  <React.Fragment key={gate.gate_id}>
                    <tr
                      className={`${rowClass}${canExpand ? ' scorecard-row--expandable' : ''}${probeGate ? ' scorecard-row--probe' : ''}`}
                      onClick={canExpand ? () => setExpandedGate(isExpanded ? null : gate.gate_id) : undefined}
                      data-testid={probeGate ? 'scorecard-probe-gate' : undefined}
                    >
                      <td>
                        {gate.gate_id}
                        {probeGate && (
                          <span
                            className="scorecard-probe-badge"
                            title="Graded against the platform-authored held-out probe pack — independent of your gold set."
                            data-testid="scorecard-probe-badge"
                          >
                            Independent ruler
                          </span>
                        )}
                      </td>
                      <td>{formatMetricLabel(gate)}</td>
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
                        {probeGate && !gate.passed && (
                          <a
                            href="#probe-pack-panel"
                            className="scorecard-probe-link"
                            data-testid="scorecard-probe-link"
                            onClick={(e) => e.stopPropagation()}
                          >
                            Inspect probes →
                          </a>
                        )}
                      </td>
                    </tr>
                    {canExpand && isExpanded && (
                      <tr className="scorecard-drilldown-row">
                        <td colSpan={5}>
                          <div className="scorecard-drilldown">
                            {aggregate && (gate.per_seed?.length ?? 0) > 0 && (
                              <>
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
                              </>
                            )}

                            {worstSlice && sliceValues.length > 0 && (
                              <>
                                <div className="scorecard-drilldown-header">
                                  Per-slice breakdown
                                  {typeof gate.min_slice_support === 'number' && (
                                    <>
                                      {' · '}
                                      <span className="scorecard-drilldown-stat">
                                        min support = {gate.min_slice_support}
                                      </span>
                                    </>
                                  )}
                                </div>
                                <table className="scorecard-drilldown-table">
                                  <thead>
                                    <tr>
                                      <th>Slice</th>
                                      <th>Value</th>
                                      <th>Gate value</th>
                                      <th>Support</th>
                                      <th>Verdict</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {sliceValues.map((sv) => {
                                      const isWorst = sv.slice_id === gate.worst_slice_id;
                                      const verdict = sv.below_min_support
                                        ? `↳ below min support (n=${sv.support})`
                                        : sv.passes
                                        ? '✅ passes'
                                        : '❌ fails';
                                      return (
                                        <tr
                                          key={`${gate.gate_id}-slice-${sv.slice_id}`}
                                          className={isWorst ? 'scorecard-drilldown-row--worst' : ''}
                                        >
                                          <td>
                                            {sv.slice_id}
                                            {isWorst && (
                                              <span className="scorecard-worst-tag" title="worst slice for this gate">
                                                {' '}(worst)
                                              </span>
                                            )}
                                          </td>
                                          <td>
                                            {sv.value.toFixed(3)}
                                            {typeof sv.std === 'number' && (
                                              <span className="scorecard-metric-std">
                                                {' '}± {sv.std.toFixed(3)}
                                              </span>
                                            )}
                                            {typeof sv.n === 'number' && sv.n > 1 && (
                                              <span className="scorecard-metric-n"> (n={sv.n})</span>
                                            )}
                                          </td>
                                          <td>{sv.gate_value != null ? sv.gate_value.toFixed(3) : '—'}</td>
                                          <td>{sv.support}</td>
                                          <td>{verdict}</td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                                {(gate.reason === 'worst_slice_below_threshold' ||
                                  gate.reason === 'worst_slice_above_threshold') && gate.worst_slice_id && (
                                  <div className="scorecard-drilldown-note">
                                    Slice <strong>{gate.worst_slice_id}</strong> dragged the worst-slice metric to
                                    {' '}{gate.actual?.toFixed(3)} {gate.operator === 'worst_slice_gte' ? '<' : '>'}{' '}threshold
                                    {' '}{gate.threshold}. Investigate that slice's data or relax the gate.
                                  </div>
                                )}
                                {policyHint && (
                                  <div className="scorecard-drilldown-policy-hint" title={policyHint}>
                                    Variance policy: <code>{gate.variance_policy}</code> ({policyHint})
                                  </div>
                                )}
                              </>
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

        {gate_report.checks.some(isBehavioralGate) && (
          <div className="scorecard-behavioral-section">
            <h3>Behavioral tests</h3>
            <table className="gates-table scorecard-behavioral-table">
              <thead>
                <tr>
                  <th>Kind</th>
                  <th><Term id="gate" /> ID</th>
                  <th>Test</th>
                  <th>Target</th>
                  <th>Pass rate</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {gate_report.checks
                  .filter(isBehavioralGate)
                  // Phase 6 slice 3 — Group by behavioral_test_id and put
                  // top-level rows (no behavioral_slice_id) above their
                  // per-slice descendants. Within a group the per-slice
                  // rows sort alphabetically by slice_id so the rendering
                  // is stable across renders + matches the order the
                  // user sees in the pack editor.
                  .sort((a, b) => {
                    const aTest = a.behavioral_test_id ?? '';
                    const bTest = b.behavioral_test_id ?? '';
                    if (aTest !== bTest) return aTest < bTest ? -1 : 1;
                    const aSlice = a.behavioral_slice_id ?? '';
                    const bSlice = b.behavioral_slice_id ?? '';
                    return aSlice < bSlice ? -1 : aSlice > bSlice ? 1 : 0;
                  })
                  .map((gate) => {
                  const notMeasured = gate.actual === null;
                  const isPerSlice = typeof gate.behavioral_slice_id === 'string'
                    && gate.behavioral_slice_id.length > 0;
                  const rowClass = notMeasured
                    ? (gate.required ? 'failed' : 'warn')
                    : (gate.passed ? 'passed' : gate.required ? 'failed' : 'warn');
                  const statusLabel = formatStatusLabel(gate, notMeasured);
                  const isExpanded = expandedGate === gate.gate_id;
                  const failedExamples = gate.behavioral_failed_examples ?? [];
                  const canExpand = failedExamples.length > 0;
                  const kindCopy = gate.behavioral_kind ? BEHAVIORAL_KIND_COPY[gate.behavioral_kind] : undefined;
                  return (
                    <React.Fragment key={gate.gate_id}>
                      <tr
                        className={`${rowClass}${canExpand ? ' scorecard-row--expandable' : ''}${isPerSlice ? ' scorecard-behavioral-row--per-slice' : ''}`}
                        onClick={canExpand ? () => setExpandedGate(isExpanded ? null : gate.gate_id) : undefined}
                      >
                        <td>
                          {kindCopy ? (
                            <span
                              className={`scorecard-behavioral-badge scorecard-behavioral-badge--${gate.behavioral_kind!.toLowerCase()}`}
                              title={kindCopy.explainer}
                            >
                              {kindCopy.label}
                            </span>
                          ) : '—'}
                        </td>
                        <td>{gate.gate_id}</td>
                        <td>
                          {/* Phase 6 slice 3 — per-slice gates render
                              with an indent + the slice id chip so
                              the visual nesting under the parent test
                              is unambiguous. */}
                          {isPerSlice && (
                            <span className="scorecard-behavioral-nesting-indent" aria-hidden="true">↳ </span>
                          )}
                          <code>{gate.behavioral_test_id}</code>
                          {isPerSlice && (
                            <>
                              {' '}
                              <span
                                className="scorecard-behavioral-slice-badge"
                                title={`Slice: ${gate.behavioral_slice_id}`}
                              >
                                slice: {gate.behavioral_slice_id}
                              </span>
                            </>
                          )}
                          {typeof gate.behavioral_passed === 'number' && typeof gate.behavioral_total === 'number' && (
                            <span className="scorecard-behavioral-counts">
                              {' '}({gate.behavioral_passed}/{gate.behavioral_total})
                            </span>
                          )}
                          {typeof gate.behavioral_capped_at_budget === 'number' && !isPerSlice && (
                            <span
                              className="scorecard-behavioral-capped"
                              title={`Capped at ${gate.behavioral_capped_at_budget} trials per budget`}
                            >
                              {' '}· capped
                            </span>
                          )}
                        </td>
                        <td>{gate.operator} {gate.threshold}</td>
                        <td>
                          {notMeasured ? 'N/A' : (gate.actual! * 100).toFixed(1) + '%'}
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
                          <td colSpan={6}>
                            <div className="scorecard-behavioral-drilldown">
                              <div className="scorecard-drilldown-header">
                                {/* Phase 6 slice 3 — Header surfaces the
                                    slice id so the user knows the
                                    failed_examples below are filtered
                                    to that slice's bucket (not the
                                    test-wide overall list). */}
                                {isPerSlice
                                  ? `Failed examples in slice "${gate.behavioral_slice_id}"`
                                  : 'Failed examples'}
                                {kindCopy && (
                                  <span className="scorecard-behavioral-explainer">
                                    {' · '}{kindCopy.explainer}
                                  </span>
                                )}
                              </div>
                              <table className="scorecard-drilldown-table">
                                <thead>
                                  {gate.behavioral_kind === 'MFT' ? (
                                    <tr>
                                      <th>Input</th>
                                      <th>Expected</th>
                                      <th>Predicted</th>
                                    </tr>
                                  ) : (
                                    <tr>
                                      <th>Original</th>
                                      <th>Perturbed</th>
                                      <th>Original label</th>
                                      <th>Perturbed label</th>
                                      {gate.behavioral_kind === 'DIR' && <th>Expected</th>}
                                    </tr>
                                  )}
                                </thead>
                                <tbody>
                                  {failedExamples.map((ex, idx) => (
                                    <tr key={`${gate.gate_id}-fail-${idx}`}>
                                      {gate.behavioral_kind === 'MFT' ? (
                                        <>
                                          <td className="scorecard-behavioral-cell">{ex.input}</td>
                                          <td><code>{ex.expected_label}</code></td>
                                          <td><code>{ex.predicted_label}</code></td>
                                        </>
                                      ) : (
                                        <>
                                          <td className="scorecard-behavioral-cell">{ex.original_input}</td>
                                          <td className="scorecard-behavioral-cell">{ex.perturbed_input}</td>
                                          <td><code>{ex.original_label ?? '—'}</code></td>
                                          <td><code>{ex.perturbed_label}</code></td>
                                          {gate.behavioral_kind === 'DIR' && (
                                            <td>
                                              <code>{
                                                Array.isArray(ex.target)
                                                  ? ex.target.join(' | ')
                                                  : (ex.target ?? 'any change')
                                              }</code>
                                            </td>
                                          )}
                                        </>
                                      )}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
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
        )}

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
