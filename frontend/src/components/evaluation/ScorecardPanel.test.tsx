/**
 * Quality-Lift phase 1, slice 3 — ScorecardPanel variance + drill-down.
 *
 * Pins:
 *  * Single-seed scorecard renders the bare ``actual`` (legacy path,
 *    no behavior change).
 *  * Aggregate gate row renders ``mean ± std (n=N)`` and the lower-bound
 *    gate_value when policy=lower_bound.
 *  * variance_below_threshold reason surfaces a "Fail (variance)" label
 *    and the click-to-drill caret.
 *  * Clicking the row reveals per-seed rows with seed/experiment/eval_result
 *    columns AND the honest-metrics explainer note.
 *  * Clicking again collapses.
 *  * Single-seed rows have NO expand caret.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

vi.mock('../../api/client', () => ({
  default: apiMock,
}));

import ScorecardPanel from './ScorecardPanel';

const SCORECARD_SINGLE_SEED = {
  experiment_id: 7,
  is_ship: true,
  decision: 'SHIP',
  reasons: [],
  failed_gates: [],
  missing_metrics: [],
  gate_report: {
    passed: true,
    failed_gate_ids: [],
    missing_required_metrics: [],
    checks: [
      {
        gate_id: 'min_f1',
        metric_id: 'macro_f1',
        operator: 'gte',
        threshold: 0.8,
        required: true,
        actual: 0.85,
        passed: true,
        reason: 'ok',
      },
    ],
  },
};

const SCORECARD_AGGREGATE_FAILING_VARIANCE = {
  experiment_id: 9,
  is_ship: false,
  decision: 'NO-SHIP',
  reasons: ['Failed 1 mandatory gates.'],
  failed_gates: ['min_f1'],
  missing_metrics: [],
  gate_report: {
    passed: false,
    failed_gate_ids: ['min_f1'],
    missing_required_metrics: [],
    checks: [
      {
        gate_id: 'min_f1',
        metric_id: 'macro_f1',
        operator: 'gte',
        threshold: 0.8,
        required: true,
        actual: 0.83,
        passed: false,
        reason: 'variance_below_threshold',
        actual_std: 0.04,
        actual_min: 0.79,
        actual_max: 0.87,
        actual_n: 3,
        gate_value: 0.79,
        variance_policy: 'lower_bound',
        per_seed: [
          { experiment_id: 11, seed_value: 42, eval_result_id: 21, pass_rate: 0.94 },
          { experiment_id: 12, seed_value: 43, eval_result_id: 22, pass_rate: 0.96 },
          { experiment_id: 13, seed_value: 44, eval_result_id: 23, pass_rate: 0.95 },
        ],
        seed_group_id: 'abc12345-deadbeef',
      },
    ],
  },
};

describe('ScorecardPanel', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
  });

  it('renders scalar actual for single-seed runs (no variance affordances)', async () => {
    apiMock.get.mockResolvedValueOnce({ data: SCORECARD_SINGLE_SEED });
    render(<ScorecardPanel projectId={1} experimentId={7} />);

    await waitFor(() => expect(screen.getByText('SHIP')).toBeInTheDocument());

    // 0.85 rendered as 0.8500 (existing toFixed(4) path).
    expect(screen.getByText('0.8500')).toBeInTheDocument();
    // No ± character anywhere — single-seed row stayed on the scalar path.
    expect(screen.queryByText(/±/)).toBeNull();
    // No expand caret on a non-aggregate row.
    expect(screen.queryByText('▸')).toBeNull();
  });

  it('renders mean ± std (n=N) and gate_value when policy=lower_bound', async () => {
    apiMock.get.mockResolvedValueOnce({ data: SCORECARD_AGGREGATE_FAILING_VARIANCE });
    render(<ScorecardPanel projectId={1} experimentId={9} />);

    await waitFor(() => expect(screen.getByText('NO-SHIP')).toBeInTheDocument());

    // Variance is rendered with toFixed(3) — different precision than
    // the scalar path so the spread reads clearly.
    expect(screen.getByText('0.830')).toBeInTheDocument();
    expect(screen.getByText(/± 0\.040/)).toBeInTheDocument();
    expect(screen.getByText(/\(n=3\)/)).toBeInTheDocument();
    // Lower-bound gate_value rendered with the arrow.
    expect(screen.getByText(/→ 0\.790/)).toBeInTheDocument();
    // Variance-specific failure label, NOT the generic "Fail".
    expect(screen.getByText(/Fail \(variance\)/)).toBeInTheDocument();
    // Expand caret present, collapsed by default.
    expect(screen.getByText('▸')).toBeInTheDocument();
    // Drill-down content NOT rendered until click.
    expect(screen.queryByText(/Per-seed runs/)).toBeNull();
  });

  it('expands the row to show per-seed drill-down on click', async () => {
    apiMock.get.mockResolvedValueOnce({ data: SCORECARD_AGGREGATE_FAILING_VARIANCE });
    const user = userEvent.setup();
    render(<ScorecardPanel projectId={1} experimentId={9} />);

    await waitFor(() => expect(screen.getByText(/Fail \(variance\)/)).toBeInTheDocument());

    // The clickable row is the gate row itself.
    const gateIdCell = screen.getByText('min_f1');
    const row = gateIdCell.closest('tr')!;
    await user.click(row);

    // Drill-down header surfaces the seed_group_id prefix.
    expect(screen.getByText(/Per-seed runs/)).toBeInTheDocument();
    expect(screen.getByText('abc12345')).toBeInTheDocument();
    // min / max stats rendered (max=0.870 from the fixture).
    expect(screen.getByText(/min 0\.790/)).toBeInTheDocument();
    expect(screen.getByText(/max 0\.870/)).toBeInTheDocument();
    // Each seed row rendered.
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('43')).toBeInTheDocument();
    expect(screen.getByText('44')).toBeInTheDocument();
    // Honest-metrics explainer note rendered for variance failure.
    expect(
      screen.getByText(/lower-bound policy applies the std/i)
    ).toBeInTheDocument();
    // Caret flipped to expanded state.
    expect(screen.getByText('▾')).toBeInTheDocument();

    // Click again collapses.
    await user.click(row);
    expect(screen.queryByText(/Per-seed runs/)).toBeNull();
    expect(screen.getByText('▸')).toBeInTheDocument();
  });

  it('skips the explainer note when the mean itself failed (not variance)', async () => {
    const scorecard = {
      ...SCORECARD_AGGREGATE_FAILING_VARIANCE,
      gate_report: {
        ...SCORECARD_AGGREGATE_FAILING_VARIANCE.gate_report,
        checks: [
          {
            ...SCORECARD_AGGREGATE_FAILING_VARIANCE.gate_report.checks[0],
            actual: 0.70,
            gate_value: 0.66,
            reason: 'below_threshold',
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    const user = userEvent.setup();
    render(<ScorecardPanel projectId={1} experimentId={9} />);

    await waitFor(() => expect(screen.getByText('NO-SHIP')).toBeInTheDocument());
    // Generic ❌ Fail label, not the variance variant.
    expect(screen.getByText(/❌ Fail/)).toBeInTheDocument();
    expect(screen.queryByText(/Fail \(variance\)/)).toBeNull();

    const row = screen.getByText('min_f1').closest('tr')!;
    await user.click(row);
    // The drill-down still renders (it's gated on per_seed presence,
    // not on the failure reason), but the explainer note doesn't.
    expect(screen.getByText(/Per-seed runs/)).toBeInTheDocument();
    expect(
      screen.queryByText(/lower-bound policy applies the std/i)
    ).toBeNull();
  });
});
