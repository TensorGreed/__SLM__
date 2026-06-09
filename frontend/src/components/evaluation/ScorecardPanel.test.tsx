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

  // ────────────────────────────────────────────────────────────────────
  // Quality-Lift phase 2 slice 3 — Per-slice gates
  // ────────────────────────────────────────────────────────────────────

  it('labels a single-slice gate with "metric on slice"', async () => {
    const scorecard = {
      experiment_id: 11,
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
            gate_id: 'min_f1_long_input',
            metric_id: 'f1',
            slice_name: 'long_input',
            operator: 'gte',
            threshold: 0.6,
            required: true,
            actual: 0.71,
            passed: true,
            reason: 'ok',
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    render(<ScorecardPanel projectId={1} experimentId={11} />);
    await waitFor(() => expect(screen.getByText('SHIP')).toBeInTheDocument());
    // The metric column reads "f1 on long_input" instead of bare "f1"
    // so the user can see which slice the gate targets.
    expect(screen.getByText('f1 on long_input')).toBeInTheDocument();
  });

  it('renders worst-slice gate with worst slice id and drill-down', async () => {
    const scorecard = {
      experiment_id: 13,
      is_ship: false,
      decision: 'NO-SHIP',
      reasons: ['Failed 1 mandatory gates.'],
      failed_gates: ['no_slice_below_60'],
      missing_metrics: [],
      gate_report: {
        passed: false,
        failed_gate_ids: ['no_slice_below_60'],
        missing_required_metrics: [],
        checks: [
          {
            gate_id: 'no_slice_below_60',
            metric_id: 'f1',
            operator: 'worst_slice_gte',
            threshold: 0.6,
            required: true,
            actual: 0.52,
            passed: false,
            reason: 'worst_slice_below_threshold',
            worst_slice_id: 'long_input',
            worst_slice_support: 40,
            min_slice_support: 5,
            per_slice_values: [
              { slice_id: 'long_input', value: 0.52, gate_value: 0.52, support: 40, passes: false, below_min_support: false },
              { slice_id: 'short_input', value: 0.85, gate_value: 0.85, support: 200, passes: true, below_min_support: false },
              { slice_id: 'tiny_slice', value: 0.30, gate_value: 0.30, support: 2, passes: false, below_min_support: true },
            ],
            variance_policy: 'scalar',
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    const user = userEvent.setup();
    render(<ScorecardPanel projectId={1} experimentId={13} />);

    await waitFor(() => expect(screen.getByText('NO-SHIP')).toBeInTheDocument());

    // The metric column reads "worst-slice f1" for the aggregate gate
    // and the status label names the worst slice directly.
    expect(screen.getByText('worst-slice f1')).toBeInTheDocument();
    expect(screen.getByText(/Worst slice fails \(long_input\)/)).toBeInTheDocument();

    // Drill-down not visible until click.
    expect(screen.queryByText(/Per-slice breakdown/)).toBeNull();

    const row = screen.getByText('no_slice_below_60').closest('tr')!;
    await user.click(row);

    // Drill-down lists every slice (including the support-floor-filtered one),
    // marks the worst slice, and renders the directional explainer note.
    // ``long_input`` appears in the status label too, so we assert at least
    // one occurrence rather than uniqueness.
    expect(screen.getByText(/Per-slice breakdown/)).toBeInTheDocument();
    expect(screen.getAllByText(/long_input/).length).toBeGreaterThan(0);
    expect(screen.getByText('short_input')).toBeInTheDocument();
    expect(screen.getByText('tiny_slice')).toBeInTheDocument();
    // Worst-slice tag on the long_input row.
    expect(screen.getByText(/\(worst\)/)).toBeInTheDocument();
    // Below-min-support flag on tiny_slice.
    expect(screen.getByText(/below min support \(n=2\)/)).toBeInTheDocument();
    // Honest-metrics explainer.
    expect(screen.getByText(/dragged the worst-slice metric/i)).toBeInTheDocument();
  });

  // ────────────────────────────────────────────────────────────────────
  // Quality-Lift phase 5 slice 3 — Behavioral tests section
  // ────────────────────────────────────────────────────────────────────

  it('renders the dedicated Behavioral tests section with INV badge + drill-down', async () => {
    const scorecard = {
      experiment_id: 21,
      is_ship: false,
      decision: 'NO-SHIP',
      reasons: ['Failed 1 mandatory gates.'],
      failed_gates: ['typo_invariance_gate'],
      missing_metrics: [],
      gate_report: {
        passed: false,
        failed_gate_ids: ['typo_invariance_gate'],
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
          {
            gate_id: 'typo_invariance_gate',
            metric_id: 'behavioral.typo_invariance.pass_rate',
            operator: 'gte',
            threshold: 0.85,
            required: true,
            actual: 0.72,
            passed: false,
            reason: 'below_threshold',
            behavioral_test_id: 'typo_invariance',
            behavioral_kind: 'INV',
            behavioral_passed: 36,
            behavioral_total: 50,
            behavioral_failed_examples: [
              {
                original_input: 'great product',
                perturbed_input: 'great pdroduct',
                perturbation_name: 'typo',
                original_label: 'positive',
                perturbed_label: 'negative',
              },
              {
                original_input: 'works as advertised',
                perturbed_input: 'wokrs as advertised',
                perturbation_name: 'typo',
                original_label: 'positive',
                perturbed_label: 'neutral',
              },
            ],
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    const user = userEvent.setup();
    render(<ScorecardPanel projectId={1} experimentId={21} />);

    await waitFor(() => {
      expect(screen.getByText(/Behavioral tests/)).toBeInTheDocument();
    });

    // Quality gates section (top) renders min_f1 but NOT the
    // behavioral gate (it lives in its own section to avoid mixing
    // metric-id and behavioral-id rendering).
    expect(screen.getByText('min_f1')).toBeInTheDocument();
    // Quality gates section uses the metric_id, not the behavioral
    // test_id, so 'macro_f1' surfaces there.
    expect(screen.getByText('macro_f1')).toBeInTheDocument();

    // INV badge with explainer tooltip.
    const invBadge = screen.getByText('INV');
    expect(invBadge).toBeInTheDocument();
    expect(invBadge).toHaveAttribute('title', expect.stringContaining('Invariance'));

    // Test id + counts (36/50).
    expect(screen.getByText('typo_invariance')).toBeInTheDocument();
    expect(screen.getByText(/\(36\/50\)/)).toBeInTheDocument();

    // Pass rate rendered as a percentage in the behavioral row.
    expect(screen.getByText('72.0%')).toBeInTheDocument();

    // Drill-down not visible until click.
    expect(screen.queryByText(/Failed examples/)).toBeNull();

    // The behavioral row is clickable (failed_examples > 0).
    const testIdCell = screen.getByText('typo_invariance');
    const row = testIdCell.closest('tr')!;
    await user.click(row);

    expect(screen.getByText(/Failed examples/)).toBeInTheDocument();
    // Both failed examples render.
    expect(screen.getByText('great product')).toBeInTheDocument();
    expect(screen.getByText('great pdroduct')).toBeInTheDocument();
    expect(screen.getByText('wokrs as advertised')).toBeInTheDocument();
    // Labels rendered for at least the original positive.
    expect(screen.getAllByText('positive').length).toBeGreaterThan(0);
  });

  it('renders MFT drill-down with input / expected / predicted columns', async () => {
    const scorecard = {
      experiment_id: 22,
      is_ship: false,
      decision: 'NO-SHIP',
      reasons: [],
      failed_gates: [],
      missing_metrics: [],
      gate_report: {
        passed: false,
        failed_gate_ids: [],
        missing_required_metrics: [],
        checks: [
          {
            gate_id: 'mft_gate',
            metric_id: 'behavioral.canonicals.pass_rate',
            operator: 'gte',
            threshold: 1.0,
            required: true,
            actual: 0.66,
            passed: false,
            reason: 'below_threshold',
            behavioral_test_id: 'canonicals',
            behavioral_kind: 'MFT',
            behavioral_passed: 2,
            behavioral_total: 3,
            behavioral_failed_examples: [
              {
                input: 'pretty good',
                expected_label: 'positive',
                predicted_label: 'neutral',
              },
            ],
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    const user = userEvent.setup();
    render(<ScorecardPanel projectId={1} experimentId={22} />);

    await waitFor(() => {
      expect(screen.getByText(/Behavioral tests/)).toBeInTheDocument();
    });

    // MFT-specific badge.
    expect(screen.getByText('MFT')).toBeInTheDocument();
    const row = screen.getByText('canonicals').closest('tr')!;
    await user.click(row);

    // MFT drill-down has Input / Expected / Predicted columns —
    // distinct from INV/DIR's Original / Perturbed / labels.
    expect(screen.getByText('pretty good')).toBeInTheDocument();
    // Expected + predicted labels rendered in <code> tags.
    expect(screen.getByText('positive')).toBeInTheDocument();
    expect(screen.getByText('neutral')).toBeInTheDocument();
  });

  it('shows the capped flag when the runner hit the prediction budget', async () => {
    const scorecard = {
      experiment_id: 23,
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
            gate_id: 'huge_inv',
            metric_id: 'behavioral.huge_test.pass_rate',
            operator: 'gte',
            threshold: 0.85,
            required: true,
            actual: 0.91,
            passed: true,
            reason: 'ok',
            behavioral_test_id: 'huge_test',
            behavioral_kind: 'INV',
            behavioral_passed: 1820,
            behavioral_total: 2000,
            behavioral_failed_examples: [],
            behavioral_capped_at_budget: 2000,
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    render(<ScorecardPanel projectId={1} experimentId={23} />);

    await waitFor(() => {
      expect(screen.getByText(/Behavioral tests/)).toBeInTheDocument();
    });
    // Capped indicator surfaces so the user knows trials were sampled.
    expect(screen.getByText(/capped/)).toBeInTheDocument();
  });

  // ────────────────────────────────────────────────────────────────────
  // Quality-Lift phase 6 slice 3 — Per-slice behavioral gates
  // ────────────────────────────────────────────────────────────────────

  it('renders per-slice behavioral gates nested under their parent test with slice id surfaced', async () => {
    const scorecard = {
      experiment_id: 31,
      is_ship: false,
      decision: 'NO-SHIP',
      reasons: ['Failed 1 mandatory gates.'],
      failed_gates: ['typo_long_input'],
      missing_metrics: [],
      gate_report: {
        passed: false,
        failed_gate_ids: ['typo_long_input'],
        missing_required_metrics: [],
        checks: [
          // Two behavioral gates on the same test: top-level + per-slice.
          // The phase 6 slice 3 grouping should put the top-level row
          // first and the per-slice row underneath.
          {
            gate_id: 'typo_overall',
            metric_id: 'behavioral.typo_invariance.pass_rate',
            operator: 'gte',
            threshold: 0.85,
            required: true,
            actual: 0.91,
            passed: true,
            reason: 'ok',
            behavioral_test_id: 'typo_invariance',
            behavioral_kind: 'INV',
            behavioral_passed: 50,
            behavioral_total: 55,
            behavioral_failed_examples: [
              {
                original_input: 'overall_failure_row',
                perturbed_input: 'overall_failure_row_p',
                perturbation_name: 'typo',
                original_label: 'positive',
                perturbed_label: 'negative',
              },
            ],
          },
          {
            gate_id: 'typo_long_input',
            metric_id: 'behavioral.typo_invariance.per_slice.long_input.pass_rate',
            operator: 'gte',
            threshold: 0.85,
            required: true,
            actual: 0.55,
            passed: false,
            reason: 'below_threshold',
            behavioral_test_id: 'typo_invariance',
            behavioral_slice_id: 'long_input',
            behavioral_kind: 'INV',
            behavioral_passed: 11,
            behavioral_total: 20,
            behavioral_failed_examples: [
              {
                original_input: 'long_specific_failure',
                perturbed_input: 'lnog_specific_failure',
                perturbation_name: 'typo',
                original_label: 'positive',
                perturbed_label: 'negative',
              },
            ],
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    const user = userEvent.setup();
    render(<ScorecardPanel projectId={1} experimentId={31} />);

    await waitFor(() => {
      expect(screen.getByText(/Behavioral tests/)).toBeInTheDocument();
    });
    // The per-slice gate surfaces the slice id explicitly in the Test column.
    expect(screen.getByText(/slice: long_input/)).toBeInTheDocument();
    // Both gates render their counts; the per-slice row uses the
    // slice-specific 11/20, NOT the test-wide 50/55.
    expect(screen.getByText(/\(50\/55\)/)).toBeInTheDocument();
    expect(screen.getByText(/\(11\/20\)/)).toBeInTheDocument();

    // Group ordering: the top-level row's gate_id appears before the
    // per-slice gate_id in the DOM.
    const topLevelRow = screen.getByText('typo_overall').closest('tr')!;
    const perSliceRow = screen.getByText('typo_long_input').closest('tr')!;
    expect(topLevelRow.compareDocumentPosition(perSliceRow)
      & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    // Click the per-slice gate → drill-down shows ONLY the slice's
    // failed_examples, NOT the test-wide overall failures.
    await user.click(perSliceRow);
    expect(screen.getByText(/Failed examples in slice "long_input"/)).toBeInTheDocument();
    expect(screen.getByText('long_specific_failure')).toBeInTheDocument();
    // The top-level test's failure row is NOT in the per-slice drill-down.
    expect(screen.queryByText('overall_failure_row')).toBeNull();
  });

  it('per-slice gate without a top-level sibling still renders with slice nesting', async () => {
    // Pin the case where a user gates only the per-slice metric (e.g.
    // they care about robustness on a specific slice but not overall).
    const scorecard = {
      experiment_id: 32,
      is_ship: false,
      decision: 'NO-SHIP',
      reasons: [],
      failed_gates: [],
      missing_metrics: [],
      gate_report: {
        passed: false,
        failed_gate_ids: [],
        missing_required_metrics: [],
        checks: [
          {
            gate_id: 'typo_hindi_only',
            metric_id: 'behavioral.typo_invariance.per_slice.hindi.pass_rate',
            operator: 'gte',
            threshold: 0.85,
            required: true,
            actual: 0.62,
            passed: false,
            reason: 'below_threshold',
            behavioral_test_id: 'typo_invariance',
            behavioral_slice_id: 'hindi',
            behavioral_kind: 'INV',
            behavioral_passed: 13,
            behavioral_total: 21,
            behavioral_failed_examples: [
              {
                original_input: 'hindi_specific_failure',
                perturbed_input: 'hindi_specific_failure_p',
                perturbation_name: 'typo',
                original_label: 'positive',
                perturbed_label: 'negative',
              },
            ],
          },
        ],
      },
    };
    apiMock.get.mockResolvedValueOnce({ data: scorecard });
    render(<ScorecardPanel projectId={1} experimentId={32} />);
    await waitFor(() => {
      expect(screen.getByText(/Behavioral tests/)).toBeInTheDocument();
    });
    // Slice badge surfaces even without a parent top-level row.
    expect(screen.getByText(/slice: hindi/)).toBeInTheDocument();
    expect(screen.getByText('typo_invariance')).toBeInTheDocument();
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
