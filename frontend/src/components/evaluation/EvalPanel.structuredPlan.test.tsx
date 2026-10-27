/** EvalPanel in structured_plan scoring mode: metrics block + per-row view, no confusion matrix. */

import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';

function mockApi(predictionsPreview: unknown[], metrics: unknown) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [{
                    id: 1001, dataset_name: 'gold_dev', eval_type: 'f1', pass_rate: 0.5, metrics,
                    details: { predictions_preview: predictionsPreview, handler_id: 'structured_extraction', task_profile_resolved: 'structured_extraction' },
                }],
            };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/scorecard/21')) {
            return { data: { experiment_id: 21, is_ship: false, decision: 'NO-SHIP', reasons: [], failed_gates: [], missing_metrics: [], gate_report: { passed: false, checks: [], missing_required_metrics: [], failed_gate_ids: [] } } };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
        }
        if (url.includes('/evaluation/gates/21')) {
            return { data: { captured_at: '2026-03-26T00:00:00Z', passed: false, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
        }
        if (url.includes('failure-clusters')) {
            return { data: { eval_result_id: 1001, experiment_id: 21, dataset_name: 'gold_dev', eval_type: 'f1', total_failures_analyzed: 0, reason_code_totals: {}, dominant_reason_code: null, clusters: [], remediation_plans: [] } };
        }
        if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
        if (url.includes('/gold/review')) return { data: { total: 0, rows: [] } };
        return { data: {} };
    });
}

const METRICS = {
    scoring_mode: 'structured_plan',
    structured_plan_pass_rate: 0.5, schema_valid_rate: 0.75, json_validity_rate: 1, total: 4, correct: 2,
    field_match_rate: 0.75,
    per_pointer: { '/environment': { match_rate: 0.75, support: 4, missing_rate: 0 } },
    pointers: ['/environment'],
    schema: { root: 'plan.schema.json', sha256: 'abcdef0123456789' },
    per_class: { a: { precision: 1, recall: 1, f1: 1, support: 1 } },
};

const ROW = {
    prompt: 'scenario', reference: '{}', prediction: '{}', is_valid_json: true, scoring_mode: 'structured_plan',
    row_schema_valid: true, schema_errors: [],
    row_pointer_results: { '/environment': { match: true, pred: ['prod'], gold: ['prod'], source: 'reference' } },
    row_structured_plan_pass: true,
};

describe('EvalPanel - structured plan', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
    });

    it('shows the structured-plan metrics block and no classification chart', async () => {
        mockApi([ROW], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('structured-plan-metrics')).toBeInTheDocument();
        expect(screen.getByTestId('sp-pass')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('sp-provenance')).toHaveTextContent('plan.schema.json (sha abcdef01)');
        expect(screen.queryByText(/confusion/i)).not.toBeInTheDocument();
    });

    it('renders the per-row view', async () => {
        mockApi([ROW], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('structured-plan-row')).toBeInTheDocument();
        expect(screen.getByTestId('sp-row-pointer-/environment')).toHaveTextContent('prod');
    });
});
