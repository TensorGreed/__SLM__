/**
 * Opt-in "null counts as absent" JSON scoring in the run form (closed More options; nothing changes
 * until opened): the payload carries ``null_as_absent`` only when this run overrides the project; the
 * project setting is saved from the same disclosure; results scored with it are labelled; a row's own
 * output schema verdict shows in the sample predictions.
 */

import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';

function mockApi(projectSetting = false) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/settings/null-as-absent')) return { data: { enabled: projectSetting } };
        if (url.includes('/eval-sets')) return { data: { eval_sets: [] } };
        if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [{
                    id: 1002, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 1,
                    metrics: { exact_match: 1, json_scoring: { null_as_absent: true, source: 'run' } }, details: {},
                }, {
                    id: 1001, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5,
                    metrics: { exact_match: 0.5 }, details: {
                        predictions_preview: [{
                            prompt: 'Ann is 30', reference: '{"name": "Ann"}', prediction: '{"name": 5}',
                            row_context: { output_schema: '{"properties": {"name": {"type": "string"}}}',
                                output_schema_source: 'row', output_schema_fields: 1, digest: 'd' },
                            row_output_schema_valid: false, row_output_schema_errors: ['/name: 5 is not of type string'],
                        }],
                    },
                }],
            };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/gates/21')) {
            return { data: { captured_at: '2026-03-26T00:00:00Z', passed: true, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return { data: { experiment_id: 21, is_ship: true, decision: 'SHIP', reasons: [], failed_gates: [], missing_metrics: [],
                gate_report: { passed: true, checks: [], missing_required_metrics: [], failed_gate_ids: [] } } };
        }
        if (url.includes('failure-clusters')) {
            return { data: { eval_result_id: 1002, experiment_id: 21, dataset_name: 'test', eval_type: 'exact_match',
                total_failures_analyzed: 0, reason_code_totals: {}, dominant_reason_code: null, clusters: [], remediation_plans: [] } };
        }
        if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
        if (url.includes('/gold/review')) return { data: { total: 0, rows: [] } };
        return { data: {} };
    });
}

async function openForm() {
    const user = userEvent.setup();
    render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
    await user.click(await screen.findByRole('button', { name: /exp-21/ }));
    await user.click(await screen.findByRole('button', { name: /Score this run/ }));
    return user;
}

describe('EvalPanel null-as-absent setting', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        useProjectStore.setState({ activeProject: null });
    });

    it('is closed by default and leaves the payload unchanged', async () => {
        mockApi();
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        const user = await openForm();
        expect(screen.getByTestId('null-as-absent-more-options')).not.toHaveAttribute('open');
        expect(apiMock.get.mock.calls.some(([url]) => String(url).includes('null-as-absent'))).toBe(false);
        await user.click(screen.getByTestId('eval-run-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1]).not.toHaveProperty('null_as_absent');
    });

    it('sends a per-run override and saves the project setting', async () => {
        mockApi(false);
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        apiMock.put.mockResolvedValue({ data: { enabled: true } });
        const user = await openForm();
        await user.click(screen.getByText(/More options: JSON scoring/));
        const toggle = await screen.findByTestId('null-as-absent-project-toggle');
        expect(screen.getByTestId('null-as-absent-run-select')).toHaveTextContent('Use the project setting (off)');
        await user.click(toggle);
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/4/settings/null-as-absent', { enabled: true }));
        await user.selectOptions(screen.getByTestId('null-as-absent-run-select'), 'off');
        await user.click(screen.getByTestId('eval-run-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1].null_as_absent).toBe(false);
    });

    it('labels results scored with it', async () => {
        mockApi();
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('eval-null-as-absent-1002')).toHaveTextContent('null counts as absent');
        expect(screen.queryByTestId('eval-null-as-absent-1001')).toBeNull();
    });
});
