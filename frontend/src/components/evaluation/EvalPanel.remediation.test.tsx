/** EvalPanel in remediation scoring mode: metrics block + per-row patch. */

import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
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

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';


function mockApi(predictionsPreview: unknown[], metrics: unknown = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/experiments')) {
            return { data: [{ id: 21, name: 'exp-21' }] };
        }
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [
                    {
                        id: 1001,
                        dataset_name: 'gold_dev',
                        eval_type: 'f1',
                        pass_rate: 0.5,
                        metrics,
                        details: {
                            predictions_preview: predictionsPreview,
                            handler_id: 'structured_extraction',
                            task_profile_resolved: 'structured_extraction',
                        },
                    },
                ],
            };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) {
            return { data: { overall_risk: 'low', red_flags: [] } };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return {
                data: {
                    experiment_id: 21,
                    is_ship: false,
                    decision: 'NO-SHIP',
                    reasons: [],
                    failed_gates: [],
                    missing_metrics: [],
                    gate_report: {
                        passed: false,
                        checks: [],
                        missing_required_metrics: [],
                        failed_gate_ids: [],
                    },
                },
            };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return {
                data: {
                    preferred_pack_id: null,
                    active_pack_id: null,
                    active_pack_source: 'default',
                    active_pack: { display_name: 'Auto' },
                },
            };
        }
        if (url.includes('/evaluation/gates/21')) {
            return {
                data: {
                    captured_at: '2026-03-26T00:00:00Z',
                    passed: false,
                    failed_gate_ids: [],
                    missing_required_metrics: [],
                    checks: [],
                },
            };
        }
        if (url.includes('failure-clusters')) {
            return {
                data: {
                    eval_result_id: 1001,
                    experiment_id: 21,
                    dataset_name: 'gold_dev',
                    eval_type: 'f1',
                    total_failures_analyzed: 0,
                    reason_code_totals: {},
                    dominant_reason_code: null,
                    clusters: [],
                    remediation_plans: [],
                },
            };
        }
        if (url.includes('remediation-plans')) {
            return { data: { project_id: 4, count: 0, plans: [] } };
        }
        if (url.includes('/gold/review')) {
            return { data: { total: 0, rows: [] } };
        }
        return { data: {} };
    });
}



const METRICS = {
    scoring_mode: 'remediation',
    remediation_pass_rate: 0.5, patch_applies_rate: 0.75, total: 4, correct: 2,
    code_checkable_count: 3, code_parses_rate: 1,
    json_validity_rate: 1, malformed_rate: 0,
    patch_reasons: { context_mismatch: 1 },
    judge_correct: 0.5,
};

const ROW = {
    prompt: 'code', reference: '{}', prediction: '{}',
    is_valid_json: true, scoring_mode: 'remediation',
    path: 'src/db.py', language: 'python', start_line: 40,
    row_finding: { label: 'injection.query', start_line: 45, end_line: 45 },
    parsed_prediction: { patch: '--- a/src/db.py\n+++ b/src/db.py\n@@ -45,1 +45,1 @@\n-a\n+b', rationale: 'bind it', citations: ['P1'] },
    row_passage_headers: [{ id: 'P1', where: 'docs/db.md:1-4' }],
    row_patch_applies: true, row_code_checkable: true, row_code_parses: true,
    row_issue_resolved: null, row_citations_valid: true,
};

describe('EvalPanel - remediation', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
    });

    it('shows the remediation metrics block and no confusion matrix', async () => {
        mockApi([ROW], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('remediation-metrics')).toBeInTheDocument();
        expect(screen.getByTestId('rem-pass')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('rem-resolved')).toHaveTextContent('not evaluated');
        expect(screen.getByTestId('rem-judge')).toHaveTextContent('0.50');
        expect(screen.queryByText(/confusion/i)).not.toBeInTheDocument();
    });

    it('renders the row view with the patch and cited passage', async () => {
        mockApi([ROW], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('remediation-row')).toBeInTheDocument();
        expect(screen.getByTestId('rem-chip-applies')).toHaveTextContent('patch applies');
        expect(screen.getByTestId('rem-passage-P1')).toHaveTextContent('docs/db.md:1-4');
    });
});
