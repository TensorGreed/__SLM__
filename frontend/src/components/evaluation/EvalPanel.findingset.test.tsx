/** EvalPanel in finding_set scoring mode: metrics block + per-row findings. */

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
    scoring_mode: 'finding_set',
    precision: 0.5, recall: 0.25, f1: 0.3333, f1_macro: 0.2,
    tp_total: 1, fp_total: 1, fn_total: 3, total: 4,
    clean_chunk_count: 4, clean_chunk_fp_count: 1, clean_chunk_fp_rate: 0.25,
    malformed_count: 2, malformed_rate: 0.1, invalid_finding_count: 3,
    attr_accuracy: { severity: { correct: 1, total: 2, accuracy: 0.5 } },
    attr_accuracy_mean: 0.5, iou_threshold: 0.5,
    hierarchical: { precision: 0.75, recall: 0.5, f1: 0.6, partial_matches: 2, parent_credit: 0.5 },
    per_class: {
        'injection.query': { precision: 1, recall: 0.5, f1: 0.6667, support: 2, tp: 1, fp: 0, fn: 1 },
        'secret.hardcoded': { precision: 0, recall: 0, f1: 0, support: 2, tp: 0, fp: 1, fn: 2 },
    },
};

const ROW = {
    prompt: 'code', reference: '{}', prediction: '{}',
    is_valid_json: true, scoring_mode: 'finding_set',
    path: 'src/db.py', language: 'python', start_line: 40,
    row_matched_findings: [{ label: 'injection.query', start_line: 45, end_line: 47, pred_start_line: 45, pred_end_line: 48, iou: 0.75 }],
    row_missed_findings: [{ label: 'secret.hardcoded', start_line: 50, end_line: 50 }],
    row_spurious_findings: [{ label: 'logging.sensitive', start_line: 52, end_line: 53 }],
    row_malformed: false, row_is_clean_reference: false,
    parsed_prediction: {}, parsed_reference: {},
};

describe('EvalPanel - finding_set', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
    });

    it('shows the metrics block with clean-chunk, malformed and per-label detail', async () => {
        mockApi([ROW], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));

        expect(await screen.findByTestId('finding-set-metrics')).toBeInTheDocument();
        expect(screen.getByTestId('fs-precision')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('fs-f1-macro')).toHaveTextContent('20.0%');
        expect(screen.getByTestId('fs-clean-fp')).toHaveTextContent('25.0%');
        expect(screen.getByTestId('fs-clean-fp')).toHaveTextContent('1 of 4 clean chunks');
        expect(screen.getByTestId('fs-malformed')).toHaveTextContent('2');
        expect(screen.getByTestId('fs-invalid')).toHaveTextContent('3');
        expect(screen.getByTestId('fs-attrs')).toHaveTextContent('severity');
        expect(screen.getByTestId('fs-hierarchical')).toHaveTextContent('2 partial matches');
        expect(screen.getByTestId('fs-label-secret.hardcoded')).toBeInTheDocument();
        // No confusion-matrix chart for findings.
        expect(screen.queryByText(/confusion/i)).not.toBeInTheDocument();
    });

    it('lists matched / missed / spurious findings with row provenance', async () => {
        mockApi([ROW], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));

        expect(await screen.findByTestId('finding-matched')).toHaveTextContent('injection.query · lines 45–47');
        expect(screen.getByTestId('finding-matched')).toHaveTextContent('predicted lines 45–48 · IoU 0.75');
        expect(screen.getByTestId('finding-missed')).toHaveTextContent('secret.hardcoded · line 50');
        expect(screen.getByTestId('finding-spurious')).toHaveTextContent('logging.sensitive · lines 52–53');
        expect(screen.getByTestId('finding-set-row-where')).toHaveTextContent('src/db.py · python · starts at line 40');
    });

    it('labels a malformed row apart from misses', async () => {
        mockApi([{ ...ROW, is_valid_json: false, row_malformed: true, row_matched_findings: [], row_spurious_findings: [] }], METRICS);
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('finding-set-row-malformed')).toBeInTheDocument();
    });
});
