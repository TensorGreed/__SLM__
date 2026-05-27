/**
 * Phase 5.3.2 — Sample Predictions card with QA-style per-row enrichment.
 *
 * The QAHandler enriches each prediction with row_exact_match / row_f1
 * (per-row scores) and answer_span / span_marker (CoT extraction). The
 * UI surfaces those as a Status column with pass/partial/fail badges
 * and a "Show extracted answer span" disclosure when extraction
 * actually changed what got scored.
 *
 * These tests pin the contract — Status column appears only when row
 * scores are present, badges colorize by EM / F1, and the span
 * disclosure renders only when extraction changed the prediction.
 */

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


function mockApi(predictionsPreview: unknown[]) {
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
                        pass_rate: 0.75,
                        metrics: {},
                        details: {
                            predictions_preview: predictionsPreview,
                            handler_id: 'qa',
                            task_profile_resolved: 'instruction_sft',
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
                    is_ship: true,
                    decision: 'SHIP',
                    reasons: [],
                    failed_gates: [],
                    missing_metrics: [],
                    gate_report: {
                        passed: true,
                        checks: [],
                        missing_required_metrics: [],
                        failed_gate_ids: [],
                    },
                },
            };
        }
        if (url.includes('/evaluation/packs')) {
            return { data: { packs: [] } };
        }
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
                    passed: true,
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
            return {
                data: { project_id: 4, count: 0, plans: [] },
            };
        }
        return { data: {} };
    });
}


describe('EvalPanel — Sample Predictions card with QAHandler enrichment', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders Status column + per-row badges when row scores are present', async () => {
        mockApi([
            {
                prompt: 'What is the capital of France?',
                reference: 'Paris',
                prediction: 'Paris',
                answer_span: 'Paris',
                span_marker: null,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
            {
                prompt: 'What is 2+2?',
                reference: '4',
                prediction: 'I think the answer is five',
                answer_span: 'five',
                span_marker: 'the\\s+answer\\s+is',
                row_exact_match: 0.0,
                row_f1: 0.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // Status column header appears alongside the standard columns.
        expect(await screen.findByRole('columnheader', { name: 'Score' })).toBeInTheDocument();
        // First row passed → pass badge.
        expect(screen.getByText('pass')).toBeInTheDocument();
        // Second row failed → fail badge.
        expect(screen.getByText('fail')).toBeInTheDocument();
    });

    it('renders "Show extracted answer span" disclosure for CoT rows', async () => {
        mockApi([
            {
                prompt: 'What is 2+2?',
                reference: '4',
                // Long CoT-style prediction — the QAHandler extracted "4" from it.
                prediction: 'Two plus two requires basic arithmetic. Final answer: 4.',
                answer_span: '4',
                span_marker: 'final\\s+answer\\s*[:\\-]\\s*',
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // The disclosure for extracted spans is present.
        expect(
            await screen.findByText(/Show extracted answer span/i),
        ).toBeInTheDocument();
    });

    it('omits the Status column entirely when no row scores are present', async () => {
        // GenericHandler / non-QA paths don't write row_exact_match, so the
        // Status column shouldn't render — preserves today's layout for
        // pre-Phase-5.3.2 results and for handlers that don't enrich.
        mockApi([
            {
                prompt: 'Best headphones I have used.',
                reference: 'positive',
                prediction: 'The battery life is amazing.',
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // Wait for the predictions card to appear with a known prompt.
        expect(
            await screen.findByText(/Best headphones I have used/i),
        ).toBeInTheDocument();
        // Status column header is absent for non-enriched runs.
        expect(
            screen.queryByRole('columnheader', { name: 'Score' }),
        ).not.toBeInTheDocument();
    });

    it('skips the span disclosure when answer_span equals the full prediction', async () => {
        // No CoT marker matched (span_marker is null) — the disclosure must
        // not render even though answer_span exists.
        mockApi([
            {
                prompt: 'Capital of France?',
                reference: 'Paris',
                prediction: 'Paris',
                answer_span: 'Paris',
                span_marker: null,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText('pass')).toBeInTheDocument();
        expect(
            screen.queryByText(/Show extracted answer span/i),
        ).not.toBeInTheDocument();
    });
});
