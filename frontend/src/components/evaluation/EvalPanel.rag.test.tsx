/**
 * Phase 5.3.5 — Sample Predictions card with RAG (grounded QA) enrichment.
 *
 * When the prediction row carries `rag_has_context: true`, the
 * Sample Predictions card grows a RAG-specific surface beneath each
 * prediction with:
 *   - "Faithful" / "Hallucinated" badge (green/red) keyed off
 *     `rag_is_faithful`.
 *   - "context covers gold: X%" inline diagnostic (retriever-side).
 *   - Red "unsupported tokens: X%" when > 0.
 *   - "Show retrieved context" disclosure with the context the
 *     model was given.
 *
 * Non-RAG runs are unaffected — the surface only renders when a row
 * carries the RAG enrichment fields. Regression-guarded below.
 */

import { render, screen } from '@testing-library/react';
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
                        pass_rate: 0.85,
                        metrics: {},
                        details: {
                            predictions_preview: predictionsPreview,
                            handler_id: 'rag_qa',
                            task_profile_resolved: 'rag_qa',
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
            return { data: { project_id: 4, count: 0, plans: [] } };
        }
        return { data: {} };
    });
}


describe('EvalPanel — Sample Predictions with RAGHandler enrichment', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders a green Faithful badge for grounded answers', async () => {
        mockApi([
            {
                prompt: 'What is the capital of France?',
                reference: 'Paris',
                prediction: 'Paris',
                rag_has_context: true,
                rag_context: 'Paris is the capital of France.',
                rag_faithfulness: 1.0,
                rag_context_recall: 1.0,
                rag_unsupported_rate: 0.0,
                rag_is_faithful: true,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // Faithful badge with the score.
        expect(await screen.findByText(/Faithful \(1\.00\)/i)).toBeInTheDocument();
        // Context recall surfaced inline.
        expect(screen.getByText(/context covers gold/i)).toBeInTheDocument();
    });

    it('renders a red Hallucinated badge for ungrounded answers', async () => {
        mockApi([
            {
                prompt: 'What is the capital of France?',
                reference: 'Paris',
                prediction: 'Tokyo and Madrid are great cities',
                rag_has_context: true,
                rag_context: 'Paris is the capital of France.',
                rag_faithfulness: 0.0,
                rag_context_recall: 1.0,
                rag_unsupported_rate: 1.0,
                rag_is_faithful: false,
                row_exact_match: 0.0,
                row_f1: 0.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(
            await screen.findByText(/Hallucinated \(0\.00\)/i),
        ).toBeInTheDocument();
        // Unsupported tokens flagged when > 0.
        expect(screen.getByText(/unsupported tokens: 100%/i)).toBeInTheDocument();
    });

    it('shows the retrieved context disclosure with the model input', async () => {
        mockApi([
            {
                prompt: 'Capital question',
                reference: 'Paris',
                prediction: 'Paris',
                rag_has_context: true,
                rag_context:
                    'Paris is the capital and most populous city of France.',
                rag_faithfulness: 1.0,
                rag_context_recall: 1.0,
                rag_unsupported_rate: 0.0,
                rag_is_faithful: true,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(
            await screen.findByText(/Show retrieved context/i),
        ).toBeInTheDocument();
        // The context body renders inside the details (kept in DOM
        // even when collapsed).
        expect(
            screen.getByText(/Paris is the capital and most populous city/i),
        ).toBeInTheDocument();
    });

    it('hides the unsupported-tokens note when fully grounded', async () => {
        mockApi([
            {
                prompt: 'Q',
                reference: 'A',
                prediction: 'A',
                rag_has_context: true,
                rag_context: 'A is the answer.',
                rag_faithfulness: 1.0,
                rag_context_recall: 1.0,
                rag_unsupported_rate: 0.0,  // zero — should not render the note
                rag_is_faithful: true,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/Faithful/i)).toBeInTheDocument();
        // No unsupported-tokens note when 0.
        expect(
            screen.queryByText(/unsupported tokens/i),
        ).not.toBeInTheDocument();
    });

    it('omits the RAG surface entirely when rag_has_context is false', async () => {
        // Plain QA row without context — RAG enrichment fields absent
        // → no RAG surface rendered (regression guard: don't accidentally
        // show a "Hallucinated" badge for non-RAG runs).
        mockApi([
            {
                prompt: 'What is 2+2?',
                reference: '4',
                prediction: '4',
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/What is 2\+2/i)).toBeInTheDocument();
        // No Faithful / Hallucinated badges for non-RAG rows.
        expect(screen.queryByText(/Faithful/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/Hallucinated/i)).not.toBeInTheDocument();
        expect(
            screen.queryByText(/Show retrieved context/i),
        ).not.toBeInTheDocument();
    });
});
