/**
 * Phase 5.3.6 — Sample Predictions card with AlignmentHandler enrichment.
 *
 * For DPO/ORPO runs, each row carries similarity-to-chosen,
 * similarity-to-rejected, margin, and a preference_correct flag.
 * The card renders a "Preferred chosen" / "Preferred rejected" badge
 * plus a "Show chosen vs rejected completions" disclosure so the
 * researcher can eyeball where the preference broke.
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
                        pass_rate: 0.5,
                        metrics: {},
                        details: {
                            predictions_preview: predictionsPreview,
                            handler_id: 'alignment',
                            task_profile_resolved: 'dpo',
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


describe('EvalPanel — Sample Predictions with AlignmentHandler enrichment', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders a green "Preferred chosen" badge when the model picked correctly', async () => {
        mockApi([
            {
                prompt: 'Explain DPO.',
                reference: 'DPO is direct preference optimization.',
                prediction: 'DPO is direct preference optimization.',
                alignment_has_pair: true,
                alignment_chosen: 'DPO is direct preference optimization.',
                alignment_rejected: 'DPO is a programming language.',
                alignment_chosen_sim: 1.0,
                alignment_rejected_sim: 0.2,
                alignment_margin: 0.8,
                alignment_preference_correct: true,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/Preferred chosen/i)).toBeInTheDocument();
        // Sims inline so the researcher can see the actual numbers.
        expect(screen.getByText(/chosen sim 1\.00/)).toBeInTheDocument();
        expect(screen.getByText(/rejected sim 0\.20/)).toBeInTheDocument();
    });

    it('renders a red "Preferred rejected" badge when the model picked wrong', async () => {
        mockApi([
            {
                prompt: 'Explain DPO.',
                reference: 'DPO is direct preference optimization.',
                prediction: 'DPO is a programming language.',
                alignment_has_pair: true,
                alignment_chosen: 'DPO is direct preference optimization.',
                alignment_rejected: 'DPO is a programming language.',
                alignment_chosen_sim: 0.1,
                alignment_rejected_sim: 1.0,
                alignment_margin: -0.9,
                alignment_preference_correct: false,
                row_exact_match: 0.0,
                row_f1: 0.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/Preferred rejected/i)).toBeInTheDocument();
    });

    it('renders the chosen vs rejected disclosure', async () => {
        mockApi([
            {
                prompt: 'Q',
                reference: 'chosen-text',
                prediction: 'chosen-text',
                alignment_has_pair: true,
                alignment_chosen: 'chosen-text',
                alignment_rejected: 'rejected-text',
                alignment_chosen_sim: 1.0,
                alignment_rejected_sim: 0.0,
                alignment_margin: 1.0,
                alignment_preference_correct: true,
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(
            await screen.findByText(/Show chosen vs rejected completions/i),
        ).toBeInTheDocument();
        expect(screen.getAllByText(/chosen/i).length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText('rejected-text')).toBeInTheDocument();
    });

    it('omits the alignment surface when alignment_has_pair is false', async () => {
        mockApi([
            {
                prompt: 'plain-qa-prompt',
                reference: 'plain-qa-reference',
                prediction: 'plain-qa-prediction',
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // Predictions card rendered.
        expect(await screen.findByText(/plain-qa-prompt/i)).toBeInTheDocument();
        // No alignment-specific UI for non-alignment rows.
        expect(screen.queryByText(/Preferred chosen/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/Preferred rejected/i)).not.toBeInTheDocument();
        expect(
            screen.queryByText(/Show chosen vs rejected/i),
        ).not.toBeInTheDocument();
    });
});
