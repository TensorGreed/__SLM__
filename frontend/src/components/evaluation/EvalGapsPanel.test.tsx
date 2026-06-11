import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

const { apiMock, navigateMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
    navigateMock: vi.fn(),
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return { ...actual, useNavigate: () => navigateMock };
});

import EvalGapsPanel from './EvalGapsPanel';

function renderPanel() {
    return render(
        <MemoryRouter>
            <EvalGapsPanel projectId={42} />
        </MemoryRouter>,
    );
}

const WARN_REPORT = {
    project_id: 42,
    computed_at: '2026-06-11T12:00:00Z',
    overall: 'warn' as const,
    severity_summary: { ok: 1, warn: 2, block: 0 },
    total_signals: 3,
    groups: [
        {
            id: 'eval_gaps',
            title: 'Eval gaps',
            subtitle: 'Does the eval set honestly predict prod performance?',
            signals: [
                {
                    id: 'eval_gaps.archetype_coverage_low',
                    severity: 'ok' as const,
                    headline: 'Gold set lands inside the archetype band.',
                    plain_english: '',
                    why_it_matters: '',
                    suggested_action: null,
                    context: { below_count: 0 },
                },
                {
                    id: 'eval_gaps.no_regression_baseline',
                    severity: 'warn' as const,
                    headline:
                        'You have completed runs but no promoted checkpoint to use as a regression baseline.',
                    plain_english:
                        "You don't have a promoted baseline checkpoint to compare new runs against.",
                    why_it_matters:
                        'Without a baseline every eval reads as standalone.',
                    suggested_action: {
                        kind: 'navigate',
                        label: 'Promote a checkpoint as baseline',
                        target: 'checkpoints-panel',
                        params: {},
                    },
                    context: { has_completed_runs: true },
                },
                {
                    id: 'eval_gaps.train_eval_label_kl_high',
                    severity: 'warn' as const,
                    headline:
                        'Train/eval label-KL = 0.250 nats. Biggest mismatch: pos (90% train vs 50% eval).',
                    plain_english:
                        "The label distribution in train doesn't match eval.",
                    why_it_matters: 'F1 you ship may not predict prod F1.',
                    suggested_action: {
                        kind: 'navigate',
                        label: 'Open Data Studio splits tab',
                        target: 'data-studio-splits',
                        params: {},
                    },
                    context: { kl_nats: 0.25 },
                },
            ],
        },
    ],
};

describe('EvalGapsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        navigateMock.mockReset();
    });

    it('fetches eval gaps on mount and renders all signals', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/projects/42/eval-gaps');
        });
        await waitFor(() => {
            expect(
                screen.getByTestId(
                    'eval-gaps-signal-eval_gaps.archetype_coverage_low',
                ),
            ).toBeInTheDocument();
            expect(
                screen.getByTestId(
                    'eval-gaps-signal-eval_gaps.no_regression_baseline',
                ),
            ).toBeInTheDocument();
            expect(
                screen.getByTestId(
                    'eval-gaps-signal-eval_gaps.train_eval_label_kl_high',
                ),
            ).toBeInTheDocument();
        });
        expect(screen.getByTestId('eval-gaps-overall-badge')).toHaveTextContent(/warning/i);
    });

    it('navigates to checkpoints-panel when the baseline action is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        const action = await screen.findByTestId(
            'eval-gaps-action-eval_gaps.no_regression_baseline',
        );
        await userEvent.click(action);
        expect(navigateMock).toHaveBeenCalledWith(
            '/project/42/training-config#checkpoints-panel',
        );
    });

    it('navigates to data-studio splits when the KL action is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        const action = await screen.findByTestId(
            'eval-gaps-action-eval_gaps.train_eval_label_kl_high',
        );
        await userEvent.click(action);
        expect(navigateMock).toHaveBeenCalledWith(
            '/project/42/data-studio#splits',
        );
    });

    it('toggles why-this-matters for a signal that carries one', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        const whyButton = await screen.findByTestId(
            'eval-gaps-why-eval_gaps.no_regression_baseline',
        );
        expect(
            screen.queryByTestId(
                'eval-gaps-why-text-eval_gaps.no_regression_baseline',
            ),
        ).not.toBeInTheDocument();
        await userEvent.click(whyButton);
        expect(
            screen.getByTestId(
                'eval-gaps-why-text-eval_gaps.no_regression_baseline',
            ),
        ).toHaveTextContent(/standalone/i);
    });
});
