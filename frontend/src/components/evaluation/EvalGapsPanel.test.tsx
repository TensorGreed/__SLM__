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
                    apply_patch_kind: 'regression_baseline_promote_last_green',
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

    it('renders Apply button only for signals carrying apply_patch_kind', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        await waitFor(() => {
            expect(
                screen.getByTestId(
                    'eval-gaps-apply-eval_gaps.no_regression_baseline',
                ),
            ).toBeInTheDocument();
        });
        // The archetype-coverage signal in WARN_REPORT has no
        // apply_patch_kind (it's ok / no-action) → no Apply button.
        expect(
            screen.queryByTestId(
                'eval-gaps-apply-eval_gaps.archetype_coverage_low',
            ),
        ).not.toBeInTheDocument();
    });

    it('opens the patch modal, applies, and re-fetches the report', async () => {
        // Sequence: panel GET (warn) → modal preview POST → apply POST →
        // panel re-GET (ok).
        const OK_REPORT = {
            ...WARN_REPORT,
            overall: 'ok' as const,
            severity_summary: { ok: 3, warn: 0, block: 0 },
            groups: WARN_REPORT.groups.map((g) => ({
                ...g,
                signals: g.signals.map((s) => ({ ...s, severity: 'ok' as const })),
            })),
        };
        const PATCH_PREVIEW = {
            project_id: 42,
            signal_id: 'eval_gaps.no_regression_baseline',
            patch_kind: 'regression_baseline_promote_last_green',
            patch_label: 'Promote last-green checkpoint as baseline',
            plain_english: 'Sets promoted_at on the best checkpoint.',
            before: {
                promoted_checkpoint_id: null,
                promoted_experiment_id: null,
                promoted_step: null,
            },
            after: {
                promoted_checkpoint_id: 17,
                promoted_experiment_id: 5,
                promoted_step: 200,
            },
            candidate: {
                experiment_id: 5,
                experiment_name: 'green-1',
                checkpoint_id: 17,
                checkpoint_step: 200,
                checkpoint_is_best: true,
                pass_rate: 0.85,
            },
            safe_to_apply: true,
        };
        const APPLIED = { ...PATCH_PREVIEW, applied: true };

        apiMock.get
            .mockResolvedValueOnce({ data: WARN_REPORT })
            .mockResolvedValueOnce({ data: OK_REPORT });
        apiMock.post
            .mockResolvedValueOnce({ data: PATCH_PREVIEW })
            .mockResolvedValueOnce({ data: APPLIED });

        renderPanel();
        await userEvent.click(
            await screen.findByTestId(
                'eval-gaps-apply-eval_gaps.no_regression_baseline',
            ),
        );
        // Modal Apply.
        await userEvent.click(
            await screen.findByTestId('eval-patch-apply'),
        );
        // Apply toast surfaces.
        await waitFor(() => {
            expect(
                screen.getByTestId('eval-gaps-apply-toast'),
            ).toBeInTheDocument();
        });
        // Panel re-fetched (2 GETs).
        expect(apiMock.get).toHaveBeenCalledTimes(2);
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
