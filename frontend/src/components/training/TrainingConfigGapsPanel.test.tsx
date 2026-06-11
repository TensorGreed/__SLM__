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

import TrainingConfigGapsPanel from './TrainingConfigGapsPanel';

function renderPanel() {
    return render(
        <MemoryRouter>
            <TrainingConfigGapsPanel projectId={42} />
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
            id: 'training_config',
            title: 'Training config',
            subtitle: 'Hyperparameters + base model vs your data scale',
            signals: [
                {
                    id: 'training_config.base_model_undersized',
                    severity: 'ok' as const,
                    headline:
                        'HuggingFaceTB/SmolLM2-135M-Instruct (135M params) is sized reasonably.',
                    plain_english: '',
                    why_it_matters: '',
                    suggested_action: null,
                    context: { current_params_m: 135, params_floor_m: 135 },
                },
                {
                    id: 'training_config.eval_cadence_too_sparse',
                    severity: 'warn' as const,
                    headline:
                        'Eval will only fire ≈ 0 times across ~15 training steps (eval_steps=100).',
                    plain_english:
                        'Trainer will barely check itself — no learning curve.',
                    why_it_matters:
                        'Without intermediate eval steps you cannot detect overfit.',
                    suggested_action: {
                        kind: 'navigate',
                        label: 'Tighten eval cadence (try eval_steps=10)',
                        target: 'training-config',
                        params: { recommended_eval_steps: 10 },
                    },
                    context: {
                        total_steps: 15,
                        eval_steps: 100,
                        eval_observations: 0,
                        recommended_eval_steps: 10,
                    },
                },
                {
                    id: 'training_config.epochs_high_for_small_data',
                    severity: 'warn' as const,
                    headline:
                        '3 epochs over only 20 labelled rows — model sees each row 3 times.',
                    plain_english: 'Small data + many epochs = memorisation.',
                    why_it_matters: 'Model will overfit on each row.',
                    suggested_action: {
                        kind: 'navigate',
                        label: 'Reduce to 3 epochs',
                        target: 'training-config',
                        params: { recommended_num_epochs: 3 },
                    },
                    context: { num_epochs: 3, labelled_rows: 20 },
                },
            ],
        },
    ],
};

describe('TrainingConfigGapsPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        navigateMock.mockReset();
    });

    it('fetches the gap report on mount and renders each signal', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/42/training-config-gaps',
            );
        });
        await waitFor(() => {
            expect(
                screen.getByTestId(
                    'training-config-gaps-signal-training_config.base_model_undersized',
                ),
            ).toBeInTheDocument();
            expect(
                screen.getByTestId(
                    'training-config-gaps-signal-training_config.eval_cadence_too_sparse',
                ),
            ).toBeInTheDocument();
            expect(
                screen.getByTestId(
                    'training-config-gaps-signal-training_config.epochs_high_for_small_data',
                ),
            ).toBeInTheDocument();
        });
        // Overall warn badge surfaces.
        expect(
            screen.getByTestId('training-config-gaps-overall-badge'),
        ).toHaveTextContent(/warning/i);
    });

    it('navigates to the training-config page with eval_steps query when the action chip is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        await waitFor(() => {
            expect(
                screen.getByTestId(
                    'training-config-gaps-action-training_config.eval_cadence_too_sparse',
                ),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId(
                'training-config-gaps-action-training_config.eval_cadence_too_sparse',
            ),
        );
        expect(navigateMock).toHaveBeenCalledWith(
            '/project/42/training-config?recommended_eval_steps=10',
        );
    });

    it('toggles the why-this-matters expander for a signal that carries one', async () => {
        apiMock.get.mockResolvedValueOnce({ data: WARN_REPORT });
        renderPanel();
        const whyButton = await screen.findByTestId(
            'training-config-gaps-why-training_config.eval_cadence_too_sparse',
        );
        // Closed by default.
        expect(
            screen.queryByTestId(
                'training-config-gaps-why-text-training_config.eval_cadence_too_sparse',
            ),
        ).not.toBeInTheDocument();
        await userEvent.click(whyButton);
        expect(
            screen.getByTestId(
                'training-config-gaps-why-text-training_config.eval_cadence_too_sparse',
            ),
        ).toHaveTextContent(/cannot detect overfit/i);
    });
});
