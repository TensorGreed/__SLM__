import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
    DeploymentDriftCheck,
    DeploymentDriftHistoryResponse,
} from '../../types/deployment';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DriftPanel from './DriftPanel';

function makeCheck(overrides: Partial<DeploymentDriftCheck>): DeploymentDriftCheck {
    return {
        id: 1,
        deployment_version_id: 99,
        project_id: 7,
        gold_set_id: 12,
        gold_set_version_id: 1,
        baseline_experiment_id: 4,
        baseline_eval_result_id: 5,
        eval_type: 'exact_match',
        baseline_pass_rate: 1.0,
        current_pass_rate: 0.5,
        delta: -0.5,
        tolerance: 0.05,
        drift_detected: true,
        samples_evaluated: 10,
        samples_failed: 0,
        samples_skipped: 0,
        mode: 'offline',
        notes: null,
        actor: 'system',
        per_row_results: [],
        summary: {},
        created_at: '2026-05-04T00:00:00Z',
        ...overrides,
    };
}

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('DriftPanel', () => {
    it('renders drift verdict + history when checks exist', async () => {
        const history: DeploymentDriftHistoryResponse = {
            deployment_version_id: 99,
            limit: 50,
            drift_checks: [
                makeCheck({
                    id: 11,
                    drift_detected: true,
                    delta: -0.4,
                    current_pass_rate: 0.6,
                }),
                makeCheck({
                    id: 12,
                    drift_detected: false,
                    delta: 0.01,
                    current_pass_rate: 0.99,
                }),
            ],
        };
        apiMock.get.mockResolvedValueOnce({ data: history });
        render(<DriftPanel deploymentVersionId={99} />);

        // The latest verdict + per-row badges all carry "drift" or
        // "within tolerance"; just confirm at least one of each is rendered.
        expect((await screen.findAllByText('drift')).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/within tolerance/i).length).toBeGreaterThan(0);
        // Pass-rates are formatted with .toFixed(3) — search both rows.
        expect(screen.getAllByText(/0\.990|0\.600/).length).toBeGreaterThan(0);
    });

    it('shows empty state when there are no drift checks', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                deployment_version_id: 99,
                limit: 50,
                drift_checks: [],
            },
        });
        render(<DriftPanel deploymentVersionId={99} />);
        expect(await screen.findByText(/No drift checks yet/i)).toBeInTheDocument();
    });

    it('launcher posts drift check with parsed predictions array', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                deployment_version_id: 99,
                limit: 50,
                drift_checks: [],
            },
        });
        apiMock.post.mockResolvedValueOnce({ data: makeCheck({}) });
        // After post, the panel re-fetches the history.
        apiMock.get.mockResolvedValueOnce({
            data: {
                deployment_version_id: 99,
                limit: 50,
                drift_checks: [makeCheck({ id: 7 })],
            },
        });

        render(<DriftPanel deploymentVersionId={99} />);
        await screen.findByText(/No drift checks yet/i);

        const user = userEvent.setup();
        // Open the launcher
        await user.click(screen.getByText(/Run a drift check/i));

        await user.type(screen.getByLabelText(/Gold-set id/i), '42');
        await user.clear(screen.getByLabelText(/Tolerance/i));
        await user.type(screen.getByLabelText(/Tolerance/i), '0.10');
        const predictionsInput = screen.getByLabelText(/Predictions JSON/i);
        await user.click(predictionsInput);
        await user.paste('[{"row_id": 1, "prediction": "yes"}]');

        await user.click(screen.getByRole('button', { name: /Run drift check/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/deployments/99/drift/check',
                {
                    gold_set_id: 42,
                    tolerance: 0.10,
                    predictions: [{ row_id: 1, prediction: 'yes' }],
                },
            );
        });
    });

    it('launcher rejects malformed predictions JSON inline', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: { deployment_version_id: 99, limit: 50, drift_checks: [] },
        });
        render(<DriftPanel deploymentVersionId={99} />);
        await screen.findByText(/No drift checks yet/i);

        const user = userEvent.setup();
        await user.click(screen.getByText(/Run a drift check/i));
        await user.type(screen.getByLabelText(/Gold-set id/i), '42');
        const predictionsInput = screen.getByLabelText(/Predictions JSON/i);
        await user.click(predictionsInput);
        await user.paste('not json');

        await user.click(screen.getByRole('button', { name: /Run drift check/i }));

        expect(await screen.findByRole('alert')).toHaveTextContent(/not valid/i);
        expect(apiMock.post).not.toHaveBeenCalled();
    });
});
