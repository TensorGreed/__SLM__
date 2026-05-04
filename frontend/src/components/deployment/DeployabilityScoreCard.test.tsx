import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { DeploymentScore } from '../../types/deployment';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DeployabilityScoreCard from './DeployabilityScoreCard';

const SAMPLE_SCORE: DeploymentScore = {
    id: 1,
    deployment_version_id: 99,
    project_id: 7,
    overall_score: 0.78,
    confidence: 0.65,
    confidence_band: 'medium',
    provenance: 'mixed',
    components: [
        {
            name: 'artifact_compat',
            score: 1.0,
            weight: 0.2,
            weight_normalised: 0.25,
            provenance: 'estimated',
            confidence: 0.85,
            signals: [
                { key: 'deployable_artifact', value: true, ok: true },
            ],
            summary: 'Artifact validation passed.',
        },
        {
            name: 'telemetry_health',
            score: null,
            weight: 0.2,
            weight_normalised: 0,
            provenance: 'measured',
            confidence: 0,
            signals: [],
            summary: 'No telemetry samples ingested yet.',
        },
    ],
    signals_summary: {
        components_present: ['artifact_compat'],
        components_missing: ['telemetry_health'],
    },
    notes: null,
    actor: 'system',
    created_at: '2026-05-04T00:00:00Z',
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('DeployabilityScoreCard', () => {
    it('renders headline score, provenance, and per-component breakdown on load', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_SCORE });
        render(<DeployabilityScoreCard deploymentVersionId={99} />);
        expect(await screen.findByText('0.78')).toBeInTheDocument();
        expect(screen.getByText('mixed')).toBeInTheDocument();
        expect(screen.getByText(/confidence: medium/)).toBeInTheDocument();
        expect(screen.getByText('artifact_compat')).toBeInTheDocument();
        expect(screen.getByText('1.00')).toBeInTheDocument();
        expect(screen.getByText('telemetry_health')).toBeInTheDocument();
        expect(screen.getByText('no signal')).toBeInTheDocument();
    });

    it('renders empty state with Compute now when /score is 404', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 404, data: { detail: 'score_not_found' } },
        });
        render(<DeployabilityScoreCard deploymentVersionId={99} />);
        expect(
            await screen.findByText(/No score has been computed for this deployment yet/i),
        ).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Compute now/i })).toBeInTheDocument();
    });

    it('Compute now POSTs to /score/compute and renders the returned score', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 404, data: { detail: 'score_not_found' } },
        });
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_SCORE });
        render(<DeployabilityScoreCard deploymentVersionId={99} />);

        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: /Compute now/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/deployments/99/score/compute',
                {},
            );
        });
        expect(await screen.findByText('0.78')).toBeInTheDocument();
    });

    it('Recompute button POSTs to /score/compute when a score is already present', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_SCORE });
        apiMock.post.mockResolvedValueOnce({
            data: { ...SAMPLE_SCORE, overall_score: 0.92 },
        });
        render(<DeployabilityScoreCard deploymentVersionId={99} />);

        await screen.findByText('0.78');
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Recompute/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/deployments/99/score/compute',
                {},
            );
        });
        expect(await screen.findByText('0.92')).toBeInTheDocument();
    });
});
