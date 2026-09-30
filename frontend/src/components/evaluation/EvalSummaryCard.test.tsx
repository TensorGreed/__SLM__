import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalSummaryCard from './EvalSummaryCard';
import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';
import type { Project } from '../../types';

const BETTER = {
    project_id: 7,
    experiment_id: 21,
    verdict: 'better',
    message: null,
    headline: {
        metric_id: 'exact_match',
        baseline_value: 0.2,
        trained_value: 0.55,
        absolute_delta: 0.35,
        relative_delta_pct: 175,
        direction: 'improved',
    },
    baseline: { experiment_id: 20, base_model: 'SmolLM2-135M' },
    trained: { experiment_id: 21, experiment_name: 'exp-21' },
    eval_type: 'exact_match',
    evaluated_samples: 20,
    failures: [
        { prompt: 'How do I reset my password?', reference: 'Settings > Reset', prediction: 'Call support' },
    ],
    failed_count: 9,
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

afterEach(() => {
    useProjectStore.setState({ activeProject: null });
});

describe('EvalSummaryCard', () => {
    it('answers better-than-base, by how much, and shows failures', async () => {
        apiMock.get.mockResolvedValue({ data: BETTER });
        render(<EvalSummaryCard projectId={7} experimentId={21} />);

        const card = await screen.findByTestId('eval-summary');
        expect(card).toHaveAttribute('data-verdict', 'better');
        expect(screen.getByText('Better than the base model')).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('0.200 (base)');
        expect(screen.getByTestId('eval-summary-headline')).toHaveTextContent('+0.350 (+175%)');
        expect(screen.getByText(/Run #21 · exp-21 vs base SmolLM2-135M/)).toBeInTheDocument();
        expect(screen.getByTestId('eval-summary-failures')).toHaveTextContent('9 of 20 test examples');
        expect(screen.getByText('Call support')).toBeInTheDocument();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/7/evaluation/summary', { params: { experiment_id: 21 } });
    });

    it('reports the server-resolved run and offers one-click eval when not evaluated', async () => {
        apiMock.get.mockResolvedValue({
            data: { project_id: 7, experiment_id: 33, verdict: 'not_evaluated', headline: null, failures: [] },
        });
        apiMock.post.mockResolvedValue({ data: { id: 1, status: 'queued' } });
        const onResolved = vi.fn();
        render(<EvalSummaryCard projectId={7} experimentId={null} onResolved={onResolved} />);

        const btn = await screen.findByTestId('eval-summary-run');
        expect(onResolved).toHaveBeenCalledWith(33);
        await userEvent.setup().click(btn);
        await waitFor(() =>
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/evaluation/run-heldout?async_job=true',
                expect.objectContaining({ experiment_id: 33, dataset_name: 'test', eval_type: 'exact_match' }),
            ),
        );
    });
});

describe('EvalPanel advanced disclosure', () => {
    function mockPanelApi() {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/evaluation/summary')) return { data: BETTER };
            if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
            if (url.includes('/evaluation/results/')) return { data: [] };
            throw new Error(`unmocked GET ${url}`);
        });
    }

    it('beginner projects see the summary card with Advanced collapsed', async () => {
        useProjectStore.setState({ activeProject: { id: 7, name: 'p', beginner_mode: true } as Project });
        mockPanelApi();
        render(<MemoryRouter><EvalPanel projectId={7} /></MemoryRouter>);

        expect(await screen.findByTestId('eval-summary')).toBeInTheDocument();
        // The server-resolved latest run is auto-selected for the user.
        await waitFor(() => expect(screen.getByRole('button', { name: 'exp-21' })).toHaveClass('btn-primary'));
        const toggle = screen.getByTestId('eval-advanced-toggle');
        expect(toggle).toHaveTextContent('Advanced evaluation');
        expect(screen.queryByRole('tablist', { name: 'Evaluation sections' })).not.toBeInTheDocument();

        await userEvent.setup().click(toggle);
        expect(screen.getByRole('tablist', { name: 'Evaluation sections' })).toBeInTheDocument();
        expect(toggle).toHaveTextContent('Hide advanced evaluation');
    });

    it('non-beginner projects open with Advanced expanded', async () => {
        useProjectStore.setState({ activeProject: { id: 7, name: 'p', beginner_mode: false } as Project });
        mockPanelApi();
        render(<MemoryRouter><EvalPanel projectId={7} /></MemoryRouter>);

        expect(await screen.findByTestId('eval-summary')).toBeInTheDocument();
        expect(screen.getByRole('tablist', { name: 'Evaluation sections' })).toBeInTheDocument();
    });
});
