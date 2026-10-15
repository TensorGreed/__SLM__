import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalSummaryCard from './EvalSummaryCard';
import SeedShipTable from './SeedShipTable';

const PICK = {
    leader_experiment_id: 40,
    task_type: 'token_classification',
    metric_id: 'span_f1',
    metric_label: 'span F1',
    higher_is_better: true,
    rule: 'highest validation span F1; ties go to the lower run id; test examples, the answer key and external benchmarks were not used to pick',
    picked_experiment_id: 42,
    picked_seed_value: 2,
    n_seeds: 3,
    n_scored: 2,
    children: [
        { experiment_id: 41, seed_value: 1, value: 0.81, picked: false, skipped_reason: null },
        { experiment_id: 42, seed_value: 2, value: 0.84, picked: true, skipped_reason: null },
        { experiment_id: 43, seed_value: 3, value: null, picked: false, skipped_reason: 'no validation span F1 in its training report' },
    ],
    text: 'Picked seed 2 (run #42) of 3 by validation span F1',
    recorded: null,
};

const HEADLINE = { metric_id: 'f1', baseline_value: 0.1, absolute_delta: 0.5, relative_delta_pct: 500, direction: 'improved' as const };
const SEEDS = [
    { experiment_id: 41, seed_value: 1, headline: { ...HEADLINE, trained_value: 0.9 } },
    { experiment_id: 42, seed_value: 2, headline: { ...HEADLINE, trained_value: 0.7 } },
    { experiment_id: 43, seed_value: 3, headline: null },
];

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('SeedShipTable', () => {
    it('lists seeds with validation metric + test score and marks the pick', async () => {
        apiMock.get.mockResolvedValue({ data: PICK });
        render(<SeedShipTable projectId={7} leaderId={40} seeds={SEEDS} />);

        const table = await screen.findByTestId('seed-ship-table');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/7/seed-groups/40/pick');
        expect(table).toHaveTextContent('Validation span F1');
        expect(table).toHaveTextContent('not used to pick');
        expect(screen.getByTestId('seed-ship-row-42')).toHaveAttribute('data-picked', 'true');
        expect(screen.getByTestId('seed-ship-row-42')).toHaveTextContent('★ picked');
        expect(screen.getByTestId('seed-ship-row-42')).toHaveTextContent('0.840');
        expect(screen.getByTestId('seed-ship-row-41')).toHaveTextContent('0.900');
        expect(screen.getByTestId('seed-ship-row-43')).toHaveTextContent('— (no value)');
        expect(screen.getByTestId('seed-ship')).toHaveTextContent('highest validation span F1');
    });

    it('ships the picked seed and shows the export', async () => {
        apiMock.get.mockResolvedValue({ data: PICK });
        apiMock.post.mockResolvedValue({
            data: {
                leader_experiment_id: 40, picked_experiment_id: 42, picked_seed_value: 2,
                selection: { ...PICK, export_id: 9 },
                export: { id: 9, experiment_id: 42, format: 'huggingface', status: 'completed', output_path: '/data/exports/9', run_id: 'r1', error: null },
            },
        });
        render(<SeedShipTable projectId={7} leaderId={40} seeds={SEEDS} />);
        await userEvent.click(await screen.findByTestId('seed-ship-button'));

        expect(apiMock.post).toHaveBeenCalledWith('/projects/7/seed-groups/40/ship', { export_format: 'huggingface' });
        const result = await screen.findByTestId('seed-ship-result');
        expect(result).toHaveTextContent('Export #9 (huggingface, completed) of seed 2 (run #42)');
        expect(result).toHaveTextContent('/data/exports/9');
        expect(screen.getByTestId('seed-ship-export-link')).toHaveAttribute('href', '/project/7/pipeline/export');
    });

    it('says when nothing can be picked', async () => {
        apiMock.get.mockResolvedValue({ data: { ...PICK, picked_experiment_id: null, picked_seed_value: null, children: PICK.children.map((c) => ({ ...c, value: null, picked: false })) } });
        render(<SeedShipTable projectId={7} leaderId={40} />);
        expect(await screen.findByText(/nothing to pick by/)).toBeInTheDocument();
        expect(screen.queryByTestId('seed-ship-button')).toBeNull();
    });
});

describe('EvalSummaryCard seed-group leader', () => {
    it('mounts the seed table for a multi-seed leader only', async () => {
        apiMock.get.mockImplementation((url: string) => Promise.resolve({
            data: url.includes('/seed-groups/')
                ? PICK
                : {
                    project_id: 7, experiment_id: 40, verdict: 'no_comparison', message: null, headline: null,
                    trained: { experiment_id: 40, experiment_name: 'tc · 3 seeds' }, n_seeds: 3, seeds: SEEDS,
                    seed_group_leader: true, failures: [],
                },
        }));
        render(<EvalSummaryCard projectId={7} experimentId={40} />);
        expect(await screen.findByTestId('seed-ship-table')).toBeInTheDocument();
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith('/projects/7/seed-groups/40/pick'));
    });
});
