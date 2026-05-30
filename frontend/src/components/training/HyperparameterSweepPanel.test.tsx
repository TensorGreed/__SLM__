import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import HyperparameterSweepPanel from './HyperparameterSweepPanel';

const SUPPORTED_KINDS = ['wall_clock_seconds', 'lora_r', 'base_params_m'];

const SWEEP_WALL_CLOCK = {
    sweep_id: 'sweep123',
    cell_count: 4,
    completed_count: 4,
    best_label: 'r16-lr0.0002',
    best_experiment_id: 42,
    cost_kind: 'wall_clock_seconds',
    supported_cost_kinds: SUPPORTED_KINDS,
    pareto: {
        quality_key: 'quality_score',
        cost_key: 'cost_score',
        cost_kind: 'wall_clock_seconds',
        optimal_labels: ['r8-lr0.0002', 'r16-lr0.0002'],
    },
    cells: [
        { label: 'r8-lr0.0002', experiment_id: 1, lora_r: 8, learning_rate: 2e-4, base_model: 'm', status: 'completed', final_train_loss: 2.0, final_eval_loss: null, quality_score: 0.333, cost_score: 60, cost_source: 'wall_clock_seconds', pareto_optimal: true, dominated_by: [] },
        { label: 'r8-lr0.0003', experiment_id: 2, lora_r: 8, learning_rate: 3e-4, base_model: 'm', status: 'completed', final_train_loss: 2.4, final_eval_loss: null, quality_score: 0.294, cost_score: 65, cost_source: 'wall_clock_seconds', pareto_optimal: false, dominated_by: ['r8-lr0.0002'] },
        { label: 'r16-lr0.0002', experiment_id: 3, lora_r: 16, learning_rate: 2e-4, base_model: 'm', status: 'completed', final_train_loss: 1.5, final_eval_loss: null, quality_score: 0.4, cost_score: 120, cost_source: 'wall_clock_seconds', pareto_optimal: true, dominated_by: [] },
        { label: 'r16-lr0.0003', experiment_id: 4, lora_r: 16, learning_rate: 3e-4, base_model: 'm', status: 'completed', final_train_loss: 2.6, final_eval_loss: null, quality_score: 0.277, cost_score: 140, cost_source: 'wall_clock_seconds', pareto_optimal: false, dominated_by: ['r8-lr0.0002', 'r16-lr0.0002'] },
    ],
};

// Same cells but the backend re-annotated for cost_kind=lora_r — the
// frontier swaps because the rank axis treats r8 cells as strictly cheaper.
const SWEEP_LORA_R = {
    ...SWEEP_WALL_CLOCK,
    cost_kind: 'lora_r',
    pareto: { ...SWEEP_WALL_CLOCK.pareto, cost_kind: 'lora_r' },
    cells: SWEEP_WALL_CLOCK.cells.map((c) => ({ ...c, cost_score: c.lora_r, cost_source: 'lora_r' })),
};

// Default SWEEP used by older tests that don't care about axis.
const SWEEP = SWEEP_WALL_CLOCK;

describe('HyperparameterSweepPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('launches a sweep with parsed ranks/lrs and renders the Pareto cells', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { sweep_id: 'sweep123', dispatched_cells: 4 } });
        apiMock.get.mockResolvedValue({ data: SWEEP });

        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="HF/SmolLM2-135M" baseConfig={{ num_epochs: 1 }} />);

        await user.click(screen.getByRole('button', { name: /Run sweep/ }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/5/training/sweeps',
                expect.objectContaining({
                    base_model: 'HF/SmolLM2-135M',
                    lora_r_values: [8, 16],
                    learning_rate_values: [2e-4, 3e-4],
                }),
            );
        });
        await waitFor(() =>
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/5/training/sweeps/sweep123',
                { params: { cost_kind: 'wall_clock_seconds' } },
            ),
        );

        // Frontier cells are classified from the backend annotation.
        await waitFor(() => {
            expect(screen.getByTestId('hp-point-r16-lr0.0002').getAttribute('class')).toMatch(/is-optimal/);
        });
        expect(screen.getByTestId('hp-point-r8-lr0.0003').getAttribute('class')).toMatch(/is-dominated/);
        expect(screen.getByText('4/4 cells complete')).toBeInTheDocument();
        // Best cell surfaced.
        expect(screen.getAllByText('r16-lr0.0002').length).toBeGreaterThan(0);
    });

    it('shows a validation error when no ranks/lrs are given', async () => {
        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="HF/SmolLM2-135M" />);
        const ranks = screen.getByPlaceholderText('8, 16, 32');
        await user.clear(ranks);
        await user.click(screen.getByRole('button', { name: /Run sweep/ }));
        expect(await screen.findByText(/at least one LoRA rank/i)).toBeInTheDocument();
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('invokes onOpenExperiment when a cell row is clicked', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { sweep_id: 'sweep123', dispatched_cells: 4 } });
        apiMock.get.mockResolvedValue({ data: SWEEP });
        const onOpen = vi.fn();
        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="m" onOpenExperiment={onOpen} />);
        await user.click(screen.getByRole('button', { name: /Run sweep/ }));
        await waitFor(() => expect(screen.getByTestId('hp-row-r8-lr0.0003')).toBeInTheDocument());
        await user.click(screen.getByTestId('hp-row-r8-lr0.0003'));
        expect(onOpen).toHaveBeenCalledWith(2);
    });

    it('switches the cost axis on picker click and re-fetches with the new kind', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { sweep_id: 'sweep123', dispatched_cells: 4 } });
        // First fetch (default): wall-clock. After picker click: lora_r.
        apiMock.get
            .mockResolvedValueOnce({ data: SWEEP_WALL_CLOCK })
            .mockResolvedValueOnce({ data: SWEEP_LORA_R })
            // Defensive: subsequent polls (if any) return the same payload.
            .mockResolvedValue({ data: SWEEP_LORA_R });

        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="m" />);
        await user.click(screen.getByRole('button', { name: /Run sweep/ }));

        // Default wall-clock axis labelled in the chart.
        await waitFor(() => {
            expect(screen.getByTestId('hp-x-axis-label').textContent).toMatch(/Wall-clock/);
        });

        // Click the LoRA-rank option in the picker.
        await user.click(screen.getByTestId('hp-cost-option-lora_r'));

        // Re-fetch must include the new cost_kind in the query.
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/5/training/sweeps/sweep123',
                { params: { cost_kind: 'lora_r' } },
            );
        });
        // Axis label updates.
        await waitFor(() => {
            expect(screen.getByTestId('hp-x-axis-label').textContent).toMatch(/LoRA rank/);
        });
        // The clicked option is marked selected (data-selected="true").
        expect(
            screen.getByTestId('hp-cost-option-lora_r').getAttribute('data-selected'),
        ).toBe('true');
        expect(
            screen.getByTestId('hp-cost-option-wall_clock_seconds').getAttribute('data-selected'),
        ).toBe('false');
    });

    it('surfaces "cost pending" when the chosen axis has no signal yet', async () => {
        // Cell completed but no wall-clock recorded — the backend returns
        // cost_score=null with cost_source="pending". The panel should drop
        // the cell off the scatter and label it "cost pending" in the row.
        const pendingCost = {
            ...SWEEP_WALL_CLOCK,
            completed_count: 0,
            cells: [{
                label: 'r8-lr0.0002', experiment_id: 1, lora_r: 8, learning_rate: 2e-4,
                base_model: 'm', status: 'completed', final_train_loss: 2.0,
                final_eval_loss: null, quality_score: 0.333,
                cost_score: null, cost_source: 'pending',
                pareto_optimal: false, dominated_by: [],
            }],
        };
        apiMock.post.mockResolvedValueOnce({ data: { sweep_id: 'sweep123', dispatched_cells: 1 } });
        apiMock.get.mockResolvedValue({ data: pendingCost });

        const user = userEvent.setup();
        render(<HyperparameterSweepPanel projectId={5} baseModel="m" />);
        await user.click(screen.getByRole('button', { name: /Run sweep/ }));

        // Cell drops off the scatter (no SVG point).
        await waitFor(() =>
            expect(screen.queryByTestId('hp-point-r8-lr0.0002')).not.toBeInTheDocument(),
        );
        // But surfaces in the row list with "cost pending".
        await waitFor(() => {
            expect(screen.getByTestId('hp-row-r8-lr0.0002').textContent).toMatch(/cost pending/);
        });
    });
});
