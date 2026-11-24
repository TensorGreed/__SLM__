import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import TrainingPanel from './TrainingPanel';
import { lastLoss, runEpochState } from './TrainingRunView';

const EXPERIMENT = {
  id: 7,
  name: 'finished-run',
  status: 'completed',
  base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
  training_mode: 'sft',
  config: { num_epochs: 2 },
};

const CHECKPOINTS = {
  experiment_id: 7,
  checkpoints: [
    { experiment_id: 7, step: 100, epoch: 1, loss: 2.9, train_loss: 2.9, eval_loss: 3.2, path: '', is_best: false },
    { experiment_id: 7, step: 108, epoch: 2, loss: 2.67, train_loss: 2.67, eval_loss: 3.1, path: '', is_best: true },
  ],
};

describe('TrainingPanel dashboard of a finished run', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/training/experiments')) return { data: [EXPERIMENT] };
      if (url.endsWith('/training/runs/7/checkpoints')) return { data: CHECKPOINTS };
      throw new Error(`unmocked GET ${url}`);
    });
  });

  it('loads the metric history from the checkpoints (no websocket for a completed run)', async () => {
    const user = userEvent.setup();
    render(<TrainingPanel projectId={1} />);

    await user.click(await screen.findByRole('button', { name: 'Dashboard' }));

    await waitFor(() => expect(screen.getByText('2.00 / 2')).toBeInTheDocument());
    expect(screen.getByText(/All 2 epochs completed/)).toBeInTheDocument();
    // The last checkpoint's losses fill the tiles.
    expect(screen.getAllByText('2.6700').length).toBeGreaterThan(0);
    expect(screen.getAllByText('3.1000').length).toBeGreaterThan(0);
    expect(screen.queryByText('Waiting for first metric...')).not.toBeInTheDocument();
    expect(apiMock.get).toHaveBeenCalledWith('/projects/1/training/runs/7/checkpoints');
  });

  it("phase 7i: a real run shows the trainer's epoch total and keeps its full loss history", async () => {
    // Auto epochs ran 8 epochs / 216 steps on a form that said 3; the trainer's log has a point every 20 steps.
    const history = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200].map((step, i) => ({
      step, epoch: Number((step / 27).toFixed(4)), train_loss: 3.8 - i * 0.25,
    }));
    history.push({ step: 216, epoch: 8, eval_loss: 2.8 } as never);
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/training/experiments')) {
        return { data: [{ ...EXPERIMENT, config: { num_epochs: 3 }, total_epochs: 8, total_steps: 216 }] };
      }
      if (url.endsWith('/training/runs/7/checkpoints')) return { data: { ...CHECKPOINTS, metrics_history: history } };
      throw new Error(`unmocked GET ${url}`);
    });
    const user = userEvent.setup();
    render(<TrainingPanel projectId={1} />);
    await user.click(await screen.findByRole('button', { name: 'Dashboard' }));

    await waitFor(() => expect(screen.getByText('8.00 / 8')).toBeInTheDocument());
    expect(screen.getByText(/All 8 epochs completed/)).toBeInTheDocument();
    expect(screen.queryByText('8.00 / 3')).not.toBeInTheDocument();
    // Every logged train-loss point is on the curve, not just the two checkpoint rows.
    const path = screen.getByTestId('loss-curve-train-path').getAttribute('d') ?? '';
    expect((path.match(/[ML]/g) ?? []).length).toBe(10);
    // Phase 7k: the last row is an eval row (no train loss); the tile keeps the newest train loss, not "--".
    expect(screen.getAllByText('1.5500').length).toBeGreaterThan(0);
    expect(screen.getAllByText('2.8000').length).toBeGreaterThan(0);
  });

  it('phase 7k: a run that finished at a fractional epoch says so (not "Epoch 3 running")', async () => {
    const history = [
      { step: 20, epoch: 1, train_loss: 3.1 },
      { step: 40, epoch: 2, train_loss: 2.7 },
      { step: 50, epoch: 2.5, eval_loss: 2.9 },
    ];
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/training/experiments')) {
        return { data: [{ ...EXPERIMENT, config: { num_epochs: 3 }, total_epochs: 3, final_train_loss: 2.65 }] };
      }
      if (url.endsWith('/training/runs/7/checkpoints')) return { data: { ...CHECKPOINTS, metrics_history: history } };
      throw new Error(`unmocked GET ${url}`);
    });
    const user = userEvent.setup();
    render(<TrainingPanel projectId={1} />);
    await user.click(await screen.findByRole('button', { name: 'Dashboard' }));
    await waitFor(() => expect(screen.getByText(/Finished after 2\.50 epochs/)).toBeInTheDocument());
    expect(screen.queryByText(/Epoch 3 running/)).not.toBeInTheDocument();
    expect(screen.getAllByText('2.7000').length).toBeGreaterThan(0);
  });
});

describe('TrainingRunView helpers (phase 7k)', () => {
  it('runEpochState reads the run status first', () => {
    expect(runEpochState('completed', 2.5, 3)).toBe('Finished after 2.50 epochs');
    expect(runEpochState('completed', 3, 3)).toBe('All 3 epochs completed');
    expect(runEpochState('failed', 1.25, 3)).toBe('Failed after 1.25 epochs');
    expect(runEpochState('cancelled', null, 3)).toBe('Cancelled');
    expect(runEpochState('running', 2.5, 3)).toBe('Epoch 3 running');
    expect(runEpochState('running', null, 3)).toBe('Waiting for first metric...');
  });

  it('lastLoss skips rows without the loss', () => {
    const rows = [{ train_loss: 2.1 }, { train_loss: 1.9 }, { eval_loss: 2.4 }];
    expect(lastLoss(rows, 'train_loss')).toBe(1.9);
    expect(lastLoss(rows, 'eval_loss')).toBe(2.4);
    expect(lastLoss([], 'train_loss')).toBeNull();
  });
});
