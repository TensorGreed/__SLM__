import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import TrainingPanel from './TrainingPanel';

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
});
