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
  name: 'smoke-run',
  status: 'pending',
  base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
  training_mode: 'sft',
  config: { num_epochs: 3 },
};

describe('TrainingPanel run dashboard', () => {
  beforeEach(() => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/training/experiments')) return { data: [EXPERIMENT] };
      throw new Error(`unmocked GET ${url}`);
    });
  });

  it('"Back to Experiments" returns to the list and stays there after the refresh', async () => {
    const user = userEvent.setup();
    render(<TrainingPanel projectId={1} hideCreateControls />);

    await user.click(await screen.findByRole('button', { name: 'Dashboard' }));
    const back = await screen.findByRole('button', { name: /Back to Experiments/ });

    await user.click(back);

    // refreshExperiments() resolves after Back; it used to re-open the run
    // from a stale closure. The list must still be showing once it settles.
    await waitFor(() => expect(apiMock.get.mock.calls.filter(([u]) => String(u).endsWith('/training/experiments')).length).toBeGreaterThanOrEqual(2));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole('button', { name: /Back to Experiments/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dashboard' })).toBeInTheDocument();
  });
});
