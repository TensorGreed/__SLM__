import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

vi.mock('../../api/client', () => ({
  default: apiMock,
}));

import TrainingPanel from './TrainingPanel';

describe('TrainingPanel model wizard', () => {
  beforeEach(() => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/preferences')) {
        return { data: { project_id: 1, preferred_plan_profile: 'balanced' } };
      }
      if (url.includes('/training/runtimes')) {
        return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
      }
      if (url.includes('/training/recipes')) {
        return { data: { project_id: 1, recipes: [] } };
      }
      if (url.includes('/training/experiments')) {
        return { data: [] };
      }
      return { data: {} };
    });

    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/training/model-selection/recommend')) {
        return {
          data: {
            project_id: 1,
            recommendations: [
              {
                model_id: 'acme/test-model',
                match_score: 0.91,
                suggested_defaults: {
                  task_type: 'seq2seq',
                  chat_template: 'chatml',
                  use_lora: false,
                  batch_size: 2,
                  max_seq_length: 1024,
                },
              },
            ],
          },
        };
      }
      if (url.includes('/training/experiments/effective-config')) {
        return { data: { resolved_training_config: {} } };
      }
      return { data: {} };
    });
    apiMock.put.mockResolvedValue({ data: {} });
  });

  it('applies recommended model defaults from wizard card', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    expect(await screen.findByText('acme/test-model')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply Model + Defaults' }));

    await waitFor(() => {
      expect(screen.getByDisplayValue('acme/test-model')).toBeInTheDocument();
    });
  });
});
