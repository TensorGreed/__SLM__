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
      if (url.includes('/training/model-selection/introspect')) {
        return {
          data: {
            project_id: 1,
            introspection: {
              model_id: 'microsoft/phi-2',
              resolved: true,
              source: 'hf_config',
              architecture: 'causal_lm',
              model_type: 'phi',
              context_length: 2048,
              license: 'mit',
              params_estimate_b: 2.7,
              memory_profile: {
                estimated_min_vram_gb: 6.5,
                estimated_ideal_vram_gb: 10.2,
              },
              warnings: [],
            },
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

    await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
    expect(await screen.findByText('acme/test-model')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply Model + Defaults' }));
    await user.click(screen.getByRole('tab', { name: /Config/i }));

    await waitFor(() => {
      expect(screen.getByDisplayValue('acme/test-model')).toBeInTheDocument();
    });
    expect(apiMock.post).toHaveBeenCalledWith(
      '/projects/1/training/model-selection/telemetry',
      expect.objectContaining({
        action: 'apply',
        selected_model_id: 'acme/test-model',
      }),
    );
  });

  it('introspects base model metadata from setup config', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    await user.click(screen.getByRole('tab', { name: /Config/i }));
    await user.click(screen.getByRole('button', { name: 'Introspect Model' }));

    expect(await screen.findByText('Model Introspection')).toBeInTheDocument();
    expect(await screen.findByText('causal_lm')).toBeInTheDocument();
    expect(apiMock.post).toHaveBeenCalledWith(
      '/projects/1/training/model-selection/introspect',
      expect.objectContaining({
        model_id: 'microsoft/phi-2',
      }),
    );
  });
});
