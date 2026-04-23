import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}));

vi.mock('../api/client', () => ({
  default: apiMock,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({ projectId: 77, project: { id: 77, name: 'Test' }, pipelineStatus: null }),
  };
});

import ProjectModelsPage from './ProjectModelsPage';

describe('ProjectModelsPage', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.put.mockReset();

    apiMock.get.mockImplementation(async (url: string, config?: { params?: Record<string, unknown> }) => {
      if (url === '/models') {
        const search = String(config?.params?.search || '').trim().toLowerCase();
        const models = search === 'qwen'
          ? [
            {
              id: 2,
              model_key: 'catalog:qwen:abc',
              source_type: 'catalog',
              source_ref: 'Qwen/Qwen2.5-1.5B-Instruct',
              display_name: 'Qwen 1.5B',
              model_family: 'qwen',
              architecture: 'causal_lm',
              context_length: 32768,
              params_estimate_b: 1.5,
              license: 'apache-2.0',
              peft_support: true,
              full_finetune_support: true,
              supported_task_families: ['instruction_sft', 'qa'],
              training_mode_support: ['sft', 'dpo'],
            },
          ]
          : [
            {
              id: 1,
              model_key: 'catalog:llama:xyz',
              source_type: 'catalog',
              source_ref: 'meta-llama/Llama-3.2-3B-Instruct',
              display_name: 'Llama 3.2 3B',
              model_family: 'llama',
              architecture: 'causal_lm',
              context_length: 8192,
              params_estimate_b: 3,
              license: 'llama',
              peft_support: true,
              full_finetune_support: true,
              supported_task_families: ['instruction_sft', 'qa'],
              training_mode_support: ['sft'],
            },
            {
              id: 2,
              model_key: 'catalog:qwen:abc',
              source_type: 'catalog',
              source_ref: 'Qwen/Qwen2.5-1.5B-Instruct',
              display_name: 'Qwen 1.5B',
              model_family: 'qwen',
              architecture: 'causal_lm',
              context_length: 32768,
              params_estimate_b: 1.5,
              license: 'apache-2.0',
              peft_support: true,
              full_finetune_support: true,
              supported_task_families: ['instruction_sft', 'qa'],
              training_mode_support: ['sft', 'dpo'],
            },
          ];
        return { data: { count: models.length, models } };
      }
      if (url === '/projects/77/models/compatible') {
        return {
          data: {
            project_id: 77,
            count: 1,
            compatible_count: 1,
            models: [
              {
                model_id: 2,
                model_key: 'catalog:qwen:abc',
                compatibility_score: 0.92,
                compatible: true,
                reason_codes: ['TASK_FAMILY_SUPPORTED', 'CHAT_TEMPLATE_OK'],
                why_recommended: [{ code: 'TASK_FAMILY_SUPPORTED', severity: 'pass', message: 'Model supports task family qa.' }],
                why_risky: [{ code: 'TARGET_PROFILE_WARNING', severity: 'warning', message: 'VRAM is close to target baseline.' }],
                recommended_next_actions: ['Use Q4 quantization for safer deployment margin.'],
                model: {
                  id: 2,
                  model_key: 'catalog:qwen:abc',
                  source_type: 'catalog',
                  source_ref: 'Qwen/Qwen2.5-1.5B-Instruct',
                  display_name: 'Qwen 1.5B',
                  model_family: 'qwen',
                  architecture: 'causal_lm',
                  context_length: 32768,
                  params_estimate_b: 1.5,
                  license: 'apache-2.0',
                  peft_support: true,
                  full_finetune_support: true,
                  supported_task_families: ['instruction_sft', 'qa'],
                  training_mode_support: ['sft', 'dpo'],
                },
              },
            ],
          },
        };
      }
      return { data: {} };
    });

    apiMock.post.mockResolvedValue({
      data: {
        model_id: 2,
        model_key: 'catalog:qwen:abc',
        compatibility_score: 0.44,
        compatible: false,
        reason_codes: ['TOKENIZER_METADATA_MISSING', 'CHAT_TEMPLATE_MISSING'],
        why_recommended: [],
        why_risky: [
          { code: 'TOKENIZER_METADATA_MISSING', severity: 'warning', message: 'Tokenizer metadata missing.' },
          { code: 'CHAT_TEMPLATE_MISSING', severity: 'warning', message: 'Chat template missing.' },
        ],
        recommended_next_actions: ['Set tokenizer explicitly.', 'Set chat template explicitly.'],
      },
    });
  });

  it('supports filter/search behavior and refreshes list', async () => {
    const user = userEvent.setup();
    render(<ProjectModelsPage />);

    expect(await screen.findByText('Llama 3.2 3B')).toBeInTheDocument();
    expect(screen.getByText('Qwen 1.5B')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Hardware Fit'), 'laptop');
    await user.type(screen.getByPlaceholderText('model name, architecture...'), 'qwen');
    await user.click(screen.getByRole('button', { name: /Apply Filters/i }));

    await waitFor(() => {
      expect(screen.queryByText('Llama 3.2 3B')).not.toBeInTheDocument();
    });
    expect(screen.getByText('Qwen 1.5B')).toBeInTheDocument();
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith(
        '/models',
        expect.objectContaining({
          params: expect.objectContaining({
            search: 'qwen',
            hardware_fit: 'laptop',
          }),
        }),
      );
    });
  });

  it('renders recommendation explanation cards', async () => {
    render(<ProjectModelsPage />);

    expect(await screen.findByText('Why recommended')).toBeInTheDocument();
    expect(screen.getByText('Model supports task family qa.')).toBeInTheDocument();
    expect(screen.getByText('VRAM is close to target baseline.')).toBeInTheDocument();
  });

  it('shows tokenizer/chat-template warnings after validation', async () => {
    const user = userEvent.setup();
    render(<ProjectModelsPage />);

    const validateButtons = await screen.findAllByRole('button', { name: /Validate For Project/i });
    await user.click(validateButtons[0]);

    expect(await screen.findByText(/Tokenizer\/Template Warning/i)).toBeInTheDocument();
    expect(screen.getByText(/Tokenizer metadata is missing/i)).toBeInTheDocument();
    expect(screen.getByText(/Chat template is missing/i)).toBeInTheDocument();
  });
});
