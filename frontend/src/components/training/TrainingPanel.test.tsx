import { render, screen, waitFor, within } from '@testing-library/react';
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
      if (url.includes('/training/model-selection/benchmark-sweep/history')) {
        return {
          data: {
            count: 1,
            runs: [
              {
                run_id: 'prev-run-1',
                benchmark_mode: 'real_sampled_heuristic',
                tradeoff_summary: { best_balance_model_id: 'acme/test-model' },
                matrix: [{ model_id: 'acme/test-model' }],
              },
            ],
          },
        };
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
      if (url.includes('/training/model-selection/benchmark-sweep')) {
        return {
          data: {
            project_id: 1,
            run_id: 'bench-run-1',
            benchmark_mode: 'real_sampled_heuristic',
            sampled_row_count: 24,
            sampled_avg_tokens: 62.5,
            tradeoff_summary: {
              best_quality_model_id: 'acme/test-model',
              best_speed_model_id: 'acme/test-model',
              best_balance_model_id: 'acme/test-model',
            },
            matrix: [
              {
                rank: 1,
                model_id: 'acme/test-model',
                estimated_accuracy_percent: 88.2,
                estimated_latency_ms: 43.5,
                estimated_throughput_tps: 22.4,
                estimated_quality_score: 0.882,
                benchmark_mode: 'sampled_heuristic',
              },
            ],
          },
        };
      }
      if (url.includes('/training/experiments/preflight/plan')) {
        return {
          data: {
            plan: {
              recommended_profile: 'balanced',
              profile_order: ['safe', 'balanced', 'max_quality'],
              base_preflight: {
                ok: true,
                errors: [],
                warnings: [],
                hints: [],
              },
              suggestions: [
                {
                  profile: 'balanced',
                  title: 'Balanced',
                  description: 'Balanced profile',
                  config: {},
                  changes: [],
                  estimated_vram_risk: 'low',
                  estimated_vram_score: 1,
                  preflight: {
                    ok: true,
                    errors: [],
                    warnings: [],
                    hints: [],
                  },
                },
              ],
            },
          },
        };
      }
      if (url.includes('/training/experiments/preflight')) {
        return {
          data: {
            preflight: {
              ok: false,
              errors: ['base_model has unsupported or unresolved architecture'],
              warnings: [],
              hints: ['Use Training > Config > Introspect Model to verify architecture/context before launching.'],
                capability_summary: {
                  task_type: 'causal_lm',
                  training_mode: 'sft',
                  trainer_backend_requested: 'auto',
                  runtime_backend: 'local',
                runtime: {
                  resolved_runtime_id: 'auto',
                  supported_modalities: ['text'],
                  modalities_declared: false,
                },
                dataset: {
                  adapter_context: {
                    adapter_id: 'default-canonical',
                    adapter_source: 'prepared_manifest',
                    task_profile: 'instruction_sft',
                    task_profile_source: 'prepared_manifest',
                    adapter_modality: 'text',
                  },
                  media_contract: {
                    ok: false,
                    expected_modality: 'vision_language',
                    sampled_rows: 3,
                    media_rows: 3,
                    image_rows: 3,
                    audio_rows: 0,
                    multimodal_rows: 0,
                    missing_local_images: 3,
                    missing_local_audios: 0,
                    remote_image_refs: 0,
                    remote_audio_refs: 0,
                    errors: [
                      'Missing local media assets for 3/3 sampled local media references; multimodal batches may fall back to text-only markers.',
                    ],
                    warnings: [],
                    hints: [
                      'Place media files under the prepared dataset directory (or provide absolute paths) before training.',
                    ],
                  },
                  },
                  model_modality_contract: {
                    architecture: 'encoder',
                    adapter_modality: 'vision_language',
                    supported_modalities: ['text'],
                    ok: false,
                    errors: [
                      "base_model architecture 'encoder' supports modalities text, but adapter resolves modality 'vision_language'.",
                    ],
                    warnings: [],
                    hints: [
                      'Choose a text adapter/task profile for this model architecture, or switch to a multimodal-capable base model.',
                    ],
                  },
                  capability_contract: {
                    task_type: 'causal_lm',
                    training_mode: 'sft',
                    trainer_backend_requested: 'auto',
                  runtime_id: 'auto',
                  runtime_backend: 'local',
                  runtime_known: true,
                  runtime_supported_modalities: ['text'],
                  runtime_modalities_declared: false,
                  adapter_id: 'default-canonical',
                  adapter_task_profile: 'instruction_sft',
                  adapter_modality: 'text',
                },
                model: {
                  id: 'acme/unknown-arch',
                  family: 'unknown',
                  architecture: 'unknown',
                  introspection: {
                    source: 'none',
                  },
                  compatibility_gate: {
                    ok: false,
                    errors: ['unsupported or unresolved architecture'],
                    hints: ['Use Training > Config > Introspect Model'],
                    supported_architectures: ['causal_lm', 'seq2seq', 'classification', 'encoder'],
                  },
                },
              },
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
        apply_source: 'recommendation',
        selected_model_id: 'acme/test-model',
      }),
    );
  });

  it('runs benchmark sweep and applies benchmark winner', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
    await user.click(screen.getByRole('button', { name: 'Run Benchmark Sweep' }));
    expect(await screen.findByRole('button', { name: 'Apply Benchmark Winner' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Apply Benchmark Winner' }));
    await user.click(screen.getByRole('tab', { name: /Config/i }));

    await waitFor(() => {
      expect(screen.getByDisplayValue('acme/test-model')).toBeInTheDocument();
    });

    expect(apiMock.post).toHaveBeenCalledWith(
      '/projects/1/training/model-selection/benchmark-sweep',
      expect.objectContaining({
        target_device: 'laptop',
      }),
    );
    expect(apiMock.post).toHaveBeenCalledWith(
      '/projects/1/training/model-selection/telemetry',
      expect.objectContaining({
        action: 'apply',
        apply_source: 'benchmark',
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

  it('surfaces model compatibility gate diagnostics in preflight panel', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
    await user.click(screen.getByRole('button', { name: 'Run Capability Preflight' }));

    expect(await screen.findByText('Model Compatibility')).toBeInTheDocument();
    expect(await screen.findByText('Blocked')).toBeInTheDocument();
    expect(await screen.findByText('unsupported or unresolved architecture')).toBeInTheDocument();
    expect(await screen.findByText('causal_lm, seq2seq, classification, encoder')).toBeInTheDocument();
  });

  it('shows model-vs-dataset modality contract diagnostics with blocked badge', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
    await user.click(screen.getByRole('button', { name: 'Run Capability Preflight' }));

    const title = await screen.findByText('Model + Dataset Modality');
    const card = title.closest('.training-preflight-contract-card');
    expect(card).not.toBeNull();
    expect(within(card as HTMLElement).getByText('BLOCKED')).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText('vision_language')).toBeInTheDocument();
    expect(
      within(card as HTMLElement).getByText(
        'Choose a text adapter/task profile for this model architecture, or switch to a multimodal-capable base model.',
      ),
    ).toBeInTheDocument();
  });

  it('shows media asset contract diagnostics with blocked badge and counts', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
    await user.click(screen.getByRole('button', { name: 'Run Capability Preflight' }));

    const title = await screen.findByText('Media Asset Contract');
    const card = title.closest('.training-preflight-contract-card');
    expect(card).not.toBeNull();
    expect(within(card as HTMLElement).getByText('BLOCKED')).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText('vision_language')).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText('3 / 3')).toBeInTheDocument();
    expect(
      within(card as HTMLElement).getByText(
        'Missing local media assets for 3/3 sampled local media references; multimodal batches may fall back to text-only markers.',
      ),
    ).toBeInTheDocument();
  });

  it('includes strict multimodal media flag in preflight-plan payload when enabled', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
      />,
    );

    await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
    await user.click(screen.getByRole('checkbox', { name: /Require local media assets for multimodal batches/i }));
    await user.click(screen.getByRole('button', { name: 'Run Preflight Plan' }));

    await waitFor(() => {
      const planCall = apiMock.post.mock.calls.find(
        (call: unknown[]) => String(call[0] || '').includes('/training/experiments/preflight/plan'),
      );
      expect(planCall).toBeTruthy();
      const payload = (planCall as [string, Record<string, unknown>])[1] || {};
      expect(payload).toEqual(
        expect.objectContaining({
          config: expect.objectContaining({
            multimodal_require_media: true,
          }),
        }),
      );
    });
  });

  it('shows recommendation vs benchmark snapshot in preflight and review', async () => {
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
    await user.click(screen.getByRole('button', { name: 'Run Capability Preflight' }));

    expect(await screen.findByText('Model Selection Snapshot')).toBeInTheDocument();
    expect(await screen.findByText('Recommendation Winner')).toBeInTheDocument();
    expect(await screen.findByText('Benchmark Winner')).toBeInTheDocument();
    expect(await screen.findByText(/Winners align/i)).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: /Review/i }));
    expect(await screen.findByText('Preflight Model Selection Snapshot')).toBeInTheDocument();
    expect(await screen.findByText('Recommendation Winner')).toBeInTheDocument();
    expect(await screen.findByText('Benchmark Winner')).toBeInTheDocument();
  });

  it('applies consensus winner from review snapshot action', async () => {
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
    await user.click(screen.getByRole('tab', { name: /Review/i }));

    expect(await screen.findByRole('button', { name: 'Use Consensus Winner' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Use Consensus Winner' }));
    expect(await screen.findByText('Applied consensus winner (acme/test-model) to base model.')).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: /Config/i }));
    await waitFor(() => {
      expect(screen.getByDisplayValue('acme/test-model')).toBeInTheDocument();
    });

    expect(apiMock.post).toHaveBeenCalledWith(
      '/projects/1/training/model-selection/telemetry',
      expect.objectContaining({
        action: 'apply',
        apply_source: 'consensus',
        selected_model_id: 'acme/test-model',
      }),
    );
  });

  it('shows compact model gate summary in essentials quick preflight', async () => {
    const user = userEvent.setup();
    render(
      <TrainingPanel
        projectId={1}
        forceCreateVisible
        hideExperimentList
        setupMode="essentials"
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Run Quick Preflight' }));

    expect(await screen.findByText('Model Gate: Blocked')).toBeInTheDocument();
    expect(await screen.findByText(/acme\/unknown-arch • unknown • source none/i)).toBeInTheDocument();
    expect(await screen.findByText(/Supported: causal_lm, seq2seq, classification, encoder/i)).toBeInTheDocument();
  });
});
