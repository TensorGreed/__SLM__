import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
  navigateMock: vi.fn(),
}));

vi.mock('../api/client', () => ({
  default: apiMock,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => navigateMock,
    useOutletContext: () => ({ projectId: 1 }),
  };
});

import ProjectWizardPage from './ProjectWizardPage';

describe('ProjectWizardPage newbie autopilot', () => {
  beforeEach(() => {
    navigateMock.mockReset();
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.put.mockReset();

    apiMock.put.mockResolvedValue({ data: {} });

    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/training/autopilot/v2/orchestrate') && !url.includes('/run')) {
        return {
          data: {
            project_id: 1,
            dry_run: true,
            strict_mode: false,
            intent: 'support intent',
            effective_target_profile_id: 'vllm_server',
            resolved_target_device: 'server',
            selected_profile: 'balanced',
            guardrails: {
              can_run: true,
              blockers: [],
              warnings: [],
              one_click_fix_available: false,
            },
            readiness: {
              status: 'pass',
              checks: [],
            },
            decision_log: [
              { step: 'initial_planning', status: 'completed', summary: 'Built initial plan.' },
              { step: 'final_guardrails', status: 'ready', summary: 'Guardrails passed.' },
            ],
            repairs: {},
            plan_v2: {
              project_id: 1,
              intent: 'support intent',
              plans: [
                {
                  profile: 'fastest',
                  title: 'Fastest',
                  description: 'Fastest profile',
                  estimate: {
                    estimated_seconds: 240,
                    estimated_cost: 1.2,
                    unit: 'credits',
                    labels: { speed: 'High', quality: 'Medium', cost: 'Low' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
                {
                  profile: 'balanced',
                  title: 'Balanced',
                  description: 'Balanced profile',
                  estimate: {
                    estimated_seconds: 420,
                    estimated_cost: 2.5,
                    unit: 'credits',
                    labels: { speed: 'Medium', quality: 'High', cost: 'Medium' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
                {
                  profile: 'best_quality',
                  title: 'Best Quality',
                  description: 'Best quality profile',
                  estimate: {
                    estimated_seconds: 900,
                    estimated_cost: 5.5,
                    unit: 'credits',
                    labels: { speed: 'Low', quality: 'Highest', cost: 'High' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
              ],
              recommended_profile: 'balanced',
              guardrails: {
                can_run: true,
                blockers: [],
                warnings: [],
                one_click_fix_available: false,
              },
              dataset_readiness: {
                ready: true,
                prepared_row_count: 128,
                blockers: [],
                auto_fixes: [],
              },
              intent_clarification: {
                required: false,
                confidence_band: 'high',
                rewrite_suggestions: [],
              },
            },
            experiment: null,
            started: false,
            start_result: null,
            start_error: null,
          },
        };
      }
      if (url.includes('/training/autopilot/v2/orchestrate/run')) {
        return {
          data: {
            project_id: 1,
            dry_run: false,
            strict_mode: false,
            intent: 'support intent',
            effective_target_profile_id: 'vllm_server',
            resolved_target_device: 'server',
            selected_profile: 'balanced',
            guardrails: {
              can_run: true,
              blockers: [],
              warnings: [],
              one_click_fix_available: false,
            },
            readiness: {
              status: 'pass',
              checks: [],
            },
            decision_log: [
              { step: 'initial_planning', status: 'completed', summary: 'Built initial plan.' },
              { step: 'start_training', status: 'completed', summary: 'Training started.' },
            ],
            repairs: {
              intent_rewrite: { applied: false },
            },
            plan_v2: {
              project_id: 1,
              intent: 'support intent',
              plans: [],
              recommended_profile: 'balanced',
              guardrails: {
                can_run: true,
                blockers: [],
                warnings: [],
              },
              dataset_readiness: {
                ready: true,
                prepared_row_count: 128,
                blockers: [],
                auto_fixes: [],
              },
              intent_clarification: {
                required: false,
                confidence_band: 'high',
                rewrite_suggestions: [],
              },
            },
            experiment: {
              id: 99,
              name: 'Autopilot - Support Q&A Assistant',
              status: 'pending',
              base_model: 'microsoft/phi-2',
            },
            started: true,
            start_result: { status: 'started' },
            start_error: null,
          },
        };
      }
      return { data: {} };
    });

    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/targets/catalog')) {
        return {
          data: [
            {
              id: 'vllm_server',
              name: 'vLLM Server',
              description: 'High-throughput GPU server using vLLM.',
              constraints: {
                min_vram_gb: 16,
                preferred_formats: ['huggingface'],
              },
            },
            {
              id: 'mobile_cpu',
              name: 'Mobile (CPU)',
              description: 'On-device inference on CPU.',
              constraints: {
                max_parameters_billions: 4,
                preferred_formats: ['gguf'],
              },
            },
          ],
        };
      }
      if (url.includes('/training/experiments/') && url.includes('/status')) {
        return {
          data: {
            experiment_id: 99,
            status: 'completed',
            checkpoints: [],
          },
        };
      }
      return { data: {} };
    });
  });

  it('maps plain intent to a safe plan and launches one-click run', async () => {
    const user = userEvent.setup();
    render(<ProjectWizardPage />);

    await screen.findByText('vLLM Server');
    await user.click(screen.getByRole('button', { name: 'Next: Describe Goal' }));

    await waitFor(() => {
      expect(apiMock.put).toHaveBeenCalledWith('/projects/1', {
        target_profile_id: 'vllm_server',
      });
    });

    await user.type(
      screen.getByLabelText('Plain-language goal'),
      'I want a model that answers customer support questions clearly.',
    );
    await user.click(screen.getByRole('button', { name: 'Build Safe Plan' }));

    expect(await screen.findByText('Choose your path')).toBeInTheDocument();
    expect(await screen.findByText('Balanced')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'One-Click Run' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/1/training/autopilot/v2/orchestrate/run',
        expect.objectContaining({
          target_profile_id: 'vllm_server',
          plan_profile: 'balanced',
        }),
      );
    });
    expect(await screen.findByText('Model Ready')).toBeInTheDocument();
  });

  it('renders estimate provenance badges and copy for measured/estimated/simulated plans', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/training/autopilot/v2/orchestrate') && !url.includes('/run')) {
        return {
          data: {
            project_id: 1,
            dry_run: true,
            strict_mode: false,
            intent: 'support intent',
            effective_target_profile_id: 'vllm_server',
            resolved_target_device: 'server',
            selected_profile: 'balanced',
            guardrails: {
              can_run: true,
              blockers: [],
              warnings: [],
              one_click_fix_available: false,
            },
            readiness: {
              status: 'pass',
              checks: [],
            },
            decision_log: [
              { step: 'initial_planning', status: 'completed', summary: 'Built initial plan.' },
            ],
            repairs: {},
            plan_v2: {
              project_id: 1,
              intent: 'support intent',
              plans: [
                {
                  profile: 'fastest',
                  title: 'Fastest',
                  description: 'Fastest profile',
                  estimate: {
                    estimated_seconds: 120,
                    estimated_cost: 0.6,
                    unit: 'credits',
                    metric_source: 'simulated',
                    labels: { speed: 'High', quality: 'Medium', cost: 'Low' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
                {
                  profile: 'balanced',
                  title: 'Balanced',
                  description: 'Balanced profile',
                  estimate: {
                    estimated_seconds: 420,
                    estimated_cost: 2.5,
                    unit: 'credits',
                    labels: { speed: 'Medium', quality: 'High', cost: 'Medium' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
                {
                  profile: 'best_quality',
                  title: 'Best Quality',
                  description: 'Best quality profile',
                  estimate: {
                    estimated_seconds: 900,
                    estimated_cost: 5.5,
                    unit: 'credits',
                    metric_source: 'measured',
                    labels: { speed: 'Low', quality: 'Highest', cost: 'High' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
              ],
              recommended_profile: 'balanced',
              guardrails: {
                can_run: true,
                blockers: [],
                warnings: [],
                one_click_fix_available: false,
              },
              dataset_readiness: {
                ready: true,
                prepared_row_count: 128,
                blockers: [],
                auto_fixes: [],
              },
              intent_clarification: {
                required: false,
                confidence_band: 'high',
                rewrite_suggestions: [],
              },
            },
            started: false,
            start_result: null,
            start_error: null,
          },
        };
      }
      if (url.includes('/training/autopilot/v2/orchestrate/run')) {
        return {
          data: {
            project_id: 1,
            dry_run: false,
            strict_mode: false,
            intent: 'support intent',
            effective_target_profile_id: 'vllm_server',
            resolved_target_device: 'server',
            selected_profile: 'balanced',
            guardrails: { can_run: true, blockers: [], warnings: [] },
            readiness: { status: 'pass', checks: [] },
            decision_log: [],
            repairs: { intent_rewrite: { applied: false } },
            plan_v2: {
              project_id: 1,
              intent: 'support intent',
              plans: [],
              recommended_profile: 'balanced',
              guardrails: { can_run: true, blockers: [], warnings: [] },
              dataset_readiness: { ready: true, blockers: [], auto_fixes: [] },
              intent_clarification: { required: false, rewrite_suggestions: [] },
            },
            experiment: {
              id: 99,
              name: 'Autopilot - Support Q&A Assistant',
              status: 'pending',
              base_model: 'microsoft/phi-2',
            },
            started: true,
            start_result: { status: 'started' },
            start_error: null,
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ProjectWizardPage />);

    await screen.findByText('vLLM Server');
    await user.click(screen.getByRole('button', { name: 'Next: Describe Goal' }));
    await user.type(screen.getByLabelText('Plain-language goal'), 'Summarize each support ticket into short answers.');
    await user.click(screen.getByRole('button', { name: 'Build Safe Plan' }));

    expect(await screen.findByText('Choose your path')).toBeInTheDocument();
    expect(screen.getByText('Simulated')).toBeInTheDocument();
    expect(screen.getByText('Measured')).toBeInTheDocument();
    expect(screen.getByText(/simulated planning values, not measured from real runs/i)).toBeInTheDocument();
    expect(screen.getByText(/heuristic estimates from dataset size and target profile/i)).toBeInTheDocument();
  });

  it('shows blocked launch messaging and disables one-click run when guardrails fail', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/training/autopilot/v2/orchestrate') && !url.includes('/run')) {
        return {
          data: {
            project_id: 1,
            dry_run: true,
            strict_mode: false,
            intent: 'blocked intent',
            effective_target_profile_id: 'edge_gpu',
            resolved_target_device: 'laptop',
            selected_profile: 'balanced',
            guardrails: {
              can_run: false,
              blockers: ['VRAM incompatibility with selected target profile.'],
              warnings: ['Target compatibility check failed.'],
              one_click_fix_available: false,
            },
            readiness: { status: 'warn', checks: [] },
            decision_log: [
              {
                step: 'final_guardrails',
                status: 'blocked',
                summary: 'Autopilot stopped after safe repairs because blockers remain.',
                fixes: [{ label: 'Choose larger target', description: 'Switch to server profile.' }],
              },
            ],
            repairs: {},
            plan_v2: {
              project_id: 1,
              intent: 'blocked intent',
              plans: [
                {
                  profile: 'balanced',
                  title: 'Balanced',
                  description: 'Balanced profile',
                  estimate: {
                    estimated_seconds: 420,
                    estimated_cost: 2.5,
                    unit: 'credits',
                    labels: { speed: 'Medium', quality: 'High', cost: 'Medium' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
              ],
              recommended_profile: 'balanced',
              guardrails: {
                can_run: false,
                blockers: ['VRAM incompatibility with selected target profile.'],
                warnings: ['Target compatibility check failed.'],
                one_click_fix_available: false,
              },
              dataset_readiness: {
                ready: true,
                prepared_row_count: 128,
                blockers: [],
                auto_fixes: [],
              },
              target_compatibility: {
                compatible: false,
                reasons: [
                  'Estimated minimum VRAM (7.6 GB) exceeds target baseline (4 GB) by 3.6 GB.',
                ],
                warnings: [],
                target: {
                  id: 'edge_gpu',
                  name: 'Edge GPU (NVIDIA Jetson/Desktop)',
                },
                model_metadata: {
                  model_id: 'microsoft/phi-2',
                  parameters_billions: 6.0,
                  estimated_min_vram_gb: 7.6,
                  source: 'hf_config',
                },
              },
              intent_clarification: {
                required: false,
                confidence_band: 'high',
                rewrite_suggestions: [],
              },
            },
            experiment: null,
            started: false,
            start_result: null,
            start_error: 'Autopilot blocked run: VRAM incompatibility with selected target profile.',
          },
        };
      }
      if (url.includes('/training/autopilot/v2/orchestrate/run')) {
        return {
          data: {
            started: true,
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ProjectWizardPage />);

    await screen.findByText('vLLM Server');
    await user.click(screen.getByRole('button', { name: 'Next: Describe Goal' }));
    await user.type(screen.getByLabelText('Plain-language goal'), 'Train a support assistant for ticket triage.');
    await user.click(screen.getByRole('button', { name: 'Build Safe Plan' }));

    expect(await screen.findByText('Choose your path')).toBeInTheDocument();
    expect(screen.getByText(/Target compatibility:/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Estimated minimum VRAM \(7.6 GB\) exceeds target baseline \(4 GB\) by 3.6 GB\./i),
    ).toBeInTheDocument();
    expect(screen.getByText('Please resolve blockers before launching.')).toBeInTheDocument();

    const launchButton = screen.getByRole('button', { name: 'One-Click Run' });
    expect(launchButton).toBeDisabled();
    expect(
      apiMock.post.mock.calls.some(([url]) => String(url).includes('/training/autopilot/v2/orchestrate/run')),
    ).toBe(false);
  });

  it('applies suggested intent rewrite before one-click launch', async () => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/targets/catalog')) {
        return {
          data: [
            {
              id: 'vllm_server',
              name: 'vLLM Server',
              description: 'High-throughput GPU server using vLLM.',
              constraints: {
                min_vram_gb: 16,
                preferred_formats: ['huggingface'],
              },
            },
          ],
        };
      }
      if (url.includes('/training/experiments/') && url.includes('/status')) {
        return {
          data: {
            experiment_id: 101,
            status: 'pending',
            checkpoints: [],
          },
        };
      }
      return { data: {} };
    });

    apiMock.post.mockImplementation(async (url: string, payload?: any) => {
      if (url.includes('/training/autopilot/v2/orchestrate') && !url.includes('/run')) {
        const rawIntent = String(payload?.intent || '').toLowerCase();
        if (rawIntent.includes('summarize each support ticket')) {
          return {
            data: {
              project_id: 1,
              dry_run: true,
              strict_mode: false,
              intent: String(payload?.intent || ''),
              effective_target_profile_id: 'vllm_server',
              resolved_target_device: 'server',
              selected_profile: 'balanced',
              guardrails: {
                can_run: true,
                blockers: [],
                warnings: [],
              },
              readiness: { status: 'pass', checks: [] },
              decision_log: [{ step: 'initial_planning', status: 'completed', summary: 'Plan generated.' }],
              repairs: {},
              plan_v2: {
                project_id: 1,
                intent: String(payload?.intent || ''),
                plans: [
                  {
                    profile: 'balanced',
                    title: 'Balanced',
                    description: 'Balanced profile',
                    estimate: {
                      estimated_seconds: 420,
                      estimated_cost: 2.5,
                      unit: 'credits',
                      labels: { speed: 'Medium', quality: 'High', cost: 'Medium' },
                    },
                    preflight: { ok: true, errors: [], warnings: [] },
                  },
                ],
                recommended_profile: 'balanced',
                intent_clarification: {
                  required: false,
                  confidence_band: 'high',
                  questions: [],
                  suggested_intent_examples: [],
                  rewrite_suggestions: [],
                },
                guardrails: {
                  can_run: true,
                  blockers: [],
                  warnings: [],
                },
                dataset_readiness: {
                  ready: true,
                  prepared_row_count: 24,
                  blockers: [],
                  auto_fixes: [],
                },
              },
              started: false,
            },
          };
        }
        return {
          data: {
            project_id: 1,
            dry_run: true,
            strict_mode: false,
            intent: String(payload?.intent || ''),
            effective_target_profile_id: 'vllm_server',
            resolved_target_device: 'server',
            selected_profile: 'balanced',
            guardrails: {
              can_run: true,
              blockers: [],
              warnings: ['Intent clarification recommended.'],
            },
            readiness: { status: 'pass', checks: [] },
            decision_log: [{ step: 'intent_rewrite', status: 'skipped', summary: 'Waiting for rewrite.' }],
            repairs: {},
            plan_v2: {
              project_id: 1,
              intent: String(payload?.intent || ''),
              plans: [
                {
                  profile: 'balanced',
                  title: 'Balanced',
                  description: 'Balanced profile',
                  estimate: {
                    estimated_seconds: 420,
                    estimated_cost: 2.5,
                    unit: 'credits',
                    labels: { speed: 'Medium', quality: 'High', cost: 'Medium' },
                  },
                  preflight: { ok: true, errors: [], warnings: [] },
                },
              ],
              recommended_profile: 'balanced',
              intent_clarification: {
                required: true,
                confidence_band: 'low',
                reason: 'intent confidence is low; intent text is very short',
                questions: ['What exact output should the model produce?'],
                suggested_intent_examples: ['Summarize support tickets into 3 bullet points and next actions.'],
                rewrite_suggestions: [
                  {
                    id: 'rewrite.summarization.support',
                    label: 'Ticket summary with actions',
                    rewritten_intent: 'Summarize each support ticket into 3 bullet points and next actions.',
                    reason: 'Defines expected output format.',
                    recommended: true,
                  },
                ],
              },
              guardrails: {
                can_run: true,
                blockers: [],
                warnings: ['Intent clarification recommended.'],
              },
              dataset_readiness: {
                ready: true,
                prepared_row_count: 24,
                blockers: [],
                auto_fixes: [],
              },
            },
            started: false,
          },
        };
      }

      if (url.includes('/training/autopilot/v2/orchestrate/run')) {
        return {
          data: {
            project_id: 1,
            dry_run: false,
            strict_mode: false,
            intent: 'help',
            effective_target_profile_id: 'vllm_server',
            resolved_target_device: 'server',
            selected_profile: 'balanced',
            guardrails: { can_run: true, blockers: [], warnings: [] },
            readiness: { status: 'pass', checks: [] },
            decision_log: [{ step: 'intent_rewrite', status: 'applied', summary: 'Applied rewrite.' }],
            repairs: {
              intent_rewrite: {
                applied: true,
                original_intent: 'help',
                rewritten_intent: String(payload?.intent_rewrite || ''),
                source: 'request.intent_rewrite',
              },
            },
            plan_v2: {
              project_id: 1,
              intent: 'help',
              plans: [],
              recommended_profile: 'balanced',
              guardrails: { can_run: true, blockers: [], warnings: [] },
              dataset_readiness: { ready: true, blockers: [], auto_fixes: [] },
              intent_clarification: { required: false, rewrite_suggestions: [] },
            },
            experiment: {
              id: 101,
              name: 'Autopilot - Summarization Assistant',
              status: 'pending',
              base_model: 'microsoft/phi-2',
            },
            started: true,
            start_result: { status: 'started' },
            start_error: null,
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ProjectWizardPage />);

    await screen.findByText('vLLM Server');
    await user.click(screen.getByRole('button', { name: 'Next: Describe Goal' }));

    await user.type(screen.getByLabelText('Plain-language goal'), 'help');
    await user.click(screen.getByRole('button', { name: 'Build Safe Plan' }));

    expect(await screen.findByText('Clarification recommended')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Ticket summary with actions' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/1/training/autopilot/v2/orchestrate',
        expect.objectContaining({
          intent: 'Summarize each support ticket into 3 bullet points and next actions.',
          dry_run: true,
        }),
      );
    });

    await user.click(screen.getByRole('button', { name: 'One-Click Run' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/1/training/autopilot/v2/orchestrate/run',
        expect.objectContaining({
          auto_apply_rewrite: true,
          intent_rewrite: 'Summarize each support ticket into 3 bullet points and next actions.',
          dry_run: false,
        }),
      );
    });
    expect(await screen.findByText(/Applied intent rewrite:/i)).toBeInTheDocument();
  });
});
