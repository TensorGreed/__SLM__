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
      if (url.includes('/training/autopilot/plan-v2')) {
        return {
          data: {
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
        };
      }
      if (url.includes('/training/autopilot/one-click-run')) {
        return {
          data: {
            project_id: 1,
            experiment: {
              id: 99,
              name: 'Autopilot - Support Q&A Assistant',
              status: 'pending',
              base_model: 'microsoft/phi-2',
            },
            started: true,
            start_result: { status: 'started' },
            start_error: null,
            applied_intent_rewrite: {
              applied: false,
              original_intent: null,
              rewritten_intent: null,
              source: null,
            },
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
        '/projects/1/training/autopilot/one-click-run',
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
      if (url.includes('/training/autopilot/plan-v2')) {
        return {
          data: {
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
        };
      }
      if (url.includes('/training/autopilot/one-click-run')) {
        return {
          data: {
            project_id: 1,
            experiment: {
              id: 99,
              name: 'Autopilot - Support Q&A Assistant',
              status: 'pending',
              base_model: 'microsoft/phi-2',
            },
            started: true,
            start_result: { status: 'started' },
            start_error: null,
            applied_intent_rewrite: {
              applied: false,
              original_intent: null,
              rewritten_intent: null,
              source: null,
            },
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
      if (url.includes('/training/autopilot/plan-v2')) {
        const rawIntent = String(payload?.intent || '').toLowerCase();
        if (rawIntent.includes('summarize each support ticket')) {
          return {
            data: {
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
          };
        }
        return {
          data: {
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
        };
      }

      if (url.includes('/training/autopilot/one-click-run')) {
        return {
          data: {
            project_id: 1,
            experiment: {
              id: 101,
              name: 'Autopilot - Summarization Assistant',
              status: 'pending',
              base_model: 'microsoft/phi-2',
            },
            started: true,
            start_result: { status: 'started' },
            start_error: null,
            applied_intent_rewrite: {
              applied: true,
              original_intent: 'help',
              rewritten_intent: String(payload?.intent_rewrite || ''),
              source: 'request.intent_rewrite',
            },
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
        '/projects/1/training/autopilot/plan-v2',
        expect.objectContaining({
          intent: 'Summarize each support ticket into 3 bullet points and next actions.',
        }),
      );
    });

    await user.click(screen.getByRole('button', { name: 'One-Click Run' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/1/training/autopilot/one-click-run',
        expect.objectContaining({
          auto_apply_rewrite: true,
          intent_rewrite: 'Summarize each support ticket into 3 bullet points and next actions.',
        }),
      );
    });
    expect(await screen.findByText(/Applied intent rewrite:/i)).toBeInTheDocument();
  });
});
