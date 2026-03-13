import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
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

    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/training/autopilot/intent-resolve')) {
        return {
          data: {
            project_id: 1,
            plan: {
              preset_id: 'autopilot.support_qa_safe',
              preset_label: 'Support Q&A Assistant',
              preset_description: 'Answer customer/support questions with grounded, concise replies.',
              task_profile: 'qa',
              confidence: 0.82,
              run_name_suggestion: 'Autopilot - Support Q&A Assistant',
              user_friendly_plan: [
                'We auto-picked a safe starter recipe for your goal.',
                'We keep memory settings conservative to reduce training failures.',
              ],
            },
            safe_training_config: {
              base_model: 'microsoft/phi-2',
              task_type: 'causal_lm',
              batch_size: 2,
            },
            model_recommendation: {
              model_id: 'microsoft/phi-2',
              match_score: 0.91,
            },
            preflight: {
              ok: true,
              errors: [],
              warnings: [],
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
          },
        };
      }
      return { data: {} };
    });

    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/experiments/99/status')) {
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

    await user.type(
      screen.getByLabelText('Plain-language goal'),
      'I want a model that answers customer support questions clearly.',
    );
    await user.click(screen.getByRole('button', { name: 'Build Safe Plan' }));

    expect(await screen.findByText('Safe plan is ready')).toBeInTheDocument();
    expect(await screen.findByText('Support Q&A Assistant')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'One-Click Run' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/1/training/autopilot/one-click-run',
        expect.objectContaining({
          target_device: 'laptop',
        }),
      );
    });
    expect(await screen.findByText('Model Ready')).toBeInTheDocument();
  });

  it('applies suggested intent rewrite before one-click launch', async () => {
    apiMock.post.mockImplementation(async (url: string, payload?: any) => {
      if (url.includes('/training/autopilot/intent-resolve')) {
        const rawIntent = String(payload?.intent || '').toLowerCase();
        if (rawIntent.includes('summarize each support ticket')) {
          return {
            data: {
              project_id: 1,
              plan: {
                preset_id: 'autopilot.summarization_safe',
                preset_label: 'Summarization Assistant',
                preset_description: 'Summarize long content into concise, useful outputs.',
                task_profile: 'summarization',
                confidence: 0.9,
                run_name_suggestion: 'Autopilot - Summarization Assistant',
              },
              intent_clarification: {
                required: false,
                confidence_band: 'high',
                questions: [],
                suggested_intent_examples: [],
                rewrite_suggestions: [],
              },
              launch_guardrails: {
                can_one_click_run: true,
                blockers: [],
                warnings: [],
              },
              safe_training_config: {
                base_model: 'microsoft/phi-2',
                task_type: 'causal_lm',
              },
              preflight: {
                ok: true,
                errors: [],
                warnings: [],
              },
            },
          };
        }
        return {
          data: {
            project_id: 1,
            plan: {
              preset_id: 'autopilot.general_chat_safe',
              preset_label: 'General Assistant',
              preset_description: 'General-purpose assistant behavior with safe starter defaults.',
              task_profile: 'instruction_sft',
              confidence: 0.35,
              run_name_suggestion: 'Autopilot - General Assistant',
            },
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
            launch_guardrails: {
              can_one_click_run: true,
              blockers: [],
              warnings: ['Intent clarification recommended.'],
            },
            safe_training_config: {
              base_model: 'microsoft/phi-2',
              task_type: 'causal_lm',
            },
            preflight: {
              ok: true,
              errors: [],
              warnings: [],
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
            started: false,
            start_result: null,
            start_error: 'Simulated launch stop for test',
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

    await user.type(screen.getByLabelText('Plain-language goal'), 'help');
    await user.click(screen.getByRole('button', { name: 'Build Safe Plan' }));

    expect(await screen.findByText('Clarification recommended')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Ticket summary with actions' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/1/training/autopilot/intent-resolve',
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
