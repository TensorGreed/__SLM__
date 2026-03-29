import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock, createProjectMock, fetchProjectsMock, deleteProjectMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
  },
  navigateMock: vi.fn(),
  createProjectMock: vi.fn(),
  fetchProjectsMock: vi.fn(),
  deleteProjectMock: vi.fn(),
}));

vi.mock('../api/client', () => ({
  default: apiMock,
}));

vi.mock('../stores/projectStore', () => ({
  useProjectStore: () => ({
    projects: [],
    totalProjects: 0,
    isLoadingProjects: false,
    fetchProjects: fetchProjectsMock,
    createProject: createProjectMock,
    deleteProject: deleteProjectMock,
  }),
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

import ProjectListPage from './ProjectListPage';

describe('ProjectListPage beginner mode wizard', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    navigateMock.mockReset();
    createProjectMock.mockReset();
    fetchProjectsMock.mockReset();
    deleteProjectMock.mockReset();

    apiMock.get.mockImplementation(async (url: string) => {
      if (url === '/starter-packs/catalog') return { data: { starter_packs: [] } };
      if (url === '/domain-packs') return { data: { packs: [] } };
      if (url === '/domain-profiles') return { data: { profiles: [] } };
      return { data: {} };
    });
    createProjectMock.mockResolvedValue({ id: 101 });
  });

  it('supports happy path create-from-brief flow', async () => {
    const user = userEvent.setup();
    apiMock.post.mockImplementation(async (url: string) => {
      if (url === '/domain-blueprints/analyze') {
        return {
          data: {
            blueprint: {
              domain_name: 'Support',
              problem_statement: 'Answer support FAQs.',
              target_user_persona: 'Support agent',
              task_family: 'qa',
              input_modality: 'text',
              expected_output_schema: { type: 'object', properties: { answer: 'string' }, required: ['answer'] },
              expected_output_examples: [{ answer: 'Use reset flow.' }],
              safety_compliance_notes: ['Do not leak secrets.'],
              deployment_target_constraints: { target_profile_id: 'vllm_server' },
              success_metrics: [{ metric_id: 'answer_correctness', label: 'Answer Correctness' }],
              glossary: [{ term: 'task family', plain_language: 'Type of behavior', category: 'general' }],
              confidence_score: 0.74,
              unresolved_assumptions: [],
            },
            validation: { ok: true, errors: [], warnings: [] },
            guidance: { recommended_next_actions: ['Review schema'], unresolved_questions: [] },
          },
        };
      }
      return { data: {} };
    });

    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    await user.type(screen.getByPlaceholderText('e.g. Support FAQ Assistant'), 'Beginner QA Project');
    await user.type(
      screen.getByPlaceholderText('Describe what model behavior you want and what success looks like.'),
      'Build a support assistant for FAQ responses.',
    );
    await user.click(screen.getByRole('button', { name: 'Next' }));

    const textareas = screen.getAllByPlaceholderText(/One example per line/i);
    await user.type(textareas[0], 'How do I reset my password?');
    await user.type(textareas[1], '{{"answer":"Use reset flow."}}');
    await user.click(screen.getByRole('button', { name: 'Next' }));

    expect(await screen.findByText('What The System Understood')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Create From Brief' }));

    await waitFor(() => {
      expect(createProjectMock).toHaveBeenCalledWith(
        'Beginner QA Project',
        'Answer support FAQs.',
        '',
        null,
        null,
        null,
        expect.objectContaining({
          beginnerMode: true,
          briefText: 'Build a support assistant for FAQ responses.',
          targetProfileId: 'vllm_server',
        }),
      );
    });
  });

  it('surfaces unresolved assumptions for ambiguous briefs', async () => {
    const user = userEvent.setup();
    apiMock.post.mockResolvedValue({
      data: {
        blueprint: {
          domain_name: 'General',
          problem_statement: 'General assistant behavior.',
          target_user_persona: 'Team users',
          task_family: 'instruction_sft',
          input_modality: 'text',
          expected_output_schema: { type: 'object', properties: { answer: 'string' }, required: ['answer'] },
          expected_output_examples: [],
          safety_compliance_notes: [],
          deployment_target_constraints: { target_profile_id: 'vllm_server' },
          success_metrics: [{ metric_id: 'helpfulness_score', label: 'Helpfulness' }],
          glossary: [{ term: 'confidence_score', plain_language: 'How complete the brief is', category: 'analysis' }],
          confidence_score: 0.36,
          unresolved_assumptions: ['No sample outputs were provided; output contract may need refinement.'],
        },
        validation: {
          ok: true,
          errors: [],
          warnings: [{ message: 'Blueprint confidence is low due to missing context.' }],
        },
        guidance: {
          recommended_next_actions: ['Add sample outputs'],
          unresolved_questions: ['What output format should be enforced?'],
        },
      },
    });

    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByPlaceholderText('e.g. Support FAQ Assistant'), 'Ambiguous Brief Project');
    await user.type(
      screen.getByPlaceholderText('Describe what model behavior you want and what success looks like.'),
      'I need a model that helps with my domain.',
    );
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    expect(await screen.findByText('Assumptions And Warnings')).toBeInTheDocument();
    expect(
      screen.getByText('No sample outputs were provided; output contract may need refinement.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Blueprint confidence is low due to missing context.')).toBeInTheDocument();
  });

  it('shows API error message when brief analysis fails', async () => {
    const user = userEvent.setup();
    apiMock.post.mockRejectedValue({
      response: { data: { detail: 'Brief analysis failed due to invalid format.' } },
    });

    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByPlaceholderText('e.g. Support FAQ Assistant'), 'Error Case Project');
    await user.type(
      screen.getByPlaceholderText('Describe what model behavior you want and what success looks like.'),
      'This brief should trigger analyze error.',
    );
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    expect(await screen.findByText('Brief analysis failed due to invalid format.')).toBeInTheDocument();
  });
});
