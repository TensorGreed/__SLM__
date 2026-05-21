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

describe('ProjectListPage create modal (Theme 1 Epic 1)', () => {
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

  it('opens to a single-question form: name + brief textarea + advanced toggle', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    expect(screen.getByTestId('create-project-name')).toBeInTheDocument();
    expect(screen.getByTestId('create-project-brief')).toBeInTheDocument();
    expect(screen.getByTestId('create-project-advanced-toggle')).toBeInTheDocument();
    // Advanced fields stay hidden by default.
    expect(screen.queryByTestId('create-project-advanced')).not.toBeInTheDocument();
    // Submit is disabled until both name + brief are filled.
    expect(
      (screen.getByTestId('create-project-submit') as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('creates a brief-driven project on submit and navigates to /guide', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    await user.type(screen.getByTestId('create-project-name'), 'Brief Driven Project');
    await user.type(
      screen.getByTestId('create-project-brief'),
      'Answer support FAQs from resolved tickets. Friendly tone.',
    );
    await user.click(screen.getByTestId('create-project-submit'));

    await waitFor(() => {
      expect(createProjectMock).toHaveBeenCalledWith(
        'Brief Driven Project',
        '',
        '',
        null,
        null,
        null,
        expect.objectContaining({
          beginnerMode: true,
          briefText: 'Answer support FAQs from resolved tickets. Friendly tone.',
          targetProfileId: 'vllm_server',
        }),
      );
    });
    expect(navigateMock).toHaveBeenCalledWith('/project/101/guide');
  });

  it('toggle reveals the dense advanced fields (sample I/O, base model, starter pack)', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    await user.click(screen.getByTestId('create-project-advanced-toggle'));
    const advanced = await screen.findByTestId('create-project-advanced');
    expect(advanced).toBeInTheDocument();

    // A handful of the advanced fields should now be reachable.
    expect(screen.getByPlaceholderText(/One example per line\. Helps the brief analyzer/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/HuggingFaceTB\/SmolLM2-135M-Instruct/)).toBeInTheDocument();
    // Deployment target select is inside the advanced block — assert one of its options exists.
    expect(screen.getByRole('option', { name: /vLLM Server/i })).toBeInTheDocument();
  });

  it('passes parsed sample inputs/outputs when the user fills the advanced fields', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    await user.type(screen.getByTestId('create-project-name'), 'With Samples');
    await user.type(
      screen.getByTestId('create-project-brief'),
      'Classify reviews as positive / neutral / negative.',
    );
    await user.click(screen.getByTestId('create-project-advanced-toggle'));

    const sampleInputs = screen.getByPlaceholderText(/One example per line\. Helps the brief analyzer/);
    const sampleOutputs = screen.getByPlaceholderText(/JSON, e\.g\. \{"label":"urgent"\}/i);
    await user.type(sampleInputs, 'I love this product');
    await user.type(sampleOutputs, 'positive');

    await user.click(screen.getByTestId('create-project-submit'));

    await waitFor(() => {
      expect(createProjectMock).toHaveBeenCalledWith(
        'With Samples',
        '',
        '',
        null,
        null,
        null,
        expect.objectContaining({
          sampleInputs: ['I love this product'],
          sampleOutputs: ['positive'],
        }),
      );
    });
  });

  // ── Decision-engine chip (Theme 7) ─────────────────────────

  it('does not render the approach chip until the brief crosses the min length', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    // Type a very short brief — well below the 40-char threshold.
    await user.type(screen.getByTestId('create-project-brief'), 'too short');

    // No chip + no analyze call after a tick.
    await new Promise((r) => setTimeout(r, 700));
    expect(screen.queryByTestId('approach-chip')).not.toBeInTheDocument();
    expect(screen.queryByTestId('approach-chip-loading')).not.toBeInTheDocument();
    expect(apiMock.post).not.toHaveBeenCalledWith(
      '/domain-blueprints/analyze',
      expect.anything(),
    );
  });

  it('renders an SFT chip with no override link when the brief is style/structure-shaped', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url === '/domain-blueprints/analyze') {
        return {
          data: {
            blueprint: { task_family: 'instruction_sft', confidence_score: 0.85 },
            validation: { ok: true, errors: [], warnings: [] },
            guidance: { recommended_next_actions: [], unresolved_questions: [] },
            recommended_approach: {
              approach: 'sft',
              confidence: 0.85,
              headline: 'SFT is the right call — style/structure shaping.',
              rationale: 'You said "respond in our company tone".',
              signals: ['keyword:style:tone'],
              cta_label: 'Proceed with SFT',
            },
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    await user.type(
      screen.getByTestId('create-project-brief'),
      'Respond in our company tone and JSON output schema, friendly voice.',
    );

    const chip = await screen.findByTestId('approach-chip', {}, { timeout: 2000 });
    expect(chip).toHaveAttribute('data-approach', 'sft');
    expect(screen.getByTestId('approach-chip-headline')).toHaveTextContent(
      /SFT is the right call/,
    );
    // SFT recommendation = no override link (no warning to dismiss).
    expect(screen.queryByTestId('approach-chip-override')).not.toBeInTheDocument();
  });

  it('renders a warning chip + "stay with SFT anyway" link when the recommendation is rag_first', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url === '/domain-blueprints/analyze') {
        return {
          data: {
            blueprint: { task_family: 'qa', confidence_score: 0.6 },
            validation: { ok: true, errors: [], warnings: [] },
            guidance: { recommended_next_actions: [], unresolved_questions: [] },
            recommended_approach: {
              approach: 'rag_first',
              confidence: 0.75,
              headline: 'Try RAG first — your brief points at retrieval.',
              rationale: 'You said "answer FAQs from our docs".',
              signals: ['keyword:rag:lookup'],
              cta_label: 'Try RAG first',
            },
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));

    await user.type(
      screen.getByTestId('create-project-brief'),
      'Look up policy answers from our internal docs and respond to users.',
    );

    const chip = await screen.findByTestId('approach-chip', {}, { timeout: 2000 });
    expect(chip).toHaveAttribute('data-approach', 'rag_first');
    expect(screen.getByTestId('approach-chip-headline')).toHaveTextContent(
      /Try RAG first/,
    );
    expect(screen.getByTestId('approach-chip-override')).toBeInTheDocument();
  });

  it('clicking the override link moves the chip to the acknowledged state', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url === '/domain-blueprints/analyze') {
        return {
          data: {
            blueprint: {},
            validation: { ok: true, errors: [], warnings: [] },
            guidance: { recommended_next_actions: [], unresolved_questions: [] },
            recommended_approach: {
              approach: 'distillation',
              confidence: 0.7,
              headline: 'Distillation could give a smaller, cheaper model.',
              rationale: 'You said "smaller cheaper model"',
              signals: ['keywords:distillation:smaller'],
              cta_label: 'Look into distillation',
            },
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(
      screen.getByTestId('create-project-brief'),
      'I want a smaller cheaper model that runs on edge devices, minimal latency.',
    );

    await screen.findByTestId('approach-chip', {}, { timeout: 2000 });
    await user.click(screen.getByTestId('approach-chip-override'));

    // Chip moves into the acknowledged state and the override link disappears.
    expect(screen.queryByTestId('approach-chip')).not.toBeInTheDocument();
    expect(screen.getByTestId('approach-chip-dismissed')).toHaveTextContent(
      /Acknowledged — proceeding with SFT/,
    );
    expect(screen.queryByTestId('approach-chip-override')).not.toBeInTheDocument();
  });

  it('surfaces the API error message inline when create rejects', async () => {
    const user = userEvent.setup();
    createProjectMock.mockRejectedValue({
      response: { data: { detail: 'Brief analysis failed due to invalid format.' } },
    });

    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-name'), 'Error Case');
    await user.type(
      screen.getByTestId('create-project-brief'),
      'This brief should trigger create error.',
    );
    await user.click(screen.getByTestId('create-project-submit'));

    expect(await screen.findByTestId('create-project-error')).toHaveTextContent(
      'Brief analysis failed due to invalid format.',
    );
    expect(navigateMock).not.toHaveBeenCalled();
  });
});
