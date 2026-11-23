import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock, createProjectMock, fetchProjectsMock, deleteProjectMock, storeState } = vi.hoisted(() => ({
  storeState: { projects: [] as Array<Record<string, unknown>> },
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
    projects: storeState.projects,
    totalProjects: storeState.projects.length,
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
import { toast } from '../stores/toastStore';

let toastInfoSpy: ReturnType<typeof vi.spyOn>;

describe('ProjectListPage create modal (Theme 1 Epic 1)', () => {
  beforeEach(() => {
    toastInfoSpy = vi.spyOn(toast, 'info');
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    navigateMock.mockReset();
    createProjectMock.mockReset();
    fetchProjectsMock.mockReset();
    deleteProjectMock.mockReset();
    storeState.projects = [];

    apiMock.get.mockImplementation(async (url: string) => {
      if (url === '/starter-packs/catalog') return { data: { starter_packs: [] } };
      if (url === '/domain-packs') return { data: { packs: [] } };
      if (url === '/domain-profiles') return { data: { profiles: [] } };
      if (url.endsWith('/stats')) return { data: { total_documents: 0, experiment_count: 0 } };
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
    // Submit is disabled until the brief is filled (the name is optional, phase 6).
    expect(
      (screen.getByTestId('create-project-submit') as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('creates a brief-driven project on submit and lands in the workspace', async () => {
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
    // A new project has no data: it lands on Data › Sources, its first step (phase 6).
    expect(navigateMock).toHaveBeenCalledWith('/project/101/data');
  });

  // ── Optional name, derived from the brief (UI simplification phase 6) ──

  const BRIEF = 'Answer customer support questions about accounts, billing and teams. Friendly tone.';

  it('derives the name from the brief when the field is empty and shows it', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    expect(screen.queryByTestId('create-project-derived-name')).toBeNull();
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    expect(screen.getByTestId('create-project-derived-name')).toHaveTextContent(
      'Will be named Answer customer support questions about accounts',
    );
    expect(screen.getByTestId('create-project-name')).toHaveAttribute(
      'placeholder', 'Answer customer support questions about accounts',
    );
    expect((screen.getByTestId('create-project-submit') as HTMLButtonElement).disabled).toBe(false);
  });

  it('payload pin: an empty name sends the derived name exactly like a typed one', async () => {
    const user = userEvent.setup();
    const typed = render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-name'), 'Answer customer support questions about accounts');
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    await user.click(screen.getByTestId('create-project-submit'));
    await waitFor(() => expect(createProjectMock).toHaveBeenCalledTimes(1));
    const typedArgs = createProjectMock.mock.calls[0];
    typed.unmount();

    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    await user.click(screen.getByTestId('create-project-submit'));
    await waitFor(() => expect(createProjectMock).toHaveBeenCalledTimes(2));
    const derivedArgs = createProjectMock.mock.calls[1];

    expect(derivedArgs).toEqual(typedArgs);
    expect(derivedArgs).toEqual([
      'Answer customer support questions about accounts',
      '',
      '',
      null,
      null,
      null,
      {
        beginnerMode: true,
        briefText: BRIEF,
        sampleInputs: [],
        sampleOutputs: [],
        targetProfileId: 'vllm_server',
      },
    ]);
  });

  it('a typed name wins over the derived one; Edit copies the derived name into the field', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    await user.click(screen.getByTestId('create-project-derived-name-edit'));
    const name = screen.getByTestId('create-project-name') as HTMLInputElement;
    expect(name.value).toBe('Answer customer support questions about accounts');
    expect(name).toHaveFocus();
    expect(screen.queryByTestId('create-project-derived-name')).toBeNull();
    await user.clear(name);
    await user.type(name, 'Support bot');
    await user.click(screen.getByTestId('create-project-submit'));
    await waitFor(() => expect(createProjectMock).toHaveBeenCalled());
    expect(createProjectMock.mock.calls[0][0]).toBe('Support bot');
  });

  it('the derived name avoids a name already taken', async () => {
    storeState.projects = [
      { id: 1, name: 'Answer customer support questions about accounts', status: 'active', pipeline_stage: 'ingestion', created_at: '2026-11-01T00:00:00Z', updated_at: '2026-11-01T00:00:00Z' },
      { id: 2, name: 'answer customer support questions about accounts 2', status: 'active', pipeline_stage: 'ingestion', created_at: '2026-11-01T00:00:00Z', updated_at: '2026-11-01T00:00:00Z' },
    ];
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    // Phase 7g (G3): the hint says the brief's name is taken and which name this one gets, with Edit.
    expect(screen.getByTestId('create-project-derived-name-taken')).toHaveTextContent(
      'A project named Answer customer support questions about accounts already exists, so this one will be named Answer customer support questions about accounts 3. Edit to name it.',
    );
    await user.click(screen.getByTestId('create-project-derived-name-edit'));
    expect(screen.getByTestId('create-project-name')).toHaveValue('Answer customer support questions about accounts 3');
  });

  it('phase 7g (G3): an untaken derived name keeps the plain hint, with no "already exists"', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    expect(screen.getByTestId('create-project-derived-name')).toHaveTextContent('Will be named Answer customer support questions about accounts');
    expect(screen.queryByTestId('create-project-derived-name-taken')).toBeNull();
  });

  it('nameFromBrief: a short first sentence whole, else its first clause, else six words, no trailing punctuation', async () => {
    const { nameFromBrief } = await import('./projectNameFromBrief');
    expect(nameFromBrief('')).toBe('');
    expect(nameFromBrief('   \n  ')).toBe('');
    expect(nameFromBrief('classify reviews as positive / neutral / negative.')).toBe('Classify reviews as positive / neutral / negative');
    // Phase 7b: never cut mid-phrase.
    expect(nameFromBrief('Turn a user request into exactly one JSON tool call. Each request lists the tools.'))
      .toBe('Turn a user request into exactly one JSON tool call');
    expect(nameFromBrief('Answer customer support questions about accounts, billing and teams from the help center.'))
      .toBe('Answer customer support questions about accounts');
    expect(nameFromBrief('Read every incoming support ticket and the attached customer history and decide the queue it belongs to'))
      .toBe('Read every incoming support ticket');
    expect(nameFromBrief('Answer questions about the billing of every account that our support desk receives each day'))
      .toBe('Answer questions about the billing');
    expect(nameFromBrief('Tag invoices, quickly')).toBe('Tag invoices, quickly');
    expect(nameFromBrief('Extract fields: date, total')).toBe('Extract fields: date, total');
    expect(nameFromBrief('Summarize (briefly), then rank.')).toBe('Summarize (briefly), then rank');
    expect(nameFromBrief(`${'x'.repeat(80)} more`).length).toBeLessThanOrEqual(60);
  });

  it('nameFromBrief: a terminator ends a sentence only before whitespace or the end', async () => {
    const { nameFromBrief } = await import('./projectNameFromBrief');
    expect(nameFromBrief('Answer questions about U.S. tax law')).toBe('Answer questions about U.S. tax law');
    expect(nameFromBrief('Label mail, e.g. spam or not. Keep it short.')).toBe('Label mail, e.g. spam or not');
    expect(nameFromBrief('\u30c1\u30b1\u30c3\u30c8\u3092\u5206\u985e\u3059\u308b\u3002\u77ed\u304f')).toBe('\u30c1\u30b1\u30c3\u30c8\u3092\u5206\u985e\u3059\u308b');
    expect(nameFromBrief('Docs Q&A for Node.js. Cite the page.')).toBe('Docs Q&A for Node.js');
    expect(nameFromBrief('Support v2.5 API docs')).toBe('Support v2.5 API docs');
    expect(nameFromBrief('Is it spam? Label each mail.')).toBe('Is it spam');
    expect(nameFromBrief('First line\nsecond line')).toBe('First line');
  });

  const nameTaken = (name: string) => ({
    response: {
      status: 400,
      data: { error_code: 'PROJECT_NAME_TAKEN', detail: `Project '${name}' already exists`, message: `Project '${name}' already exists` },
    },
  });

  it('a derived name the server already holds is retried with the next suffix', async () => {
    const base = 'Answer customer support questions about accounts';
    createProjectMock
      .mockRejectedValueOnce(nameTaken(base))
      .mockRejectedValueOnce(nameTaken(`${base} 2`))
      .mockResolvedValueOnce({ id: 102 });
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    await user.click(screen.getByTestId('create-project-submit'));
    await waitFor(() => expect(navigateMock).toHaveBeenCalledWith('/project/102/data'));
    expect(createProjectMock.mock.calls.map((call) => call[0])).toEqual([base, `${base} 2`, `${base} 3`]);
    // Phase 7g (G3): the suffix the server forced is said, not silent.
    expect(toastInfoSpy).toHaveBeenCalledWith(`Named "${base} 3": a project named "${base}" already exists. Rename it in Settings.`, 6000);
    // Every retry sends the same body apart from the name.
    expect(createProjectMock.mock.calls[2].slice(1)).toEqual(createProjectMock.mock.calls[0].slice(1));
  });

  it('a derived name gives up after a bounded number of retries and shows the error', async () => {
    createProjectMock.mockImplementation(async (name: string) => { throw nameTaken(name); });
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    await user.click(screen.getByTestId('create-project-submit'));
    expect(await screen.findByTestId('create-project-error')).toHaveTextContent(
      "Project 'Answer customer support questions about accounts 6' already exists",
    );
    expect(createProjectMock).toHaveBeenCalledTimes(6);
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it('a typed name that is taken is never retried', async () => {
    createProjectMock.mockRejectedValue(nameTaken('Support bot'));
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-name'), 'Support bot');
    await user.type(screen.getByTestId('create-project-brief'), BRIEF);
    await user.click(screen.getByTestId('create-project-submit'));
    expect(await screen.findByTestId('create-project-error')).toHaveTextContent("Project 'Support bot' already exists");
    expect(createProjectMock).toHaveBeenCalledTimes(1);
  });

  it('a brief with no derivable name asks for a name instead of saying leave it empty', async () => {
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    expect(screen.getByTestId('create-project-name-hint')).toHaveTextContent('Leave it empty');
    await user.type(screen.getByTestId('create-project-brief'), '???');
    expect(screen.queryByTestId('create-project-derived-name')).toBeNull();
    expect(screen.getByTestId('create-project-name-hint')).toHaveTextContent('Give the project a name');
    expect(screen.getByTestId('create-project-name-hint')).not.toHaveTextContent('Leave it empty');
    expect((screen.getByTestId('create-project-submit') as HTMLButtonElement).disabled).toBe(true);
  });

  it('toggle reveals the dense advanced fields (sample I/O, base model, domain preset)', async () => {
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
    expect(screen.getByRole('option', { name: 'vLLM Server' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'vLLM Server (large GPU)' })).toBeInTheDocument();
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
    // SFT recommendation = no advice-only note.
    expect(screen.queryByTestId('approach-chip-advice-only')).not.toBeInTheDocument();
  });

  it('renders a tip chip with an advice-only note (no override click) when the recommendation is rag_first', async () => {
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
    // Phase 7b (gap N6): Create sets up a fine-tune either way, so there is no click that changes nothing.
    expect(screen.getByTestId('approach-chip-advice-only')).toHaveTextContent('Create still sets up a fine-tuning project');
    expect(screen.queryByTestId('approach-chip-override')).not.toBeInTheDocument();
    // One confidence number.
    expect(screen.getByTestId('approach-chip-signals')).toHaveTextContent('75% sure');
    expect(chip.textContent?.match(/\d+%/g)).toEqual(['75%']);
  });

  it('phase 7b (N6/N7): a headline that states the analysis confidence shows no second number; the base model is on the main form', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url === '/domain-blueprints/analyze') {
        return {
          data: {
            blueprint: {},
            suggested_recipe_id: 'generic-sft',
            suggested_base_model: 'Org/Suggested-1B',
            validation: { ok: true, errors: [], warnings: [] },
            guidance: { recommended_next_actions: [], unresolved_questions: [] },
            recommended_approach: {
              approach: 'distillation',
              confidence: 0.7,
              headline: 'Distillation could give a smaller, cheaper model (analysis confidence 40%).',
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

    const chip = await screen.findByTestId('approach-chip', {}, { timeout: 2000 });
    expect(chip.textContent?.match(/\d+%/g)).toEqual(['40%']);
    expect(screen.getByTestId('approach-chip-signals')).toHaveTextContent(/^why\?$/);

    // The base model field is on the main step (not under Advanced options), with the task's suggestion.
    const base = screen.getByTestId('create-project-base-model');
    expect(base).toHaveAttribute('placeholder', 'Org/Suggested-1B (suggested from the brief)');
    expect(screen.getByTestId('create-project-base-model-hint')).toHaveTextContent(
      'Leave it empty to use Org/Suggested-1B, suggested from the brief alone.');
    await user.type(base, 'Org/Mine');
    expect(screen.getByTestId('create-project-base-model-hint')).toHaveTextContent('used instead of the suggestion');
  });

  it('review S3: the empty-field hint names the model create will use (a domain preset wins; samples can change it)', async () => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url === '/starter-packs/catalog') {
        return { data: { starter_packs: [{ id: 'pack-a', display_name: 'Pack A', description: 'd', domain: 'x',
          recommended_model_families: [], recommended_models: [], default_base_model_name: 'Org/Pack-3B',
          adapter_task_defaults: {}, evaluation_gate_defaults: {}, safety_compliance_reminders: [],
          target_profile_default: 'vllm_server', catalog_source: 'builtin' }] } };
      }
      if (url === '/domain-packs') return { data: { packs: [] } };
      if (url === '/domain-profiles') return { data: { profiles: [] } };
      return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string) => (url === '/domain-blueprints/analyze'
      ? { data: { blueprint: {}, suggested_recipe_id: 'classification', suggested_base_model: 'Org/Suggested-1B',
        validation: { ok: true, errors: [], warnings: [] }, guidance: { recommended_next_actions: [], unresolved_questions: [] },
        recommended_approach: null } }
      : { data: {} }));
    const user = userEvent.setup();
    render(<ProjectListPage />);
    await user.click(screen.getByRole('button', { name: /\+ New Project/i }));
    await user.type(screen.getByTestId('create-project-brief'), 'Sort each support ticket into billing, bug or account.');
    const hint = screen.getByTestId('create-project-base-model-hint');
    await waitFor(() => expect(hint).toHaveTextContent('Org/Suggested-1B, suggested from the brief alone'), { timeout: 2000 });

    await user.click(screen.getByTestId('create-project-advanced-toggle'));
    // Samples: the create re-reads the brief with them, so the hint does not promise the brief's model.
    await user.type(screen.getByPlaceholderText(/One example per line. Helps the brief analyzer/), 'My card was charged twice');
    expect(hint).toHaveTextContent(
      'Leave it empty to use the model suggested for the task. From the brief alone that is Org/Suggested-1B; Create reads your samples too');
    // A domain preset with a default base model wins over the task's suggestion.
    await user.selectOptions(screen.getByDisplayValue('No domain preset'), 'pack-a');
    expect(hint).toHaveTextContent('Leave it empty to use Org/Pack-3B, the default of the domain preset Pack A.');
    expect(screen.getByTestId('create-project-base-model')).toHaveAttribute('placeholder', "Org/Pack-3B (the domain preset's default)");
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
