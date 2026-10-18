import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, mockLocation } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    mockLocation: { pathname: '/', search: '', hash: '' },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../utils/workflowGraphPrefill', () => ({
    loadWorkflowStagePrefill: vi.fn(async () => null),
}));
// SyntheticPanel uses `useLocation` (Phase 5c) to read the Coach
// landing's ?focus_synth_source param + #synth-review-queue hash.
// Stub the hook so the panel can be rendered without a <Router>.
vi.mock('react-router-dom', () => ({
    useLocation: () => mockLocation,
}));

import SyntheticPanel from './SyntheticPanel';

// Map of GET URLs → response data for the bevy of mount-time fetches
// the panel fires (prepared-manifest, playbooks, review-queue,
// cleaning/chunks). We keep these minimal — none of the children
// under test depend on their content other than "the request didn't
// throw."
function installGetRouter(overrides: Record<string, any> = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('prepared-manifest')) {
            return { data: overrides['prepared-manifest'] ?? {} };
        }
        if (url.includes('/synthetic/weak-types')) {
            return { data: overrides['weak-types'] ?? { eval_result_id: null, experiment_id: null, types: [], all_types: [], suggested_rows: 0 } };
        }
        if (url.includes('/synthetic/playbooks')) {
            return {
                data: overrides['playbooks'] ?? {
                    // Default mocks model a project that already has a
                    // recipe applied (the new normal post-1449877 —
                    // brief-driven + magic-create auto-apply at create
                    // time). Tests that want the legacy NULL-recipe
                    // shape override 'playbooks' to set recipe_required.
                    project_id: 1,
                    recipe_id: 'qa-sft',
                    recipe_required: false,
                    playbooks: [
                        { recipe_id: 'qa-sft', mode: 'positives_paraphrase' },
                    ],
                },
            };
        }
        if (url.includes('/synthetic/review-queue')) {
            return {
                data: overrides['review-queue'] ?? {
                    project_id: 1,
                    dataset_id: null,
                    total_rows: 0,
                    total_pending: 0,
                    total_accepted: 0,
                    groups: [],
                    accepted_groups: [],
                },
            };
        }
        if (url.includes('cleaning/chunks')) {
            return {
                data: overrides['cleaning/chunks'] ?? {
                    chunks: [],
                    total: 0,
                },
            };
        }
        if (url.includes('/synthetic/backends/ollama/models')) {
            return {
                data: overrides['ollama-models'] ?? {
                    project_id: 1,
                    models: [],
                    default: null,
                    ollama_available: false,
                },
            };
        }
        if (url.endsWith('/secrets')) {
            return {
                data: overrides['secrets'] ?? { secrets: [] },
            };
        }
        if (url.includes('/synthetic/tasks/')) {
            // Default polling reply: completed immediately with zero
            // rows. Tests that exercise the async path override this.
            return {
                data: overrides['tasks'] ?? {
                    task_id: 'task-1',
                    task_kind: 'qa',
                    status: 'completed',
                    target_rows: 0,
                    rows_so_far: 0,
                    batches_done: 0,
                    batches_total: 0,
                    rows: [],
                    error: null,
                },
            };
        }
        return { data: {} };
    });
}


describe('SyntheticPanel — QA + Conversation parity (USER-SUCCESS Epic 2c)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        installGetRouter();
        // Reset the URL stub each test — Phase 5c plumbing reads it.
        mockLocation.pathname = '/';
        mockLocation.search = '';
        mockLocation.hash = '';
    });

    it('shows the "sample randomly from all cleaned chunks" toggle in QA mode (not just span)', async () => {
        render(<SyntheticPanel projectId={1} />);
        // QA mode is the default for non-span projects.
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        // The toggle used to be span-only — now it's a shared row and
        // must be visible in QA mode too.
        expect(screen.getByTestId('synth-use-all-chunks')).toBeInTheDocument();
    });

    it('lets the QA "pairs to generate" input accept values above 50', async () => {
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        const input = screen.getByTestId('qa-num-pairs') as HTMLInputElement;
        expect(input.getAttribute('max')).toBe('5000');
        // fireEvent.change is more reliable than userEvent.type for
        // number inputs since the latter sometimes prepends to existing
        // digits rather than replacing them.
        fireEvent.change(input, { target: { value: '500' } });
        expect(input.value).toBe('500');
        // Hint reflects the batched path (500 / 50 = 10 batches).
        expect(screen.getByText(/Will run in 10 background batches/i)).toBeInTheDocument();
    });

    it('lets the Conversation "dialogues" input accept values above 20 and shows the batched hint', async () => {
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        // Switch generation mode to conversation.
        const modeSelect = screen.getByTestId('synth-generation-mode') as HTMLSelectElement;
        await userEvent.selectOptions(modeSelect, 'conversation');
        const input = screen.getByTestId('conversation-num-dialogues') as HTMLInputElement;
        expect(input.getAttribute('max')).toBe('5000');
        fireEvent.change(input, { target: { value: '50' } });
        // Per-batch cap for conversations is 5 → 50 / 5 = 10 batches.
        expect(screen.getByText(/Will run in 10 background batches/i)).toBeInTheDocument();
    });

    it('"Refine weak types" reads the last span evaluation and generates rows for the weak types from all passages', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email', 'passport', 'medication'] },
            'weak-types': {
                eval_result_id: 74, experiment_id: 55, dataset_name: 'test', overall_f1: 0.9, suggested_rows: 100,
                types: [{ type: 'passport', f1: 0.0 }, { type: 'medication', f1: 0.6 }, { type: 'provider_name', f1: 0.6667 }, { type: 'national_id', f1: 0.7619 }],
                under_sampled: [{ type: 'passport', support: 3, needed: 800 }, { type: 'email', support: 10, needed: 152 }],
                sizing_gap: 0.05,
                types_to_refine: ['passport', 'medication', 'provider_name', 'national_id', 'email'],
                all_types: ['email', 'passport', 'medication', 'provider_name', 'national_id'],
            },
            tasks: { task_id: 'task-span-9', task_kind: 'span', status: 'completed', target_rows: 100, rows_so_far: 100, batches_done: 9, batches_total: 9, rows: [{ text: 'x', entities: [] }], error: null },
        });
        apiMock.post.mockResolvedValueOnce({ data: { task_id: 'task-span-9', status: 'pending', target_rows: 100, batches_total: 9 } });

        render(<SyntheticPanel projectId={1} />);
        const button = await screen.findByTestId('synthetic-refine-weak-types');
        await userEvent.click(button);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/generate-spans-async',
                expect.objectContaining({
                    target_rows: 100,
                    entity_types: ['email', 'passport', 'medication', 'provider_name', 'national_id'],
                    focus_types: ['passport', 'medication', 'provider_name', 'national_id', 'email'],
                    use_all_chunks: true,
                    backend: 'ollama',
                }),
            );
        });
        expect(screen.getByTestId('synthetic-refine-info')).toHaveTextContent('Weak on run #55: passport (0.00), medication (0.60), provider_name (0.67), national_id (0.76). Too few reference spans to show a 0.05 F1 difference: email (10 of 152). Generating 100 rows that must contain them; every other type in the list is still labelled. New rows reach the answer key through a re-split without “keep test” and a re-key.');
    });

    it('"Write answer key with this model" starts the span_relabel Job with the provider pin', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email', 'passport'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 120, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        const button = await screen.findByTestId('synthetic-write-answer-key');
        expect(button).toHaveTextContent('Write answer key with ollama');
        await userEvent.click(button);
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/relabel-answer-key', {
                split: 'test', backend: 'ollama', entity_types: ['email', 'passport'],
            });
        });
    });

    it('"Relabel training rows" starts the span_relabel Job with target=training', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 121, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        await userEvent.click(await screen.findByTestId('synthetic-relabel-training'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/relabel-answer-key', {
                split: 'test', backend: 'ollama', entity_types: ['email'], target: 'training',
            });
        });
    });

    it('"Relabel test examples" relabels the test split in place (target=split)', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 141, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        await userEvent.click(await screen.findByTestId('synthetic-relabel-test-split'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/relabel-answer-key', {
                split: 'test', backend: 'ollama', entity_types: ['email'], target: 'split',
            });
        });
    });

    it('"Label document passages" starts the span_relabel Job with target=passages', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 131, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        await userEvent.click(await screen.findByTestId('synthetic-label-passages'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/relabel-answer-key', {
                split: 'test', backend: 'ollama', entity_types: ['email'], target: 'passages',
            });
        });
    });

    it('"Relabel training rows" can be limited to one source group', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 133, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        await userEvent.type(await screen.findByTestId('synthetic-relabel-source'), 'document_labels:doc=740');
        await userEvent.click(screen.getByTestId('synthetic-relabel-training'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/relabel-answer-key', {
                split: 'test', backend: 'ollama', entity_types: ['email'], target: 'training', source: 'document_labels:doc=740',
            });
        });
    });

    it('"Label whole documents" starts the span_relabel Job with target=documents', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 132, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        await userEvent.click(await screen.findByTestId('synthetic-label-documents'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/relabel-answer-key', {
                split: 'test', backend: 'ollama', entity_types: ['email'], target: 'documents',
            });
        });
    });

    it('the Batch checkbox is enabled only for an Anthropic pin and sends use_batch', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['email'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: { id: 151, kind: 'span_relabel', status: 'queued' } });
        render(<SyntheticPanel projectId={1} />);
        const user = userEvent.setup();
        const box = await screen.findByTestId('synthetic-relabel-batch') as HTMLInputElement;
        expect(box).toBeDisabled();
        const providerSelect = (await screen.findAllByRole('combobox')).find((el) => within(el).queryByText('Anthropic')) as HTMLElement;
        await user.selectOptions(providerSelect, 'anthropic');
        expect(box).not.toBeDisabled();
        await user.click(box);
        await user.click(screen.getByTestId('synthetic-relabel-training'));
        await waitFor(() => {
            const [url, body] = apiMock.post.mock.calls[apiMock.post.mock.calls.length - 1];
            expect(url).toBe('/projects/1/synthetic/relabel-answer-key');
            expect(body).toMatchObject({ target: 'training', use_batch: true });
            expect(String(body.backend)).toMatch(/^cloud:anthropic:/);
        });
    });

    it('saves a pasted cloud key on the project and clears the field', async () => {
        installGetRouter({});
        apiMock.put.mockResolvedValueOnce({ data: { provider: 'cloud_llm_anthropic', key_name: 'api_key' } });
        render(<SyntheticPanel projectId={1} />);
        const user = userEvent.setup();
        const providerSelect = (await screen.findAllByRole('combobox')).find((el) => within(el).queryByText('Anthropic')) as HTMLElement;
        await user.selectOptions(providerSelect, 'anthropic');
        expect(screen.queryByTestId('synth-save-api-key')).toBeNull();
        await user.type(screen.getByTestId('synth-legacy-api-key'), 'sk-ant-test');
        await user.click(screen.getByTestId('synth-save-api-key'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/1/secrets', {
            provider: 'cloud_llm_anthropic', key_name: 'api_key', value: 'sk-ant-test',
        }));
        await waitFor(() => expect(screen.getByTestId('synth-legacy-saved-key-badge')).toBeInTheDocument());
        expect(screen.getByTestId('synth-legacy-api-key')).toHaveValue('');
    });

    it('uses the QA async endpoint when num_pairs > 50', async () => {
        installGetRouter({
            tasks: {
                task_id: 'task-qa-1',
                task_kind: 'qa',
                status: 'completed',
                target_rows: 75,
                rows_so_far: 75,
                batches_done: 2,
                batches_total: 2,
                rows: [{ question: 'q', answer: 'a' }],
                error: null,
            },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                task_id: 'task-qa-1',
                status: 'pending',
                target_rows: 75,
                batches_total: 2,
            },
        });

        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        const input = screen.getByTestId('qa-num-pairs') as HTMLInputElement;
        fireEvent.change(input, { target: { value: '75' } });
        // Provide source text so the form isn't disabled.
        const sourceTextarea = screen.getByPlaceholderText(/Paste domain text here/i);
        fireEvent.change(sourceTextarea, { target: { value: 'seed text seed text' } });
        await userEvent.click(screen.getByTestId(`synth-legacy-generate`));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/generate-async',
                expect.objectContaining({
                    target_rows: 75,
                    use_all_chunks: false,
                }),
            );
        });
    });

    it('uses the conversation async endpoint when num_dialogues > 5', async () => {
        installGetRouter({
            tasks: {
                task_id: 'task-conv-1',
                task_kind: 'conversation',
                status: 'completed',
                target_rows: 10,
                rows_so_far: 10,
                batches_done: 2,
                batches_total: 2,
                rows: [{ turns: [{ role: 'user', content: 'hi' }] }],
                error: null,
            },
        });
        apiMock.post.mockResolvedValueOnce({
            data: {
                task_id: 'task-conv-1',
                status: 'pending',
                target_rows: 10,
                batches_total: 2,
            },
        });

        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('qa-num-pairs')).toBeInTheDocument();
        });
        const modeSelect = screen.getByTestId('synth-generation-mode') as HTMLSelectElement;
        await userEvent.selectOptions(modeSelect, 'conversation');
        const input = screen.getByTestId('conversation-num-dialogues') as HTMLInputElement;
        fireEvent.change(input, { target: { value: '10' } });
        const sourceTextarea = screen.getByPlaceholderText(/Paste domain text here/i);
        fireEvent.change(sourceTextarea, { target: { value: 'seed text seed text' } });
        await userEvent.click(screen.getByTestId(`synth-legacy-generate`));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/generate-conversations-async',
                expect.objectContaining({
                    target_rows: 10,
                    min_turns: 3,
                    use_all_chunks: false,
                }),
            );
        });
    });

    it('forwards ?focus_synth_source from the URL to SynthReviewQueue (Phase 5c)', async () => {
        // Coach Mode lands the user here with a focus source on the
        // URL. The panel must read it + pass it down so the queue
        // renders its focused banner. We override the review-queue
        // fetch to include a matching pending group; SynthReviewQueue's
        // banner only renders when the source matches a group.
        installGetRouter({
            'review-queue': {
                project_id: 1,
                dataset_id: 7,
                total_rows: 3,
                total_pending: 3,
                total_accepted: 0,
                groups: [
                    {
                        synth_source: 'playbook:classification:class_balance_fill:class=technical',
                        count: 3,
                        truncated: false,
                        rows: [
                            { id: 1, synth_confidence: 0.9, preview: 'a', payload: {} },
                            { id: 2, synth_confidence: 0.85, preview: 'b', payload: {} },
                            { id: 3, synth_confidence: 0.8, preview: 'c', payload: {} },
                        ],
                    },
                ],
                accepted_groups: [],
            },
        });
        mockLocation.search =
            '?focus_synth_source=playbook%3Aclassification%3Aclass_balance_fill%3Aclass%3Dtechnical';
        mockLocation.hash = '#synth-review-queue';

        render(<SyntheticPanel projectId={1} openSection="review-queue" />);

        // The focused banner appears with the right copy.
        await waitFor(() => {
            expect(
                screen.getByTestId('synth-review-queue-focus-banner'),
            ).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('synth-review-queue-focus-source').textContent,
        ).toBe('playbook:classification:class_balance_fill:class=technical');
        const acceptAll = screen.getByTestId('synth-review-queue-focus-accept-all');
        expect(acceptAll.textContent).toMatch(/Accept all 3 rows/);
    });

    it('collapses to only the playbook picker when the project has no recipe (legacy NULL)', async () => {
        // Pre-this-fix, a legacy NULL-recipe project would render the
        // "Pick a recipe first" CTA at the top of the panel AND a
        // fully-functional-looking Generate form 200px below. That
        // form would either fail or silently write recipe-wrong rows.
        // The fix gates the lower sections (review queue + legacy
        // generator form) on the same ``recipe_required`` flag the
        // picker uses, so the legacy user sees one prompt, not two
        // competing surfaces. See /tmp/e2e-screens/01-synthetic-cta.png
        // for the pre-fix visual.
        installGetRouter({
            playbooks: {
                project_id: 1,
                recipe_id: null,
                recipe_required: true,
                playbooks: [],
            },
        });
        render(<SyntheticPanel projectId={1} />);

        // The collapse wrapper renders; PlaybookPickerPanel shows its
        // own NoRecipeEmptyState CTA inside it.
        await waitFor(() => {
            expect(
                screen.getByTestId('synthetic-panel-recipe-required'),
            ).toBeInTheDocument();
        });
        // The picker mounts when the wrapper does and fetches its own catalog.
        expect(
            await screen.findByTestId('playbook-picker-empty-recipe-required'),
        ).toBeInTheDocument();

        // The legacy form's mode select must NOT render.
        expect(
            screen.queryByTestId('synth-generation-mode'),
        ).not.toBeInTheDocument();
        // The Synth review queue wrapper must also be gone.
        expect(
            document.querySelector('#synth-review-queue'),
        ).toBeNull();
    });

    it('renders the Ollama model dropdown in the legacy Q&A form when models are installed', async () => {
        installGetRouter({
            'ollama-models': {
                project_id: 1,
                ollama_available: true,
                default: 'qwen2.5:14b-instruct-q4_K_M',
                models: [
                    { name: 'qwen2.5:14b-instruct-q4_K_M', size_bytes: 1, parameter_size: '14.8B', family: 'qwen2' },
                    { name: 'llama3:latest', size_bytes: 1, parameter_size: '8.0B', family: 'llama' },
                ],
            },
        });
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            // Provider defaults to ollama, so the dropdown should
            // render automatically once the models endpoint resolves.
            expect(screen.getByTestId('synth-legacy-ollama-model')).toBeInTheDocument();
        });
        const select = screen.getByTestId('synth-legacy-ollama-model') as HTMLSelectElement;
        // Both installed models appear with parameter-size badges.
        const text = select.textContent ?? '';
        expect(text).toMatch(/qwen2\.5:14b/);
        expect(text).toMatch(/14\.8B/);
        expect(text).toMatch(/llama3:latest/);
        // Server's auto-pick won the initial selection (user's saved
        // 'llama3' free-text default isn't in the installed list).
        expect(select.value).toBe('qwen2.5:14b-instruct-q4_K_M');
    });

    it('exposes OpenAI / Deepseek / Anthropic as first-class provider options with pre-filled URLs', async () => {
        installGetRouter();
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-legacy-provider')).toBeInTheDocument();
        });
        const select = screen.getByTestId('synth-legacy-provider') as HTMLSelectElement;
        // All four cloud-or-local providers present (custom still
        // available for power users who point at something else).
        const text = select.textContent ?? '';
        expect(text).toMatch(/Ollama/);
        expect(text).toMatch(/OpenAI/);
        expect(text).toMatch(/Deepseek/);
        expect(text).toMatch(/Anthropic/);
        expect(text).toMatch(/Custom Endpoint/);

        // Pick Deepseek → URL pre-fills to the Deepseek host + model
        // suggestion populates. Locked URL badge appears.
        await userEvent.selectOptions(select, 'deepseek');
        const urlInput = screen.getByTestId('synth-legacy-api-url') as HTMLInputElement;
        await waitFor(() => {
            expect(urlInput.value).toMatch(/api\.deepseek\.com/);
        });
        expect(urlInput.readOnly).toBe(true);

        // Pick Anthropic → URL pre-fills to the Anthropic messages
        // endpoint (backend detects this + dispatches to
        // call_anthropic_chat).
        await userEvent.selectOptions(select, 'anthropic');
        await waitFor(() => {
            expect(urlInput.value).toMatch(/api\.anthropic\.com/);
        });
    });

    it('shows the "saved on project" badge when a cloud provider has a key already', async () => {
        installGetRouter({
            secrets: {
                secrets: [
                    {
                        provider: 'cloud_llm_deepseek',
                        key_name: 'api_key',
                        value_hint: 'sk-...AB',
                    },
                ],
            },
        });
        render(<SyntheticPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('synth-legacy-provider')).toBeInTheDocument();
        });
        // Provider dropdown labels the deepseek option with the saved-key check.
        const select = screen.getByTestId('synth-legacy-provider') as HTMLSelectElement;
        const options = Array.from(select.options).map((o) => o.textContent);
        expect(options.find((t) => t?.includes('Deepseek'))).toMatch(/saved/);
        // After selecting it, the inline 'saved on project' badge renders.
        await userEvent.selectOptions(select, 'deepseek');
        await waitFor(() => {
            expect(screen.getByTestId('synth-legacy-saved-key-badge')).toBeInTheDocument();
        });
        // The API-key input shows the 'using saved key' placeholder
        // instead of forcing the user to paste anything.
        const keyInput = screen.getByTestId('synth-legacy-api-key') as HTMLInputElement;
        expect(keyInput.placeholder).toMatch(/saved key/);
    });

    it('falls back to the free-text model input when Ollama is unreachable', async () => {
        installGetRouter({
            'ollama-models': {
                project_id: 1,
                ollama_available: false,
                default: null,
                models: [],
                error: 'connection refused',
            },
        });
        render(<SyntheticPanel projectId={1} />);
        // Default form should mount + the free-text input is the
        // only model-picker rendered (no dropdown when daemon is
        // down — picker would be empty + misleading).
        await waitFor(() => {
            expect(screen.getByTestId('synth-generation-mode')).toBeInTheDocument();
        });
        expect(
            screen.queryByTestId('synth-legacy-ollama-model'),
        ).not.toBeInTheDocument();
    });

    it('span mode: "Check prompt against benchmarks" dry-runs the types with the conventions, and a refusal offers the override', async () => {
        installGetRouter({
            'prepared-manifest': { task_profile: 'structured_extraction', output_schema: { scoring_mode: 'span_set' }, entity_types: ['employee_id', 'plate'] },
        });
        apiMock.post.mockResolvedValueOnce({ data: {
            clean: false, offending_total: 1, values_checked: 1480, benchmarks: ['gretel-pii'], fields: ['definition:employee_id', 'entity_types'],
            offending: [{ value: 'EMP-48213', type: 'employee_id', benchmarks: ['gretel-pii'], fields: ['definition:employee_id'] }],
        } });
        render(<SyntheticPanel projectId={1} />);
        await userEvent.click(await screen.findByTestId('synthetic-prompt-check'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/1/synthetic/prompt-check', {
            fields: { entity_types: ['employee_id', 'plate'] }, include_conventions: true,
        }));
        expect(screen.getByTestId('synthetic-prompt-check-result')).toHaveTextContent('EMP-48213 (employee_id, in the definition of employee_id)');

        apiMock.post.mockRejectedValueOnce({ response: { status: 400, data: {
            error_code: 'PROMPT_CARRIES_BENCHMARK_VALUES', detail: 'The prompt carries 1 value(s) from a benchmark\'s gold rows: EMP-48213 (employee_id, gretel-pii).',
            message: 'The prompt carries 1 value(s) from a benchmark\'s gold rows: EMP-48213 (employee_id, gretel-pii).',
            metadata: { offending: [{ value: 'EMP-48213', type: 'employee_id', benchmarks: ['gretel-pii'], fields: ['definition:employee_id'] }], override_field: 'ignore_benchmark_values' },
        } } });
        fireEvent.change(screen.getByPlaceholderText(/Paste domain text here/i), { target: { value: 'seed text seed text' } });
        await userEvent.click(screen.getByTestId('synth-legacy-generate'));
        expect(await screen.findByTestId('synthetic-prompt-blocked')).toHaveTextContent('EMP-48213 (employee_id, gretel-pii)');
        await userEvent.click(screen.getByTestId('synthetic-ignore-benchmark-values'));
        apiMock.post.mockResolvedValueOnce({ data: { rows: [], count: 0 } });
        await userEvent.click(screen.getByTestId('synth-legacy-generate'));
        await waitFor(() => expect(apiMock.post).toHaveBeenLastCalledWith(
            '/projects/1/synthetic/generate-spans',
            expect.objectContaining({ entity_types: ['employee_id', 'plate'], ignore_benchmark_values: true }),
            expect.anything(),
        ));
    });
});
