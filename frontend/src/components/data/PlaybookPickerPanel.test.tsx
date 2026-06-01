import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import type { ReactElement } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import PlaybookPickerPanel from './PlaybookPickerPanel';

/** Render the panel inside a MemoryRouter so the
 *  trainability-forecast prefill reader (``useLocation``) can mount.
 *  Pass ``search`` to simulate a deep-link from the forecast panel. */
function renderPanel(element: ReactElement, opts: { search?: string } = {}) {
    const path = `/route${opts.search ?? ''}`;
    return render(
        <MemoryRouter initialEntries={[path]}>
            {element}
        </MemoryRouter>,
    );
}


describe('PlaybookPickerPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders the available paraphrase mode for a classification recipe', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker')).toBeInTheDocument();
        });
        // Header notes the recipe + the count.
        expect(screen.getByText(/classification/)).toBeInTheDocument();
        expect(screen.getByText(/1 mode available/)).toBeInTheDocument();
        // The single paraphrase mode appears with its label and hint.
        const modeOption = screen.getByTestId('playbook-picker-mode-positives_paraphrase');
        expect(modeOption.textContent).toContain('Paraphrase positives');
        // Auto-selected on first render.
        const radio = modeOption.querySelector('input[type="radio"]') as HTMLInputElement;
        expect(radio.checked).toBe(true);
    });

    it('shows an empty-state message when no playbooks are registered', async () => {
        apiMock.get.mockResolvedValue({
            data: { project_id: 1, recipe_id: null, playbooks: [] },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-empty')).toBeInTheDocument();
        });
        expect(screen.getByText(/Select a recipe/)).toBeInTheDocument();
    });

    it('renders the shared "pick a recipe first" CTA when the server flags recipe_required', async () => {
        // Legacy project (pre-dating the auto-apply-on-create fix):
        // server returns empty playbooks + recipe_required=true instead
        // of dumping the full cross-task-shape catalog. The panel must
        // surface the shared directive CTA pointing at the recipe
        // picker. Three panels (synth playbooks, auto-RAG comparison,
        // archetype comparison) share the same NoRecipeEmptyState
        // component so the legacy user sees identical wording + button
        // regardless of which tab they land on.
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 7,
                recipe_id: null,
                recipe_required: true,
                playbooks: [],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={7} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('playbook-picker-empty-recipe-required'),
            ).toBeInTheDocument();
        });
        const cta = screen.getByTestId('playbook-picker-empty-recipe-required');
        expect(cta.textContent).toMatch(/Pick a recipe first/);
        // CTA button points at the standalone task-shape recipe-picker
        // page (NOT /recipes — that's the pipeline-DAG recipes page,
        // a different concept). Always includes a return_to so the
        // user lands back on the same panel after applying.
        const button = cta.querySelector('a') as HTMLAnchorElement;
        const href = button.getAttribute('href') || '';
        expect(href.startsWith('/project/7/recipe-picker?')).toBe(true);
        expect(href).toMatch(/return_to=/);
    });

    it('runs a playbook via the async-job endpoint and shows a queued confirmation', async () => {
        // Hardening Phase H1 — runs now go through the background-
        // job framework instead of blocking on the LLM call. The
        // panel renders a tiny "job #N queued" confirmation; actual
        // rows surface in the notification bell when the job
        // completes.
        //
        // P1/P2 — every Generate click now starts with a 1-row
        // dry-run via POST /run-playbook/dry-run. If the dry-run
        // returns ok=true the panel kicks the real async job. Test
        // mocks both endpoints via URL dispatch.
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 5,
                recipe_id: 'qa-sft',
                playbooks: [{ recipe_id: 'qa-sft', mode: 'positives_paraphrase' }],
            },
        });
        apiMock.post.mockImplementation((url: string) => {
            if (url.endsWith('/run-playbook/dry-run')) {
                return Promise.resolve({
                    data: {
                        ok: true,
                        accepted_count: 1,
                        refusal_detected: false,
                        raw_llm_snippet: '{"q":"x","a":"y"}',
                        backend_used: 'ollama:qwen2.5:14b',
                        elapsed_sec: 2.1,
                        prompt_snippet: 'p',
                        rows: [{ payload: { q: 'x', a: 'y' }, synth_confidence: 1, synth_source: 's' }],
                    },
                });
            }
            return Promise.resolve({
                data: {
                    id: 11,
                    kind: 'synth_playbook',
                    title: 'Synth · positives_paraphrase · 30 rows',
                    status: 'queued',
                    progress: null,
                    progress_message: null,
                    project_id: 5,
                    user_id: null,
                    params: { mode: 'positives_paraphrase', target_count: 30 },
                    result: null,
                    error: null,
                    queued_at: '2026-05-26T12:00:00Z',
                    started_at: null,
                    completed_at: null,
                    dismissed_at: null,
                },
            });
        });

        renderPanel(<PlaybookPickerPanel projectId={5} />);

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-run'));

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-result')).toBeInTheDocument();
        });
        // Confirmation preview shows the queued job marker.
        expect(screen.getByText(/job #11 queued/)).toBeInTheDocument();
        // Pre-flight POST fired first…
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/run-playbook/dry-run',
            expect.objectContaining({
                mode: 'positives_paraphrase',
                target_count: 1,
            }),
        );
        // …then the real async-job POST landed.
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/5/synthetic/run-playbook?async_job=true',
            {
                mode: 'positives_paraphrase',
                target_count: 30,
                target_class: null,
                backend: null,
            },
        );
    });

    it('surfaces a no-backend message when the run hits a 503', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'qa-sft',
                playbooks: [{ recipe_id: 'qa-sft', mode: 'positives_paraphrase' }],
            },
        });
        apiMock.post.mockRejectedValue({
            response: { status: 503, data: { detail: 'No synth backend available.' } },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('playbook-picker-run-error').textContent).toMatch(/No synth backend/i);
    });

    it('clamps the target-count input to the [1, 500] range', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);

        const countInput = await screen.findByTestId('playbook-picker-count') as HTMLInputElement;
        // Default is 30.
        expect(countInput.value).toBe('30');

        // Set to 9999 — should clamp to 500.
        fireEvent.change(countInput, { target: { value: '9999' } });
        expect(countInput.value).toBe('500');

        // Set to 0 — should clamp to 1.
        fireEvent.change(countInput, { target: { value: '0' } });
        expect(countInput.value).toBe('1');
    });

    // ── Backend picker (Epic 5 Phase 5a) ─────────────────────────────

    /** Route-aware GET stub: distinguishes /playbooks from /backends so
     * the panel can fetch both on mount without one overwriting the
     * other. ``backendsResponse`` defaults to a single-backend payload
     * so most tests get the picker-hidden behavior. */
    function installRouter(opts: {
        playbooks: unknown;
        backendsResponse?: unknown;
        ollamaModelsResponse?: unknown;
        cloudModelsResponse?: unknown;
    }) {
        const backendsResponse = opts.backendsResponse ?? {
            project_id: 1,
            backends: [
                { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b' },
            ],
        };
        const ollamaModelsResponse = opts.ollamaModelsResponse ?? {
            project_id: 1,
            models: [],
            default: null,
            ollama_available: false,
        };
        const cloudModelsResponse = opts.cloudModelsResponse ?? {
            project_id: 1,
            providers: [],
        };
        apiMock.get.mockImplementation(async (url: string) => {
            // Order matters — /backends/ollama/models + /cloud/models
            // must match before the generic /backends prefix.
            if (url.includes('/synthetic/backends/ollama/models')) {
                return { data: ollamaModelsResponse };
            }
            if (url.includes('/synthetic/backends/cloud/models')) {
                return { data: cloudModelsResponse };
            }
            if (url.includes('/synthetic/backends')) {
                return { data: backendsResponse };
            }
            return { data: opts.playbooks };
        });
    }

    it('hides the backend picker when only one backend is available', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        // Picker UI not in the DOM.
        expect(screen.queryByTestId('playbook-picker-backend')).not.toBeInTheDocument();
    });

    it('shows the backend picker when 2+ backends are available', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b' },
                    { name: 'nemo', available: true, describe: 'nemo:meta/llama-3.1-70b-instruct' },
                ],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });
        const select = screen.getByTestId('playbook-picker-backend') as HTMLSelectElement;
        // "Auto (recommended)" + one option per available backend.
        expect(select.options.length).toBe(3);
        expect(select.options[0].text).toMatch(/Auto/);
        expect(select.options[1].value).toBe('ollama:llama3.1:8b');
        expect(select.options[2].value).toBe('nemo:meta/llama-3.1-70b-instruct');
    });

    // ── Schema-aware badge (Epic 5 Phase 5c) ─────────────────────────

    it('renders the schema-aware badge when the active backend honors response_schema', async () => {
        // Two backends; vLLM is schema_aware. Default selection is
        // "Auto" → first available backend → ollama (NOT schema-aware),
        // so the badge should NOT render at first.
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b', schema_aware: false },
                    { name: 'vllm', available: true, describe: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct', schema_aware: true },
                ],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });

        // Auto picks ollama → no badge.
        expect(screen.queryByTestId('playbook-picker-schema-badge')).not.toBeInTheDocument();

        // Switch to vLLM → badge appears.
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-backend'),
            'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
        );
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-schema-badge')).toBeInTheDocument();
        });
        // Switch back to Ollama → badge disappears.
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-backend'),
            'ollama:llama3.1:8b',
        );
        await waitFor(() => {
            expect(screen.queryByTestId('playbook-picker-schema-badge')).not.toBeInTheDocument();
        });
    });

    it('suffixes schema-aware options in the dropdown text', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b', schema_aware: false },
                    { name: 'nemo', available: true, describe: 'nemo:meta/llama-3.1-70b-instruct', schema_aware: true },
                    { name: 'vllm', available: true, describe: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct', schema_aware: true },
                ],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });
        const select = screen.getByTestId('playbook-picker-backend') as HTMLSelectElement;
        // Auto + 3 backends.
        expect(select.options.length).toBe(4);
        expect(select.options[1].text).toBe('ollama:llama3.1:8b');           // no suffix
        expect(select.options[2].text).toMatch(/nemo:.*· schema-aware$/);
        expect(select.options[3].text).toMatch(/vllm:.*· schema-aware$/);
        // The "what does schema-aware mean?" hint appears when at least
        // one schema-aware option is in the picker.
        expect(screen.getByTestId('playbook-picker-schema-hint')).toBeInTheDocument();
    });

    it('omits the schema hint and badge when no available backend is schema-aware', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 1,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b', schema_aware: false },
                    { name: 'teacher', available: true, describe: 'teacher:llama3', schema_aware: false },
                ],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('playbook-picker-schema-hint')).not.toBeInTheDocument();
        expect(screen.queryByTestId('playbook-picker-schema-badge')).not.toBeInTheDocument();
    });

    it('passes the selected backend pin through to runPlaybook', async () => {
        installRouter({
            playbooks: {
                project_id: 3,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            backendsResponse: {
                project_id: 3,
                backends: [
                    { name: 'ollama', available: true, describe: 'ollama:llama3.1:8b' },
                    { name: 'nemo', available: true, describe: 'nemo:meta/llama-3.1-70b-instruct' },
                ],
            },
        });
        // Hardening Phase H1 — async-job POST returns a Job stub
        // (202). The pinned backend still flows through verbatim in
        // the request body. P2 added a pre-flight dry-run; it must
        // also carry the pinned backend through.
        apiMock.post.mockImplementation((url: string) => {
            if (url.endsWith('/run-playbook/dry-run')) {
                return Promise.resolve({
                    data: {
                        ok: true,
                        accepted_count: 1,
                        refusal_detected: false,
                        raw_llm_snippet: '{"x":1}',
                        backend_used: 'nemo:meta/llama-3.1-70b-instruct',
                        elapsed_sec: 1.0,
                        prompt_snippet: 'p',
                        rows: [{ payload: { x: 1 }, synth_confidence: 1, synth_source: 's' }],
                    },
                });
            }
            return Promise.resolve({
                data: {
                    id: 22,
                    kind: 'synth_playbook',
                    title: 'Synth · positives_paraphrase · 30 rows',
                    status: 'queued',
                    progress: null,
                    progress_message: null,
                    project_id: 3,
                    user_id: null,
                    params: {},
                    result: null,
                    error: null,
                    queued_at: '2026-05-26T12:00:00Z',
                    started_at: null,
                    completed_at: null,
                    dismissed_at: null,
                },
            });
        });

        renderPanel(<PlaybookPickerPanel projectId={3} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-backend')).toBeInTheDocument();
        });

        // Select the NeMo backend.
        const select = screen.getByTestId('playbook-picker-backend') as HTMLSelectElement;
        await userEvent.selectOptions(select, 'nemo:meta/llama-3.1-70b-instruct');
        expect(select.value).toBe('nemo:meta/llama-3.1-70b-instruct');

        // Click generate.
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/3/synthetic/run-playbook?async_job=true',
                expect.objectContaining({
                    mode: 'positives_paraphrase',
                    backend: 'nemo:meta/llama-3.1-70b-instruct',
                }),
            );
        });
    });

    // ── Trainability-forecast prefill (T4) ────────────────────────────
    // The TrainabilityForecastPanel emits suggested_action clicks via
    // routeForecastAction, which lands here with
    // ?prefill_mode=<SynthMode>&prefill_count=<N>. The picker honors
    // the prefill ONLY when the requested mode is in the recipe's
    // catalog — silently falls back to the catalog default otherwise.

    it('applies prefill_mode + prefill_count when the requested mode is in the catalog', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [
                    { recipe_id: 'classification', mode: 'positives_paraphrase' },
                    { recipe_id: 'classification', mode: 'class_balance_fill' },
                ],
            },
        });

        renderPanel(<PlaybookPickerPanel projectId={1} />, {
            search: '?prefill_mode=class_balance_fill&prefill_count=80',
        });

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-prefill-banner')).toBeInTheDocument();
        });

        // The class_balance_fill radio is selected (not the catalog default).
        const balanceRadio = screen
            .getByTestId('playbook-picker-mode-class_balance_fill')
            .querySelector('input[type="radio"]') as HTMLInputElement;
        expect(balanceRadio.checked).toBe(true);

        // Target count carries through.
        expect((screen.getByTestId('playbook-picker-count') as HTMLInputElement).value).toBe('80');

        // Banner carries the prefilled mode label + count.
        const banner = screen.getByTestId('playbook-picker-prefill-banner');
        expect(banner.textContent).toMatch(/Balance class distribution/);
        expect(banner.textContent).toMatch(/80/);
    });

    it('warns when the requested mode is not in the recipe catalog and falls back', async () => {
        // qa-sft doesn't ship class_balance_fill — the picker should
        // fall back to the catalog's default mode and surface the
        // fallback note in the banner so the user knows what happened.
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 2,
                recipe_id: 'qa-sft',
                playbooks: [
                    { recipe_id: 'qa-sft', mode: 'positives_paraphrase' },
                ],
            },
        });

        renderPanel(<PlaybookPickerPanel projectId={2} />, {
            search: '?prefill_mode=class_balance_fill&prefill_count=40',
        });

        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-prefill-banner')).toBeInTheDocument();
        });

        // Catalog default (the only one available) is selected.
        const paraphraseRadio = screen
            .getByTestId('playbook-picker-mode-positives_paraphrase')
            .querySelector('input[type="radio"]') as HTMLInputElement;
        expect(paraphraseRadio.checked).toBe(true);

        // Count still applies independently of the mode fallback.
        expect((screen.getByTestId('playbook-picker-count') as HTMLInputElement).value).toBe('40');

        // Banner explains the fallback so the user isn't confused.
        const banner = screen.getByTestId('playbook-picker-prefill-banner');
        expect(banner.textContent).toMatch(/isn't in this recipe/);
    });

    it('renders no prefill banner when the URL carries no prefill params', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('playbook-picker-prefill-banner')).not.toBeInTheDocument();
    });

    it('dismisses the prefill banner when the dismiss button is clicked', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />, {
            search: '?prefill_mode=positives_paraphrase&prefill_count=25',
        });
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-prefill-banner')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-prefill-dismiss'));
        expect(screen.queryByTestId('playbook-picker-prefill-banner')).not.toBeInTheDocument();
    });

    it('ignores an unknown prefill_mode value (forward-compat with future backend hints)', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
        });
        // ``some_future_mode`` isn't in VALID_SYNTH_MODES — the panel
        // must fall through to the catalog default rather than crash
        // or pre-select a non-existent radio.
        renderPanel(<PlaybookPickerPanel projectId={1} />, {
            search: '?prefill_mode=some_future_mode&prefill_count=10',
        });
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker')).toBeInTheDocument();
        });
        const radio = screen
            .getByTestId('playbook-picker-mode-positives_paraphrase')
            .querySelector('input[type="radio"]') as HTMLInputElement;
        expect(radio.checked).toBe(true);
        // Count still applied (independent of the rejected mode).
        expect((screen.getByTestId('playbook-picker-count') as HTMLInputElement).value).toBe('10');
    });

    // ── P1/P2/P3 — pre-flight + model picker + inline diagnostic ────

    /** Boilerplate: route both GETs (playbooks + ollama models) + the
     *  dry-run POST. The async-job POST stays as the secondary
     *  resolver — only fires when dry-run reports ok=true. */
    function installFullRouter(opts: {
        playbooks: unknown;
        ollamaModelsResponse?: unknown;
        dryRunResponse: unknown;
        asyncJobResponse?: unknown;
    }) {
        installRouter({
            playbooks: opts.playbooks,
            ollamaModelsResponse: opts.ollamaModelsResponse,
        });
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.endsWith('/run-playbook/dry-run')) {
                return { data: opts.dryRunResponse };
            }
            return { data: opts.asyncJobResponse ?? { id: 1, status: 'queued', kind: 'synth_playbook' } };
        });
    }

    it('renders the Ollama model picker when models are installed and pins the choice through to dry-run', async () => {
        installFullRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'class_balance_fill' }],
            },
            ollamaModelsResponse: {
                project_id: 1,
                ollama_available: true,
                default: 'qwen2.5:14b-instruct-q4_K_M',
                models: [
                    { name: 'qwen2.5:14b-instruct-q4_K_M', size_bytes: 1, parameter_size: '14.8B', family: 'qwen2' },
                    { name: 'llama3:latest', size_bytes: 1, parameter_size: '8.0B', family: 'llama' },
                ],
            },
            dryRunResponse: {
                ok: true,
                accepted_count: 1,
                refusal_detected: false,
                raw_llm_snippet: '{"text":"x","label":"y"}',
                backend_used: 'ollama:llama3:latest',
                elapsed_sec: 2.0,
                prompt_snippet: '',
                rows: [{ payload: { text: 'x', label: 'y' }, synth_confidence: 1, synth_source: 's' }],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-ollama-model')).toBeInTheDocument();
        });
        // Auto option labels the resolved auto-pick so the user sees
        // what 'Auto' actually means before they pick anything.
        const select = screen.getByTestId('playbook-picker-ollama-model') as HTMLSelectElement;
        expect(select.options[0].textContent).toMatch(/Auto.*qwen2\.5:14b/);
        // Pick the Llama 3 option explicitly.
        await userEvent.selectOptions(select, 'llama3:latest');
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/run-playbook/dry-run',
                expect.objectContaining({ backend: 'ollama:llama3:latest' }),
            );
        });
    });

    it('renders the refusal banner + Retry-with-Qwen button when dry-run detects a refusal', async () => {
        installFullRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'class_balance_fill' }],
            },
            ollamaModelsResponse: {
                project_id: 1,
                ollama_available: true,
                default: 'llama3:latest',
                models: [
                    { name: 'qwen2.5:14b-instruct-q4_K_M', size_bytes: 1, parameter_size: '14.8B', family: 'qwen2' },
                    { name: 'qwen2.5:7b-instruct-q4_K_M', size_bytes: 1, parameter_size: '7.6B', family: 'qwen2' },
                    { name: 'llama3:latest', size_bytes: 1, parameter_size: '8.0B', family: 'llama' },
                ],
            },
            dryRunResponse: {
                ok: false,
                accepted_count: 0,
                refusal_detected: true,
                raw_llm_snippet: 'I cannot generate malicious or harmful examples.',
                backend_used: 'ollama:llama3:latest',
                elapsed_sec: 0.6,
                prompt_snippet: '',
                rows: [],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-preflight-failure')).toBeInTheDocument();
        });
        // The model's raw response is shown so the user can see what
        // happened, not just a generic "0 rows" error.
        expect(
            screen.getByTestId('playbook-picker-preflight-snippet').textContent,
        ).toMatch(/cannot generate/);
        // Retry button picks the largest qwen2.5 (14B beats 7B).
        const retry = screen.getByTestId('playbook-picker-retry-qwen') as HTMLButtonElement;
        expect(retry.textContent).toMatch(/qwen2\.5:14b/);
        // The actual async-job endpoint never fired — only the dry-run.
        expect(apiMock.post).toHaveBeenCalledTimes(1);
        expect(apiMock.post.mock.calls[0][0]).toMatch(/\/run-playbook\/dry-run$/);
    });

    it('Retry-with-Qwen re-runs dry-run with the new model and submits the real job on success', async () => {
        // Two dry-run rounds: first refuses (llama3), second passes
        // (qwen2.5:14b). After the second passes, the async job fires.
        const dryRunCalls: any[] = [];
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'class_balance_fill' }],
            },
            ollamaModelsResponse: {
                project_id: 1,
                ollama_available: true,
                default: 'llama3:latest',
                models: [
                    { name: 'qwen2.5:14b-instruct-q4_K_M', size_bytes: 1, parameter_size: '14.8B', family: 'qwen2' },
                    { name: 'llama3:latest', size_bytes: 1, parameter_size: '8.0B', family: 'llama' },
                ],
            },
        });
        apiMock.post.mockImplementation(async (url: string, body: any) => {
            if (url.endsWith('/run-playbook/dry-run')) {
                dryRunCalls.push(body);
                if (body.backend && body.backend.includes('qwen2.5')) {
                    return {
                        data: {
                            ok: true,
                            accepted_count: 1,
                            refusal_detected: false,
                            raw_llm_snippet: '{"text":"x","label":"y"}',
                            backend_used: 'ollama:qwen2.5:14b-instruct-q4_K_M',
                            elapsed_sec: 3.2,
                            prompt_snippet: '',
                            rows: [{ payload: { text: 'x', label: 'y' }, synth_confidence: 1, synth_source: 's' }],
                        },
                    };
                }
                return {
                    data: {
                        ok: false,
                        accepted_count: 0,
                        refusal_detected: true,
                        raw_llm_snippet: 'I cannot generate malicious examples.',
                        backend_used: 'ollama:llama3:latest',
                        elapsed_sec: 0.6,
                        prompt_snippet: '',
                        rows: [],
                    },
                };
            }
            return { data: { id: 42, status: 'queued', kind: 'synth_playbook' } };
        });

        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        // First click — llama3 refuses.
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-retry-qwen')).toBeInTheDocument();
        });
        // Click the retry button.
        await userEvent.click(screen.getByTestId('playbook-picker-retry-qwen'));
        // After retry: dry-run passes + async job fires.
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-result')).toBeInTheDocument();
        });
        // Two dry-runs total: first with llama3, second with qwen.
        expect(dryRunCalls.length).toBe(2);
        expect(dryRunCalls[0].backend).toBeNull();  // auto-pick = llama3 server-side
        expect(dryRunCalls[1].backend).toMatch(/qwen2\.5/);
        // Async-job call also fired with the qwen backend.
        const asyncCalls = apiMock.post.mock.calls.filter(
            (c) => String(c[0]).endsWith('?async_job=true'),
        );
        expect(asyncCalls.length).toBe(1);
        expect(asyncCalls[0][1].backend).toMatch(/qwen2\.5/);
    });

    it('renders the cloud provider picker with key-saved status + curated models per provider', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            cloudModelsResponse: {
                project_id: 1,
                providers: [
                    {
                        provider: 'openai',
                        key_saved: true,
                        models: [
                            { id: 'gpt-4o-mini', label: 'GPT-4o mini' },
                            { id: 'gpt-4o', label: 'GPT-4o' },
                        ],
                    },
                    { provider: 'anthropic', key_saved: false, models: [{ id: 'claude-haiku-4-5-20251001', label: 'Haiku' }] },
                    { provider: 'deepseek', key_saved: false, models: [{ id: 'deepseek-chat', label: 'V3 chat' }] },
                ],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-cloud-provider')).toBeInTheDocument();
        });
        const providerSelect = screen.getByTestId('playbook-picker-cloud-provider') as HTMLSelectElement;
        // All 3 providers appear with their key-saved badges.
        const text = providerSelect.textContent ?? '';
        expect(text).toMatch(/openai/);
        expect(text).toMatch(/key saved/);
        expect(text).toMatch(/no key/);
        // Picking the no-key provider surfaces the inline 'save key first' hint.
        await userEvent.selectOptions(providerSelect, 'anthropic');
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-cloud-no-key')).toBeInTheDocument();
        });
        // Picking the key-saved provider enables the model dropdown.
        await userEvent.selectOptions(providerSelect, 'openai');
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-cloud-model')).toBeInTheDocument();
        });
        const modelSelect = screen.getByTestId('playbook-picker-cloud-model') as HTMLSelectElement;
        expect(modelSelect.disabled).toBe(false);
        expect(modelSelect.textContent).toMatch(/GPT-4o mini/);
    });

    it('pins the chosen cloud provider + model through to dry-run as cloud:<provider>:<model>', async () => {
        const dryRunCalls: any[] = [];
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            cloudModelsResponse: {
                project_id: 1,
                providers: [
                    { provider: 'openai', key_saved: true, models: [{ id: 'gpt-4o-mini', label: 'GPT-4o mini' }] },
                    { provider: 'anthropic', key_saved: false, models: [] },
                    { provider: 'deepseek', key_saved: false, models: [] },
                ],
            },
        });
        apiMock.post.mockImplementation(async (url: string, body: any) => {
            if (url.endsWith('/run-playbook/dry-run')) {
                dryRunCalls.push(body);
                return {
                    data: {
                        ok: true, accepted_count: 1, refusal_detected: false,
                        raw_llm_snippet: '{"x":1}', backend_used: 'cloud:openai:gpt-4o-mini',
                        elapsed_sec: 1.2, prompt_snippet: '',
                        rows: [{ payload: { x: 1 }, synth_confidence: 1, synth_source: 's' }],
                    },
                };
            }
            return { data: { id: 99, status: 'queued', kind: 'synth_playbook' } };
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-cloud-provider')).toBeInTheDocument();
        });
        await userEvent.selectOptions(screen.getByTestId('playbook-picker-cloud-provider'), 'openai');
        await userEvent.selectOptions(screen.getByTestId('playbook-picker-cloud-model'), 'gpt-4o-mini');
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(dryRunCalls.length).toBeGreaterThanOrEqual(1);
        });
        expect(dryRunCalls[0].backend).toBe('cloud:openai:gpt-4o-mini');
        // Async-job call also fired with the cloud backend.
        const asyncCalls = apiMock.post.mock.calls.filter(
            (c) => String(c[0]).endsWith('?async_job=true'),
        );
        expect(asyncCalls.length).toBe(1);
        expect(asyncCalls[0][1].backend).toBe('cloud:openai:gpt-4o-mini');
    });

    it('shows an active-backend indicator that resolves cloud > ollama > auto', async () => {
        installRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'positives_paraphrase' }],
            },
            ollamaModelsResponse: {
                project_id: 1,
                ollama_available: true,
                default: 'qwen2.5:14b-instruct-q4_K_M',
                models: [
                    { name: 'qwen2.5:14b-instruct-q4_K_M', size_bytes: 1, parameter_size: '14.8B', family: 'qwen2' },
                    { name: 'llama3:latest', size_bytes: 1, parameter_size: '8.0B', family: 'llama' },
                ],
            },
            cloudModelsResponse: {
                project_id: 1,
                providers: [
                    { provider: 'openai', key_saved: true, models: [{ id: 'gpt-4o-mini', label: 'GPT-4o mini' }] },
                    { provider: 'anthropic', key_saved: false, models: [] },
                    { provider: 'deepseek', key_saved: false, models: [] },
                ],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-active-backend')).toBeInTheDocument();
        });
        // Initial: no pick → indicator shows the auto-pick.
        let indicator = screen.getByTestId('playbook-picker-active-backend');
        expect(indicator.textContent).toMatch(/auto.*qwen2\.5:14b/);

        // Pick Ollama llama3 → indicator updates to ollama:llama3:latest.
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-ollama-model'),
            'llama3:latest',
        );
        await waitFor(() => {
            indicator = screen.getByTestId('playbook-picker-active-backend');
            expect(indicator.textContent).toMatch(/ollama:llama3:latest/);
        });

        // Pick OpenAI → ollama pin should clear AND indicator shows cloud.
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-cloud-provider'),
            'openai',
        );
        await userEvent.selectOptions(
            screen.getByTestId('playbook-picker-cloud-model'),
            'gpt-4o-mini',
        );
        await waitFor(() => {
            indicator = screen.getByTestId('playbook-picker-active-backend');
            expect(indicator.textContent).toMatch(/cloud:openai:gpt-4o-mini/);
        });
        // The Ollama dropdown reset to Auto when cloud was picked.
        const ollamaSelect = screen.getByTestId('playbook-picker-ollama-model') as HTMLSelectElement;
        expect(ollamaSelect.value).toBe('');

        // Now pick Ollama again — that should clear the cloud pick
        // (bidirectional mutual exclusion).
        await userEvent.selectOptions(ollamaSelect, 'llama3:latest');
        await waitFor(() => {
            indicator = screen.getByTestId('playbook-picker-active-backend');
            expect(indicator.textContent).toMatch(/ollama:llama3:latest/);
        });
        const cloudSelect = screen.getByTestId('playbook-picker-cloud-provider') as HTMLSelectElement;
        expect(cloudSelect.value).toBe('');
    });

    it('shows a 0-rows diagnostic (no retry) when dry-run returns empty output without a refusal', async () => {
        installFullRouter({
            playbooks: {
                project_id: 1,
                recipe_id: 'classification',
                playbooks: [{ recipe_id: 'classification', mode: 'class_balance_fill' }],
            },
            dryRunResponse: {
                ok: false,
                accepted_count: 0,
                refusal_detected: false,
                raw_llm_snippet: '```json\n[\n```',  // malformed but not a refusal
                backend_used: 'ollama:qwen2.5:14b',
                elapsed_sec: 4.2,
                prompt_snippet: '',
                rows: [],
            },
        });
        renderPanel(<PlaybookPickerPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-run')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('playbook-picker-run'));
        await waitFor(() => {
            expect(screen.getByTestId('playbook-picker-preflight-failure')).toBeInTheDocument();
        });
        // No retry-with-Qwen for non-refusal failures — the issue
        // isn't the model, it's the prompt/parser.
        expect(screen.queryByTestId('playbook-picker-retry-qwen')).not.toBeInTheDocument();
        expect(screen.getByTestId('playbook-picker-preflight-snippet').textContent).toMatch(/json/);
    });
});
