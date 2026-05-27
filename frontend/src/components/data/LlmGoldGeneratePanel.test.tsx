import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import type { ReactElement } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

import LlmGoldGeneratePanel from './LlmGoldGeneratePanel';

/** Render the panel inside a MemoryRouter so the cluster-fix
 *  prefill reader (``useLocation``) can mount. Pass ``search`` to
 *  simulate a deep-link from FailureClustersPanel's "Fix in gold set"
 *  button. */
function renderPanel(element: ReactElement, opts: { search?: string } = {}) {
    const path = `/route${opts.search ?? ''}`;
    return render(
        <MemoryRouter initialEntries={[path]}>
            {element}
        </MemoryRouter>,
    );
}


function makeGenerateResponse(overrides: Record<string, unknown> = {}) {
    return {
        rows: [
            { question: 'Q1?', answer: 'A1.', rationale: 'because', source_excerpt: '' },
            { question: 'Q2?', answer: 'A2.', rationale: '', source_excerpt: '' },
            { question: 'Q3?', answer: 'A3.', rationale: '', source_excerpt: '' },
        ],
        recipe_id: 'qa-sft',
        provider: 'openai',
        model: 'gpt-4o-mini',
        usage: { prompt_tokens: 120, completion_tokens: 350 },
        prompt_preview: 'PROJECT: …',
        reference_chunk_count: 0,
        estimated_cost_usd: 0.00023,
        ...overrides,
    };
}


function makeCostEstimateResponse(overrides: Record<string, unknown> = {}) {
    return {
        provider: 'openai',
        model: 'gpt-4o-mini',
        count: 10,
        ground_in_source_requested: true,
        ground_in_source_effective: true,
        estimated_cost_usd: 0.0009,
        estimated_prompt_tokens: 2900,
        estimated_completion_tokens: 1600,
        reference_chunk_count: 6,
        ...overrides,
    };
}


/**
 * Route-aware POST mock — the panel fires:
 *   * /cost-estimate on mount + on any cost-relevant input change
 *   * /generate-via-llm on the Generate button click
 *   * /gold/import on Save
 * Tests that don't override return sensible defaults so the cost
 * badge renders cleanly without interfering with assertions on
 * the other two endpoints.
 */
function installPostRouter(opts: {
    generate?: unknown;
    generateError?: unknown;
    import?: unknown;
    importError?: unknown;
    estimate?: unknown;
    previewPrompt?: unknown;
    previewPromptError?: unknown;
} = {}) {
    apiMock.post.mockImplementation(async (url: string) => {
        if (url.includes('/gold/generate-via-llm/cost-estimate')) {
            return { data: opts.estimate ?? makeCostEstimateResponse() };
        }
        if (url.includes('/gold/generate-via-llm/preview-prompt')) {
            if (opts.previewPromptError) throw opts.previewPromptError;
            return {
                data: opts.previewPrompt ?? {
                    recipe_id: 'qa-sft',
                    system_prompt: 'SYSTEM: be helpful.',
                    user_prompt: 'USER: generate 3 QA pairs about refunds.',
                    reference_chunk_count: 0,
                    known_labels: [],
                },
            };
        }
        if (url.includes('/gold/generate-via-llm')) {
            if (opts.generateError) throw opts.generateError;
            return { data: opts.generate ?? makeGenerateResponse() };
        }
        if (url.includes('/gold/import')) {
            if (opts.importError) throw opts.importError;
            return { data: opts.import ?? { imported: 0 } };
        }
        return { data: {} };
    });
}


/**
 * Default GET responder for /generate-via-llm/saved-key — the panel
 * fires this on mount + on provider change. Tests that don't care
 * about the stored-key UX get back "no stored key" so the existing
 * input + Save-this-key checkbox render. Tests that DO care override
 * by re-mocking apiMock.get directly.
 */
function installGetRouter(opts: { savedKey?: unknown } = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/gold/generate-via-llm/saved-key')) {
            return {
                data: opts.savedKey ?? {
                    has_stored_key: false,
                    value_hint: null,
                },
            };
        }
        return { data: {} };
    });
}


describe('LlmGoldGeneratePanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        apiMock.delete.mockReset();
        toastMock.success.mockReset();
        toastMock.warning.mockReset();
        toastMock.error.mockReset();
        // Sensible default — no stored key for any provider. Tests
        // exercising the stored-key UX override this with their own
        // get-mock.
        installGetRouter();
    });

    it('Anthropic model dropdown swaps when provider switches', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        // OpenAI default — first option is gpt-4o-mini.
        const model = screen.getByTestId('llm-gold-model') as HTMLSelectElement;
        expect(model.value).toBe('gpt-4o-mini');

        const provider = screen.getByTestId('llm-gold-provider') as HTMLSelectElement;
        await userEvent.selectOptions(provider, 'anthropic');
        await waitFor(() => {
            expect((screen.getByTestId('llm-gold-model') as HTMLSelectElement).value)
                .toBe('claude-haiku-4-5-20251001');
        });
    });

    it('POSTs the right payload + renders preview rows on happy path', async () => {
        installPostRouter();
        const onRowsSaved = vi.fn();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                onRowsSaved={onRowsSaved}
            />,
        );

        const apiKey = screen.getByTestId('llm-gold-api-key') as HTMLInputElement;
        fireEvent.change(apiKey, { target: { value: 'sk-test' } });
        const count = screen.getByTestId('llm-gold-count') as HTMLInputElement;
        fireEvent.change(count, { target: { value: '3' } });
        const focus = screen.getByTestId('llm-gold-focus-hint') as HTMLTextAreaElement;
        fireEvent.change(focus, { target: { value: 'edge cases around refunds' } });

        await userEvent.click(screen.getByTestId('llm-gold-generate'));

        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        // All 3 rows rendered + selected by default.
        for (const i of [0, 1, 2]) {
            const row = screen.getByTestId(`llm-gold-preview-row-${i}`);
            expect(row.textContent).toMatch(/Q\d\?/);
            const cb = screen.getByTestId(
                `llm-gold-preview-row-${i}-toggle`,
            ) as HTMLInputElement;
            expect(cb.checked).toBe(true);
        }

        // POST body shape — grounding default ON, focus hint flows through.
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/42/gold/generate-via-llm',
            expect.objectContaining({
                provider: 'openai',
                model: 'gpt-4o-mini',
                count: 3,
                api_key: 'sk-test',
                focus_hint: 'edge cases around refunds',
                ground_in_source: true,
            }),
            expect.objectContaining({ timeout: 420_000 }),
        );

        // Save button reflects the all-selected default.
        expect(screen.getByTestId('llm-gold-save').textContent).toMatch(
            /Save 3 of 3/,
        );
        // Meta line surfaces provider + model + token total + cost spent.
        const metaText = screen.getByTestId('llm-gold-preview-meta').textContent || '';
        expect(metaText).toMatch(/openai · gpt-4o-mini · 470 tokens/);
        expect(metaText).toMatch(/\$0\.0002 spent/);
    });

    it('Save selected calls /gold/import with only the checked rows', async () => {
        installPostRouter({ import: { imported: 2 } });
        const onRowsSaved = vi.fn();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_test"
                onRowsSaved={onRowsSaved}
            />,
        );
        // Generate first.
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });

        // Deselect row 1 (Q2).
        await userEvent.click(screen.getByTestId('llm-gold-preview-row-1-toggle'));
        expect(screen.getByTestId('llm-gold-save').textContent).toMatch(
            /Save 2 of 3/,
        );

        await userEvent.click(screen.getByTestId('llm-gold-save'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenLastCalledWith(
                '/projects/42/gold/import',
                expect.objectContaining({
                    dataset_type: 'gold_test',
                    pairs: [
                        { question: 'Q1?', answer: 'A1.' },
                        { question: 'Q3?', answer: 'A3.' },
                    ],
                }),
            );
        });
        expect(toastMock.success).toHaveBeenCalledWith(
            expect.stringContaining('Saved 2 rows to Test set'),
            4000,
        );
        expect(onRowsSaved).toHaveBeenCalledTimes(1);
        // Preview wiped after save.
        await waitFor(() => {
            expect(screen.queryByTestId('llm-gold-preview')).not.toBeInTheDocument();
        });
    });

    it('Discard wipes the preview without POSTing /gold/import', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        // Generate is the load-bearing call; cost-estimate fires on
        // mount + on form changes (don't pin a hard count there since
        // React's render schedule makes it brittle). Discard must NOT
        // POST anything new — assert by checking no /gold/import URL
        // shows up in the call list.
        await userEvent.click(screen.getByTestId('llm-gold-discard'));
        const importCalls = apiMock.post.mock.calls.filter(
            (call: unknown[]) => String(call[0] || '').includes('/gold/import'),
        );
        expect(importCalls).toHaveLength(0);
        expect(screen.queryByTestId('llm-gold-preview')).not.toBeInTheDocument();
    });

    it('renders structured error code + message on 400 RECIPE_NOT_SUPPORTED', async () => {
        installPostRouter({
            generateError: {
                response: {
                    status: 400,
                    data: {
                        detail: {
                            error_code: 'RECIPE_NOT_SUPPORTED',
                            message:
                                "LLM-assisted gold generation v1 only supports the 'qa-sft' recipe.",
                        },
                    },
                },
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('llm-gold-error').textContent).toMatch(
            /RECIPE_NOT_SUPPORTED/,
        );
        expect(screen.getByTestId('llm-gold-error').textContent).toMatch(
            /qa-sft/,
        );
        // No preview rendered.
        expect(screen.queryByTestId('llm-gold-preview')).not.toBeInTheDocument();
    });

    it('surfaces axios "Network Error" with actionable explanation (NETWORK_ERROR)', async () => {
        // axios produces ``{message: "Network Error"}`` with no
        // response when the connection drops mid-request — common
        // after burning LLM tokens on a slow reasoning model that
        // outlasted the frontend timeout. User MUST be told what
        // likely happened + where to look for the actual outcome,
        // not just "UNKNOWN".
        installPostRouter({
            generateError: { message: 'Network Error' },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-error')).toBeInTheDocument();
        });
        const text = screen.getByTestId('llm-gold-error').textContent || '';
        expect(text).toMatch(/NETWORK_ERROR/);
        // Useful diagnostics in the message body.
        expect(text).toMatch(/tokens were billed/);
        expect(text).toMatch(/uvicorn\.log/);
    });

    it('renders the upstream provider error (502) as a plain string', async () => {
        installPostRouter({
            generateError: {
                response: {
                    status: 502,
                    data: { detail: 'OpenAI returned 401: invalid API key' },
                },
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-bad' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('llm-gold-error').textContent).toMatch(
            /invalid API key/,
        );
    });

    it('clamps the count input to [1, 50]', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        const count = screen.getByTestId('llm-gold-count') as HTMLInputElement;
        fireEvent.change(count, { target: { value: '999' } });
        expect(count.value).toBe('50');
        fireEvent.change(count, { target: { value: '0' } });
        expect(count.value).toBe('1');
    });

    // ── Grounding + cost transparency (new) ─────────────────────

    it('grounding is on by default + cost badge renders with chunk count', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        const toggle = await screen.findByTestId('llm-gold-ground-toggle') as HTMLInputElement;
        expect(toggle.checked).toBe(true);
        // Cost badge resolves with the estimate from the router.
        await waitFor(() => {
            const amount = screen.getByTestId('llm-gold-cost-amount');
            expect(amount.textContent).toMatch(/\$0\.0009/);
        });
        const badge = screen.getByTestId('llm-gold-cost-estimate');
        expect(badge.textContent).toMatch(/grounded in 6 chunks/);
    });

    it('unchecking grounding sends ground_in_source=false', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await screen.findByTestId('llm-gold-ground-toggle');
        await userEvent.click(screen.getByTestId('llm-gold-ground-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            expect(generateCalls.length).toBeGreaterThanOrEqual(1);
            const lastBody = generateCalls[generateCalls.length - 1][1];
            expect(lastBody).toEqual(
                expect.objectContaining({ ground_in_source: false }),
            );
        });
    });

    it('cost badge shows "grounding off (no cleaned chunks)" when pool is empty', async () => {
        installPostRouter({
            estimate: makeCostEstimateResponse({
                ground_in_source_effective: false,
                reference_chunk_count: 0,
                estimated_cost_usd: 0.0003,
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await waitFor(() => {
            const badge = screen.getByTestId('llm-gold-cost-estimate');
            expect(badge.textContent).toMatch(/grounding off \(no cleaned chunks\)/);
        });
    });

    // ── Deepseek + custom model override (Deepseek-V4-Pro etc) ───

    it('selecting Deepseek sends provider=openai + api_url=Deepseek host on the wire', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.selectOptions(
            screen.getByTestId('llm-gold-provider'),
            'deepseek',
        );
        // Default Deepseek model is deepseek-chat.
        await waitFor(() => {
            expect(
                (screen.getByTestId('llm-gold-model') as HTMLSelectElement).value,
            ).toBe('deepseek-chat');
        });
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-deepseek-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            expect(generateCalls.length).toBeGreaterThanOrEqual(1);
            const body = generateCalls[generateCalls.length - 1][1];
            // Deepseek maps to provider=openai (their API is OpenAI-
            // compatible) + the Deepseek host via api_url.
            expect(body).toEqual(
                expect.objectContaining({
                    provider: 'openai',
                    api_url: 'https://api.deepseek.com/v1/chat/completions',
                    model: 'deepseek-chat',
                }),
            );
        });
    });

    it('custom model override beats the dropdown — e.g. DeepSeek-V4-Pro', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.selectOptions(
            screen.getByTestId('llm-gold-provider'),
            'deepseek',
        );
        // Type a model the dropdown doesn't carry.
        fireEvent.change(screen.getByTestId('llm-gold-custom-model'), {
            target: { value: 'DeepSeek-V4-Pro' },
        });
        // Dropdown becomes disabled to signal the override is in
        // charge.
        await waitFor(() => {
            expect(
                (screen.getByTestId('llm-gold-model') as HTMLSelectElement).disabled,
            ).toBe(true);
        });
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-deepseek-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            const body = generateCalls[generateCalls.length - 1][1];
            // Custom override wins; api_url still points at Deepseek.
            expect(body).toEqual(
                expect.objectContaining({
                    provider: 'openai',
                    api_url: 'https://api.deepseek.com/v1/chat/completions',
                    model: 'DeepSeek-V4-Pro',
                }),
            );
        });
    });

    it('switching provider clears any custom-model override', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-custom-model'), {
            target: { value: 'gpt-5' },
        });
        expect((screen.getByTestId('llm-gold-custom-model') as HTMLInputElement).value)
            .toBe('gpt-5');

        await userEvent.selectOptions(
            screen.getByTestId('llm-gold-provider'),
            'anthropic',
        );
        // Effect that resets the dropdown also clears the custom
        // override — the gpt-5 string would be nonsense for Anthropic.
        await waitFor(() => {
            expect(
                (screen.getByTestId('llm-gold-custom-model') as HTMLInputElement).value,
            ).toBe('');
        });
        // Anthropic dropdown re-enabled.
        expect(
            (screen.getByTestId('llm-gold-model') as HTMLSelectElement).disabled,
        ).toBe(false);
    });

    // ── Stored-key UX (saved-key endpoints) ────────────────────────

    it('renders the "Using stored key" row when GET returns has_stored_key=true', async () => {
        installPostRouter();
        // GET returns a stored hint up front.
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/gold/generate-via-llm/saved-key')) {
                return {
                    data: { has_stored_key: true, value_hint: 'sk************xyz' },
                };
            }
            return { data: {} };
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        // Stored-key row renders + masked hint visible.
        const hint = await screen.findByTestId('llm-gold-stored-key-hint');
        expect(hint.textContent).toMatch(/Using stored key/);
        expect(hint.textContent).toMatch(/sk\*+xyz/);
        // The raw input is hidden by default in the stored-key state.
        expect(screen.queryByTestId('llm-gold-api-key')).not.toBeInTheDocument();
        // "Save this key" checkbox should NOT show — a key is already
        // saved.
        expect(screen.queryByTestId('llm-gold-save-key-toggle')).not.toBeInTheDocument();
        // GET fired with provider=openai (the default).
        expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/42/gold/generate-via-llm/saved-key',
            expect.objectContaining({ params: { provider: 'openai' } }),
        );
    });

    it('"Save this key for future generations" triggers PUT before generate', async () => {
        installPostRouter();
        // No stored key initially.
        installGetRouter();
        apiMock.put.mockResolvedValue({
            data: { has_stored_key: true, value_hint: 'sk************123' },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        // Wait for stored-key fetch to settle so the input + checkbox
        // render.
        await screen.findByTestId('llm-gold-save-key-toggle');
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-new-key-123' },
        });
        // Opt in to save.
        await userEvent.click(screen.getByTestId('llm-gold-save-key-toggle'));
        await userEvent.click(screen.getByTestId('llm-gold-generate'));

        await waitFor(() => {
            // PUT fired with the typed key + current provider.
            expect(apiMock.put).toHaveBeenCalledWith(
                '/projects/42/gold/generate-via-llm/saved-key',
                expect.objectContaining({
                    provider: 'openai',
                    api_key: 'sk-new-key-123',
                }),
            );
        });
        // Generate call still fires.
        const generateCalls = apiMock.post.mock.calls.filter(
            (call: unknown[]) =>
                String(call[0] || '').endsWith('/gold/generate-via-llm'),
        );
        expect(generateCalls.length).toBeGreaterThanOrEqual(1);
    });

    it('does NOT fire PUT when the "Save this key" checkbox is unchecked', async () => {
        installPostRouter();
        installGetRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await screen.findByTestId('llm-gold-save-key-toggle');
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-one-shot' },
        });
        // Leave checkbox unchecked.
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/gold/generate-via-llm',
                expect.objectContaining({ api_key: 'sk-one-shot' }),
                expect.anything(),
            );
        });
        // No PUT to /saved-key.
        expect(apiMock.put).not.toHaveBeenCalled();
    });

    it('"Remove" clicks DELETE + refetches the stored-key state', async () => {
        installPostRouter();
        // First GET returns stored key; after DELETE the refetch
        // returns the no-key state. Use a counter to switch responses.
        let getCallCount = 0;
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/gold/generate-via-llm/saved-key')) {
                getCallCount += 1;
                if (getCallCount === 1) {
                    return {
                        data: { has_stored_key: true, value_hint: 'sk********end' },
                    };
                }
                return { data: { has_stored_key: false, value_hint: null } };
            }
            return { data: {} };
        });
        apiMock.delete.mockResolvedValue({ status: 204, data: '' });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={7}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        // Stored row shows up.
        await screen.findByTestId('llm-gold-stored-key-row');
        await userEvent.click(screen.getByTestId('llm-gold-stored-key-remove'));

        await waitFor(() => {
            expect(apiMock.delete).toHaveBeenCalledWith(
                '/projects/7/gold/generate-via-llm/saved-key',
                expect.objectContaining({ params: { provider: 'openai' } }),
            );
        });
        // After refetch, the inline input + Save checkbox reappear.
        await screen.findByTestId('llm-gold-api-key');
        expect(screen.getByTestId('llm-gold-save-key-toggle')).toBeInTheDocument();
        // Stored row gone.
        expect(screen.queryByTestId('llm-gold-stored-key-row')).not.toBeInTheDocument();
        // GET was called at least twice — initial mount + post-delete refetch.
        expect(getCallCount).toBeGreaterThanOrEqual(2);
    });

    // ── Difficulty + hallucination-trap distribution (qa-sft) ──────

    it('mix toggle is qa-sft-only (hidden for classification)', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="classification"
                onRowsSaved={() => {}}
            />,
        );
        expect(screen.queryByTestId('llm-gold-mix-toggle')).not.toBeInTheDocument();
        expect(screen.queryByTestId('llm-gold-mix-group')).not.toBeInTheDocument();
    });

    it('mix toggle visible on qa-sft; off by default; Generate sends count, not distribution', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                recipeId="qa-sft"
                onRowsSaved={() => {}}
            />,
        );
        const toggle = screen.getByTestId('llm-gold-mix-toggle') as HTMLInputElement;
        expect(toggle.checked).toBe(false);
        // Mix inputs hidden when toggle is off.
        expect(screen.queryByTestId('llm-gold-mix-inputs')).not.toBeInTheDocument();

        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));

        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            expect(generateCalls.length).toBeGreaterThanOrEqual(1);
            const body = generateCalls[generateCalls.length - 1][1];
            expect(body.distribution).toBeUndefined();
            expect(body.count).toBe(10);  // default count
        });
    });

    it('opening the mix toggle reveals 4 inputs + total; total replaces count', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="qa-sft"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-mix-toggle'));
        const inputs = await screen.findByTestId('llm-gold-mix-inputs');
        expect(inputs).toBeInTheDocument();
        // Defaults: 5/3/2/0 → total 10.
        expect(
            (screen.getByTestId('llm-gold-mix-easy') as HTMLInputElement).value,
        ).toBe('5');
        expect(
            (screen.getByTestId('llm-gold-mix-medium') as HTMLInputElement).value,
        ).toBe('3');
        expect(
            (screen.getByTestId('llm-gold-mix-hard') as HTMLInputElement).value,
        ).toBe('2');
        expect(
            (screen.getByTestId('llm-gold-mix-hallucination-traps') as HTMLInputElement)
                .value,
        ).toBe('0');
        expect(screen.getByTestId('llm-gold-mix-total').textContent).toMatch(/Total:\s*10/);
        // Count input is now read-only + reflects the mix-total.
        const countInput = screen.getByTestId('llm-gold-count') as HTMLInputElement;
        expect(countInput.disabled).toBe(true);
        expect(countInput.value).toBe('10');
    });

    it('Generate with mix on sends distribution payload + correct total count', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                recipeId="qa-sft"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-mix-toggle'));
        // Set 5 easy / 3 medium / 2 hard / 2 traps = 12 total.
        fireEvent.change(screen.getByTestId('llm-gold-mix-hallucination-traps'), {
            target: { value: '2' },
        });
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));

        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            expect(generateCalls.length).toBeGreaterThanOrEqual(1);
            const body = generateCalls[generateCalls.length - 1][1];
            expect(body.distribution).toEqual({
                easy: 5,
                medium: 3,
                hard: 2,
                hallucination_traps: 2,
            });
            // count carries the mix total.
            expect(body.count).toBe(12);
        });
    });

    it('mix total of 0 disables Generate + surfaces an error hint', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="qa-sft"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-mix-toggle'));
        // Zero everything out.
        for (const id of ['easy', 'medium', 'hard', 'hallucination-traps']) {
            fireEvent.change(screen.getByTestId(`llm-gold-mix-${id}`), {
                target: { value: '0' },
            });
        }
        // Total error visible + Generate button disabled.
        expect(screen.getByTestId('llm-gold-mix-total-error')).toBeInTheDocument();
        expect(
            (screen.getByTestId('llm-gold-generate') as HTMLButtonElement).disabled,
        ).toBe(true);
    });

    it('preview row shows difficulty + trap badges when the LLM tagged them', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'qa-sft',
                rows: [
                    {
                        question: 'Easy Q?',
                        answer: 'Easy A.',
                        rationale: '',
                        source_excerpt: '',
                        difficulty: 'easy',
                        is_hallucination_trap: false,
                    },
                    {
                        question: 'Trap Q?',
                        answer: "I don't know.",
                        rationale: '',
                        source_excerpt: '',
                        difficulty: 'hard',
                        is_hallucination_trap: true,
                    },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="qa-sft"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('llm-gold-preview-row-0-difficulty').textContent,
        ).toBe('easy');
        // Row 0 has no trap → trap badge absent.
        expect(
            screen.queryByTestId('llm-gold-preview-row-0-trap'),
        ).not.toBeInTheDocument();
        // Row 1 has trap → both badges present.
        expect(
            screen.getByTestId('llm-gold-preview-row-1-difficulty').textContent,
        ).toBe('hard');
        expect(
            screen.getByTestId('llm-gold-preview-row-1-trap').textContent,
        ).toMatch(/trap/);
    });

    it('Save selected forwards difficulty + trap fields in qa-sft pairs', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'qa-sft',
                rows: [
                    {
                        question: 'Easy Q?',
                        answer: 'Easy A.',
                        rationale: '',
                        source_excerpt: '',
                        difficulty: 'easy',
                        is_hallucination_trap: false,
                    },
                    {
                        question: 'Trap Q?',
                        answer: "I don't know.",
                        rationale: '',
                        source_excerpt: '',
                        difficulty: 'hard',
                        is_hallucination_trap: true,
                    },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={7}
                datasetType="gold_dev"
                recipeId="qa-sft"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('llm-gold-save'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/gold/import',
                expect.objectContaining({
                    dataset_type: 'gold_dev',
                    pairs: [
                        {
                            question: 'Easy Q?',
                            answer: 'Easy A.',
                            difficulty: 'easy',
                            is_hallucination_trap: false,
                        },
                        {
                            question: 'Trap Q?',
                            answer: "I don't know.",
                            difficulty: 'hard',
                            is_hallucination_trap: true,
                        },
                    ],
                }),
            );
        });
    });

    // ── Review-prompt-before-sending (advanced UX) ─────────────────

    it('Review-prompt toggle is OFF by default + Generate fires LLM directly', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        const toggle = screen.getByTestId('llm-gold-review-toggle') as HTMLInputElement;
        expect(toggle.checked).toBe(false);

        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        // No review section — went straight to generation.
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('llm-gold-prompt-review')).not.toBeInTheDocument();
        // preview-prompt was NOT called.
        const previewCalls = apiMock.post.mock.calls.filter(
            (call: unknown[]) =>
                String(call[0] || '').includes('/gold/generate-via-llm/preview-prompt'),
        );
        expect(previewCalls).toHaveLength(0);
    });

    it('Review-prompt toggle ON: Generate opens review section with prompts', async () => {
        installPostRouter({
            previewPrompt: {
                recipe_id: 'classification',
                system_prompt: 'You are a sentiment classifier.',
                user_prompt: 'Generate 3 classification rows. Labels: positive, negative.',
                reference_chunk_count: 0,
                known_labels: ['positive', 'negative'],
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                recipeId="classification"
                onRowsSaved={() => {}}
            />,
        );
        // Flip the toggle on.
        await userEvent.click(screen.getByTestId('llm-gold-review-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));

        // Review section appears with pre-populated textareas.
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-prompt-review')).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('llm-gold-prompt-review-header').textContent,
        ).toMatch(/classification/);
        const userArea = screen.getByTestId('llm-gold-prompt-review-user') as HTMLTextAreaElement;
        expect(userArea.value).toMatch(/Generate 3 classification/);
        const systemArea = screen.getByTestId('llm-gold-prompt-review-system') as HTMLTextAreaElement;
        expect(systemArea.value).toMatch(/sentiment classifier/);
        // Known-labels hint visible.
        expect(
            screen.getByTestId('llm-gold-prompt-review-labels').textContent,
        ).toMatch(/positive, negative/);
        // No /generate-via-llm POST yet — only preview-prompt.
        const genCalls = apiMock.post.mock.calls.filter(
            (call: unknown[]) => {
                const u = String(call[0] || '');
                return u.endsWith('/gold/generate-via-llm');
            },
        );
        expect(genCalls).toHaveLength(0);
    });

    it('Send from review fires /generate-via-llm with edited overrides', async () => {
        installPostRouter({
            previewPrompt: {
                recipe_id: 'qa-sft',
                system_prompt: 'Default system prompt.',
                user_prompt: 'Default user prompt.',
                reference_chunk_count: 0,
                known_labels: [],
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={7}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-review-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-prompt-review')).toBeInTheDocument();
        });

        // User edits both prompts.
        const userArea = screen.getByTestId('llm-gold-prompt-review-user');
        fireEvent.change(userArea, {
            target: { value: 'MY CUSTOM USER PROMPT — please return JSON.' },
        });
        const systemArea = screen.getByTestId('llm-gold-prompt-review-system');
        fireEvent.change(systemArea, {
            target: { value: 'MY CUSTOM SYSTEM PROMPT.' },
        });

        await userEvent.click(screen.getByTestId('llm-gold-prompt-review-send'));

        // Generate POST fires with both overrides.
        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            expect(generateCalls.length).toBeGreaterThanOrEqual(1);
            const body = generateCalls[generateCalls.length - 1][1];
            expect(body).toEqual(
                expect.objectContaining({
                    user_prompt_override: 'MY CUSTOM USER PROMPT — please return JSON.',
                    system_prompt_override: 'MY CUSTOM SYSTEM PROMPT.',
                }),
            );
        });
        // Review section gone (rows preview appears in its place).
        await waitFor(() => {
            expect(screen.queryByTestId('llm-gold-prompt-review')).not.toBeInTheDocument();
        });
        expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
    });

    it('Send from review with unchanged prompts omits both overrides', async () => {
        // When the user clicks Send without editing, the backend
        // should run its default prompt-building path — verify by
        // checking that override fields are undefined on the wire.
        installPostRouter({
            previewPrompt: {
                recipe_id: 'qa-sft',
                system_prompt: 'Default system.',
                user_prompt: 'Default user.',
                reference_chunk_count: 0,
                known_labels: [],
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-review-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-prompt-review')).toBeInTheDocument();
        });
        // Click Send without editing.
        await userEvent.click(screen.getByTestId('llm-gold-prompt-review-send'));

        await waitFor(() => {
            const generateCalls = apiMock.post.mock.calls.filter(
                (call: unknown[]) =>
                    String(call[0] || '').endsWith('/gold/generate-via-llm'),
            );
            expect(generateCalls.length).toBeGreaterThanOrEqual(1);
            const body = generateCalls[generateCalls.length - 1][1];
            expect(body.user_prompt_override).toBeUndefined();
            expect(body.system_prompt_override).toBeUndefined();
        });
    });

    it('Cancel from review discards edits + does NOT fire /generate-via-llm', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-review-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-prompt-review')).toBeInTheDocument();
        });
        // Edit the user prompt (would be lost on Cancel).
        fireEvent.change(screen.getByTestId('llm-gold-prompt-review-user'), {
            target: { value: 'will be discarded' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-prompt-review-cancel'));
        expect(screen.queryByTestId('llm-gold-prompt-review')).not.toBeInTheDocument();
        // No /generate-via-llm POST fired.
        const genCalls = apiMock.post.mock.calls.filter(
            (call: unknown[]) =>
                String(call[0] || '').endsWith('/gold/generate-via-llm'),
        );
        expect(genCalls).toHaveLength(0);
    });

    it('Reference-chunk count surfaces in the review section when grounded', async () => {
        installPostRouter({
            previewPrompt: {
                recipe_id: 'qa-sft',
                system_prompt: 'System.',
                user_prompt: 'User prompt with REFERENCE MATERIAL ... and refs.',
                reference_chunk_count: 4,
                known_labels: [],
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-review-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-prompt-review')).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('llm-gold-prompt-review-refs').textContent,
        ).toMatch(/4 reference chunks/);
    });

    it('Token-count hints update as the user types in the review textareas', async () => {
        installPostRouter({
            previewPrompt: {
                recipe_id: 'qa-sft',
                system_prompt: 'sys',
                user_prompt: 'user',
                reference_chunk_count: 0,
                known_labels: [],
            },
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        await userEvent.click(screen.getByTestId('llm-gold-review-toggle'));
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        const userArea = await screen.findByTestId('llm-gold-prompt-review-user');
        // Add a 400-char string → ~100 tokens.
        fireEvent.change(userArea, { target: { value: 'x'.repeat(400) } });
        expect(
            screen.getByTestId('llm-gold-prompt-review-user-tokens').textContent,
        ).toMatch(/100 tokens/);
    });

    // ── Per-recipe rendering + save payload ────────────────────────

    it('classification: renders text + label badges + headline copy', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'classification',
                rows: [
                    { text: 'I love this app', label: 'positive' },
                    { text: 'It crashes constantly', label: 'negative' },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="classification"
                onRowsSaved={() => {}}
            />,
        );
        // Recipe-specific headline copy.
        expect(screen.getByText(/classification examples/i)).toBeInTheDocument();
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        // text + label appear via recipe-specific testids.
        expect(screen.getByTestId('llm-gold-preview-row-0-text').textContent).toBe(
            'I love this app',
        );
        expect(screen.getByTestId('llm-gold-preview-row-0-label').textContent).toContain(
            'positive',
        );
        // No qa-sft framing on the row body.
        expect(
            screen.queryByText(/^Q:/),
        ).not.toBeInTheDocument();
    });

    it('classification: Save sends pairs with text/label keys only', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'classification',
                rows: [
                    { text: 'good', label: 'positive' },
                    { text: 'bad', label: 'negative' },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                recipeId="classification"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('llm-gold-save'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/gold/import',
                expect.objectContaining({
                    dataset_type: 'gold_dev',
                    pairs: [
                        { text: 'good', label: 'positive' },
                        { text: 'bad', label: 'negative' },
                    ],
                }),
            );
        });
    });

    it('span-extraction: renders entities with type + offsets', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'span-extraction',
                rows: [
                    {
                        text: 'Contact jane@example.com today',
                        entities: [
                            {
                                type: 'email',
                                start: 8,
                                end: 24,
                                text: 'jane@example.com',
                            },
                        ],
                    },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="span-extraction"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        const entitiesBlock = screen.getByTestId('llm-gold-preview-row-0-entities');
        expect(entitiesBlock.textContent).toContain('email');
        expect(entitiesBlock.textContent).toContain('jane@example.com');
        // Offsets surfaced inline so the user can sanity-check positions.
        expect(entitiesBlock.textContent).toContain('[8:24]');
    });

    it('span-extraction: empty entities renders the negative-example hint', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'span-extraction',
                rows: [{ text: 'clean text, no PII', entities: [] }],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="span-extraction"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('llm-gold-preview-row-0-entities').textContent,
        ).toMatch(/negative example/i);
    });

    it('span-extraction: Save sends pairs with text + entities arrays', async () => {
        const spans = [
            { type: 'email', start: 8, end: 24, text: 'jane@example.com' },
        ];
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'span-extraction',
                rows: [
                    { text: 'Contact jane@example.com today', entities: spans },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={7}
                datasetType="gold_test"
                recipeId="span-extraction"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('llm-gold-save'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/7/gold/import',
                expect.objectContaining({
                    dataset_type: 'gold_test',
                    pairs: [
                        {
                            text: 'Contact jane@example.com today',
                            entities: spans,
                        },
                    ],
                }),
            );
        });
    });

    it('summarization: renders document (collapsed) + summary', async () => {
        const longDoc = 'This is a long document. '.repeat(8);
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'summarization',
                rows: [
                    { document: longDoc, summary: 'Short summary.' },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                recipeId="summarization"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        // Document collapsed by default in a <details> element.
        const docBlock = screen.getByTestId('llm-gold-preview-row-0-document');
        expect(docBlock.tagName.toLowerCase()).toBe('details');
        expect(docBlock.textContent).toMatch(/Document \(\d+ chars\)/);
        // Summary visible.
        expect(
            screen.getByTestId('llm-gold-preview-row-0-summary').textContent,
        ).toContain('Short summary.');
    });

    it('summarization: Save sends pairs with document/summary keys', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                recipe_id: 'summarization',
                rows: [
                    {
                        document: 'Long document about the meeting.',
                        summary: 'Meeting summary.',
                    },
                ],
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={42}
                datasetType="gold_dev"
                recipeId="summarization"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('llm-gold-save'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/gold/import',
                expect.objectContaining({
                    dataset_type: 'gold_dev',
                    pairs: [{
                        document: 'Long document about the meeting.',
                        summary: 'Meeting summary.',
                    }],
                }),
            );
        });
    });

    it('source_excerpt renders per row when present', async () => {
        installPostRouter({
            generate: makeGenerateResponse({
                rows: [
                    {
                        question: 'Q?',
                        answer: 'A.',
                        rationale: '',
                        source_excerpt: 'this is the supporting passage',
                    },
                ],
                reference_chunk_count: 5,
                estimated_cost_usd: 0.0012,
            }),
        });
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        fireEvent.change(screen.getByTestId('llm-gold-api-key'), {
            target: { value: 'sk-test' },
        });
        await userEvent.click(screen.getByTestId('llm-gold-generate'));
        await waitFor(() => {
            expect(screen.getByTestId('llm-gold-preview')).toBeInTheDocument();
        });
        const source = screen.getByTestId('llm-gold-preview-row-0-source');
        expect(source.textContent).toMatch(/From source/);
        expect(source.textContent).toMatch(/this is the supporting passage/);
        // Meta line shows the actual cost + chunk count.
        const meta = screen.getByTestId('llm-gold-preview-meta').textContent || '';
        expect(meta).toMatch(/\$0\.0012 spent/);
        expect(meta).toMatch(/grounded in 5 chunks/);
    });

    // ── E1 — cluster-fix deep link prefill ───────────────────────────

    it('renders no cluster-fix banner when the URL carries no cluster params', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
        );
        expect(screen.queryByTestId('llm-gold-cluster-fix-banner')).not.toBeInTheDocument();
    });

    it('prefills focusHint + trap count and renders the banner for qa-sft', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
            {
                search: '?focus_cluster_id=cluster-42&focus_hint=%5Bhallucination%5D+Model+fabricates+dates&trap_count=5',
            },
        );
        // Banner present + carries the cluster id + the hint text.
        const banner = await screen.findByTestId('llm-gold-cluster-fix-banner');
        expect(banner.textContent).toMatch(/cluster-42/);
        const hintLine = screen.getByTestId('llm-gold-cluster-fix-hint');
        expect(hintLine.textContent).toMatch(/Model fabricates dates/);
        // Focus textarea prefilled with the hint (without the brackets
        // collapsing — the hint went through encodeURIComponent so the
        // [ and ] survived).
        const focus = screen.getByTestId('llm-gold-focus-hint') as HTMLTextAreaElement;
        expect(focus.value).toMatch(/\[hallucination\] Model fabricates dates/);
        // qa-sft default → mix is customizable → traps slot populated.
        const trapLine = screen.getByTestId('llm-gold-cluster-fix-trap-count');
        expect(trapLine.textContent).toMatch(/5/);
    });

    it('on a non-qa-sft recipe, applies the focus hint but flags the trap-count skip', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
                recipeId="classification"
            />,
            {
                search: '?focus_cluster_id=cluster-7&focus_hint=label+drift&trap_count=5',
            },
        );
        await screen.findByTestId('llm-gold-cluster-fix-banner');
        // Hint flowed through.
        const focus = screen.getByTestId('llm-gold-focus-hint') as HTMLTextAreaElement;
        expect(focus.value).toBe('label drift');
        // Banner explains that the trap mix is qa-sft-only — the user
        // shouldn't expect the row mix UI to appear.
        expect(screen.getByTestId('llm-gold-cluster-fix-trap-skip').textContent)
            .toMatch(/qa-sft-only/);
        // qa-sft trap count line is NOT rendered (recipeId !== 'qa-sft').
        expect(screen.queryByTestId('llm-gold-cluster-fix-trap-count')).not.toBeInTheDocument();
    });

    it('dismiss button hides the banner (URL params + applied prefill stay)', async () => {
        installPostRouter();
        renderPanel(
            <LlmGoldGeneratePanel
                projectId={1}
                datasetType="gold_dev"
                onRowsSaved={() => {}}
            />,
            {
                search: '?focus_cluster_id=cluster-9&focus_hint=foo&trap_count=3',
            },
        );
        await screen.findByTestId('llm-gold-cluster-fix-banner');

        await userEvent.click(screen.getByTestId('llm-gold-cluster-fix-dismiss'));
        expect(screen.queryByTestId('llm-gold-cluster-fix-banner')).not.toBeInTheDocument();
        // The prefill itself stays — focusHint textarea retains the value.
        const focus = screen.getByTestId('llm-gold-focus-hint') as HTMLTextAreaElement;
        expect(focus.value).toBe('foo');
    });
});
