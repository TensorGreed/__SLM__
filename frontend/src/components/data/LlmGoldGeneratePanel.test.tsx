import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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


function makeGenerateResponse(overrides: Record<string, unknown> = {}) {
    return {
        rows: [
            { question: 'Q1?', answer: 'A1.', rationale: 'because', source_excerpt: '' },
            { question: 'Q2?', answer: 'A2.', rationale: '', source_excerpt: '' },
            { question: 'Q3?', answer: 'A3.', rationale: '', source_excerpt: '' },
        ],
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
} = {}) {
    apiMock.post.mockImplementation(async (url: string) => {
        if (url.includes('/gold/generate-via-llm/cost-estimate')) {
            return { data: opts.estimate ?? makeCostEstimateResponse() };
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


describe('LlmGoldGeneratePanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.success.mockReset();
        toastMock.warning.mockReset();
        toastMock.error.mockReset();
    });

    it('Anthropic model dropdown swaps when provider switches', async () => {
        installPostRouter();
        render(
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
        render(
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
            expect.objectContaining({ timeout: 180_000 }),
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
        render(
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
        render(
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
        render(
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

    it('renders the upstream provider error (502) as a plain string', async () => {
        installPostRouter({
            generateError: {
                response: {
                    status: 502,
                    data: { detail: 'OpenAI returned 401: invalid API key' },
                },
            },
        });
        render(
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
        render(
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
        render(
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
        render(
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
        render(
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
        render(
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
});
