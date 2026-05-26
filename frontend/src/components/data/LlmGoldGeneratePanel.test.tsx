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
            { question: 'Q1?', answer: 'A1.', rationale: 'because' },
            { question: 'Q2?', answer: 'A2.', rationale: '' },
            { question: 'Q3?', answer: 'A3.', rationale: '' },
        ],
        provider: 'openai',
        model: 'gpt-4o-mini',
        usage: { prompt_tokens: 120, completion_tokens: 350 },
        prompt_preview: 'PROJECT: …',
        ...overrides,
    };
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
        apiMock.post.mockResolvedValueOnce({ data: makeGenerateResponse() });
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

        // POST body shape.
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/42/gold/generate-via-llm',
            expect.objectContaining({
                provider: 'openai',
                model: 'gpt-4o-mini',
                count: 3,
                api_key: 'sk-test',
                focus_hint: 'edge cases around refunds',
            }),
            expect.objectContaining({ timeout: 180_000 }),
        );

        // Save button reflects the all-selected default.
        expect(screen.getByTestId('llm-gold-save').textContent).toMatch(
            /Save 3 of 3/,
        );
        // Meta line surfaces provider + model + token total.
        expect(screen.getByTestId('llm-gold-preview-meta').textContent).toMatch(
            /openai · gpt-4o-mini · 470 tokens/,
        );
    });

    it('Save selected calls /gold/import with only the checked rows', async () => {
        apiMock.post.mockResolvedValueOnce({ data: makeGenerateResponse() });
        apiMock.post.mockResolvedValueOnce({ data: { imported: 2 } });
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
        apiMock.post.mockResolvedValueOnce({ data: makeGenerateResponse() });
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
        // 1 POST so far (generate). After discard, should still be 1.
        expect(apiMock.post).toHaveBeenCalledTimes(1);
        await userEvent.click(screen.getByTestId('llm-gold-discard'));
        expect(apiMock.post).toHaveBeenCalledTimes(1);
        expect(screen.queryByTestId('llm-gold-preview')).not.toBeInTheDocument();
    });

    it('renders structured error code + message on 400 RECIPE_NOT_SUPPORTED', async () => {
        apiMock.post.mockRejectedValueOnce({
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
        apiMock.post.mockRejectedValueOnce({
            response: {
                status: 502,
                data: { detail: 'OpenAI returned 401: invalid API key' },
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
});
