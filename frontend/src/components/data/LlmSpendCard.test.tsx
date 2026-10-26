import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import LlmSpendCard, { budgetLineText, formatTokens, formatUsd } from './LlmSpendCard';
import { budgetExceededMessage, budgetWarningMessage } from '../../api/llmSpend';

const SPEND = {
    project_id: 7,
    days: 30,
    since: '2026-09-15T00:00:00+00:00',
    totals: {
        operations: 3, calls: 41, input_tokens: 90_000, output_tokens: 12_000, cache_write_tokens: 2_000,
        cache_read_tokens: 40_000, total_tokens: 144_000, cost_usd: 0.42, unpriced_tokens: 0,
    },
    by_model: [
        { model: 'claude-haiku-4-5', calls: 40, input_tokens: 90_000, output_tokens: 12_000, cache_write_tokens: 2_000,
            cache_read_tokens: 40_000, total_tokens: 144_000, cost_usd: 0.42, priced: true },
        { model: 'mystery', calls: 1, input_tokens: 0, output_tokens: 0, cache_write_tokens: 0, cache_read_tokens: 0,
            total_tokens: 0, cost_usd: 0, priced: false },
    ],
    by_kind: [
        { kind: 'span_relabel', operations: 2, calls: 40, total_tokens: 140_000, cost_usd: 0.4 },
        { kind: 'gold_llm_generate', operations: 1, calls: 1, total_tokens: 4_000, cost_usd: 0.02 },
    ],
    basis: 'list-price estimate',
    note: 'Estimated from token counts at list prices.',
};

describe('LlmSpendCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('formats tokens and dollars', () => {
        expect(formatTokens(940)).toBe('940');
        expect(formatTokens(12_400)).toBe('12k');
        expect(formatTokens(1_250_000)).toBe('1.3M');
        expect(formatUsd(0.004)).toBe('under $0.01');
        expect(formatUsd(1.5)).toBe('~$1.50');
    });

    it('shows tokens, the list-price estimate, models and operations', async () => {
        apiMock.get.mockResolvedValue({ data: SPEND });
        render(<LlmSpendCard projectId={7} />);
        expect(apiMock.get).toHaveBeenCalledWith('/projects/7/llm-spend', { params: { days: 30 } });
        const headline = await screen.findByTestId('llm-spend-headline');
        expect(headline.textContent).toContain('144k tokens');
        expect(headline.textContent).toContain('~$0.42');
        expect(headline.textContent).toContain('list-price estimate');
        expect(headline.textContent).toContain('40k read from the prompt cache');
        expect(screen.getByText('Cloud LLM spend (last 30 days)')).toBeTruthy();
        const table = screen.getByTestId('llm-spend-by-model');
        expect(table.textContent).toContain('claude-haiku-4-5');
        expect(table.textContent).toContain('price unknown');
        expect(screen.getByTestId('llm-spend-by-kind').textContent).toContain('Relabels / answer keys ~$0.40');
        expect(screen.getByTestId('llm-spend-by-kind').textContent).toContain('LLM answer key');
    });

    it('says so when nothing was spent', async () => {
        apiMock.get.mockResolvedValue({ data: { ...SPEND, totals: { ...SPEND.totals, operations: 0 }, by_model: [], by_kind: [] } });
        render(<LlmSpendCard projectId={7} />);
        expect((await screen.findByTestId('llm-spend-empty')).textContent).toContain('No cloud-model calls');
    });

    it('renders nothing when the tally cannot load', async () => {
        apiMock.get.mockRejectedValue(new Error('down'));
        const { container } = render(<LlmSpendCard projectId={7} />);
        await vi.waitFor(() => expect(container.querySelector('[data-testid="llm-spend-card"]')).toBeNull());
    });

    it('shows the month-to-date budget line with warning / over styling', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                ...SPEND,
                month: { since: '2026-10-01T00:00:00+00:00', operations: 2, total_tokens: 100_000, cost_usd: 4.1 },
                budget: { monthly: 5, warn_at: 0.8 },
                budget_status: { budget: 5, warn_at: 0.8, spent: 4.1, share: 0.82, status: 'warning' },
            },
        });
        render(<LlmSpendCard projectId={7} />);
        const line = await screen.findByTestId('llm-spend-budget-line');
        expect(line.textContent).toContain('This month: ~$4.10 of $5.00 (82%)');
        expect(line.className).toContain('llm-spend-card__budget-line--warning');
        expect(screen.getByTestId('llm-spend-budget-edit').textContent).toBe('Edit budget');
        expect(budgetLineText({ spent: 6, budget: 5, share: 1.2 })).toBe('This month: ~$6.00 of $5.00 (120%)');
    });

    it('marks an over-budget month', async () => {
        apiMock.get.mockResolvedValue({
            data: { ...SPEND, budget: { monthly: 1, warn_at: 0.8 }, budget_status: { budget: 1, warn_at: 0.8, spent: 1.5, share: 1.5, status: 'over' } },
        });
        render(<LlmSpendCard projectId={7} />);
        const line = await screen.findByTestId('llm-spend-budget-line');
        expect(line.className).toContain('llm-spend-card__budget-line--over');
        expect(line.textContent).toContain('over budget');
    });

    it('sets a budget from the editor and re-reads the tally', async () => {
        apiMock.get.mockResolvedValue({ data: { ...SPEND, month: { since: 'x', operations: 0, total_tokens: 0, cost_usd: 0.3 }, budget: null, budget_status: null } });
        apiMock.put.mockResolvedValue({ data: { project_id: 7, budget: { monthly: 10, warn_at: 0.5 }, status: null } });
        render(<LlmSpendCard projectId={7} />);
        expect((await screen.findByTestId('llm-spend-budget-line')).textContent).toContain('no monthly budget set');
        fireEvent.click(screen.getByTestId('llm-spend-budget-edit'));
        fireEvent.change(screen.getByTestId('llm-spend-budget-monthly'), { target: { value: '0' } });
        fireEvent.click(screen.getByTestId('llm-spend-budget-save'));
        expect((await screen.findByTestId('llm-spend-budget-error')).textContent).toContain('above $0');
        expect(apiMock.put).not.toHaveBeenCalled();
        fireEvent.change(screen.getByTestId('llm-spend-budget-monthly'), { target: { value: '10' } });
        fireEvent.change(screen.getByTestId('llm-spend-budget-warn'), { target: { value: '50' } });
        fireEvent.click(screen.getByTestId('llm-spend-budget-save'));
        await vi.waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/7/llm-budget', { monthly: 10, warn_at: 0.5 }));
        await vi.waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
    });

    it('reads a Job start\'s budget_warning', () => {
        expect(budgetWarningMessage({ id: 3 })).toBeNull();
        expect(budgetWarningMessage({ id: 3, budget_warning: { message: 'over its $1.00 budget' } })).toBe('over its $1.00 budget');
    });
    it('reads a 402 llm_budget_exceeded start error, and nothing else', () => {
        const envelope = { response: { status: 402, data: { error_code: 'llm_budget_exceeded', message: 'API budget reached (global: $5.00 of $5.00 this month) — switch to a local or subscription backend', detail: 'x' } } };
        expect(budgetExceededMessage(envelope)).toContain('API budget reached');
        expect(budgetExceededMessage({ response: { status: 402, data: { error_code: 'CLOUD_KEY_MISSING', detail: 'no key' } } })).toBeNull();
        expect(budgetExceededMessage(new Error('boom'))).toBeNull();
        expect(budgetExceededMessage({ response: { data: { error_code: 'llm_budget_exceeded' } } })).toContain('monthly API budget');
    });
    it('shows subscription rows without a $ figure, a Subscription CLI line, and the account-wide cap', async () => {
        apiMock.get.mockImplementation((url: string) => {
            if (url.endsWith('/llm-budget')) {
                return Promise.resolve({ data: { project_id: 7, budget: null, status: null, hard_stop: true, global: { monthly: 100, spent_usd: 42.5, status: 'ok' } } });
            }
            return Promise.resolve({
                data: {
                    ...SPEND,
                    totals: { ...SPEND.totals, subscription_calls: 12, subscription_tokens: 30_000 },
                    by_model: [
                        ...SPEND.by_model,
                        { model: 'subscription:claude', calls: 12, input_tokens: 0, output_tokens: 0, cache_write_tokens: 0,
                            cache_read_tokens: 0, total_tokens: 30_000, cost_usd: 0, priced: true, billing: 'subscription' },
                    ],
                },
            });
        });
        render(<LlmSpendCard projectId={7} />);
        const sub = await screen.findByTestId('llm-spend-subscription');
        expect(sub.textContent).toContain('12 calls');
        expect(sub.textContent).toContain('not counted against the API budget');
        const row = screen.getByTestId('llm-spend-by-model').textContent ?? '';
        expect(row).toContain('subscription:claude');
        expect(row).toContain('subscription');
        const line = await screen.findByTestId('llm-spend-global-line');
        expect(line.textContent).toContain('All projects this month: ~$42.50 of $100.00 (API credits)');
    });

    it('describes the budget as a hard stop for cloud API calls', async () => {
        apiMock.get.mockResolvedValue({ data: SPEND });
        render(<LlmSpendCard projectId={7} />);
        fireEvent.click(await screen.findByTestId('llm-spend-budget-edit'));
        expect(screen.getByTestId('llm-spend-budget-form').textContent).toContain('Cloud API calls stop at the cap');
    });
});
