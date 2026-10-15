import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import LlmSpendCard, { formatTokens, formatUsd } from './LlmSpendCard';

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
});
