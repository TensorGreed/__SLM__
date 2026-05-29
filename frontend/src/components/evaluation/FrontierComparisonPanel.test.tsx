import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('../../api/frontierComparison', () => ({ fetchFrontierComparison: fetchMock }));

import FrontierComparisonPanel from './FrontierComparisonPanel';

function comparison(overrides: Record<string, unknown> = {}) {
    return {
        project_id: 5,
        frontier_model: { id: 'gpt-4o-mini', display_name: 'GPT-4o mini', source: 'OpenAI published API pricing', as_of: '2026-05' },
        slm: { experiment_id: 7, experiment_name: 'sft', base_model: 'HF/SmolLM2-135M' },
        frontier: null,
        quality: { status: 'ok', metric_comparisons: [{ metric_id: 'f1', slm_value: 0.46, frontier_value: 0.5, quality_pct: 0.92, direction: 'behind', is_headline: true }], headline_quality_pct: 0.92, frontier_baseline_run_id: 9, message: null },
        cost: { frontier_usd_per_1m_tokens: 0.375, frontier_source: 'OpenAI', slm_usd_per_1m_tokens: 0.05, cost_pct: 13.3, provenance: 'estimated', source: 'self-host estimate', message: null },
        latency: { frontier_latency_ms: 700, frontier_source: 'OpenAI typical', slm_latency_ms: 70, latency_ratio: 0.1, provenance: 'estimated', source: 'benchmark sweep', message: null },
        headline: 'Your model is 92% as good as GPT-4o mini at 13.3% of the cost and 0.1× the latency.',
        ...overrides,
    };
}

describe('FrontierComparisonPanel', () => {
    beforeEach(() => fetchMock.mockReset());

    it('renders the headline + per-metric quality table', async () => {
        fetchMock.mockResolvedValueOnce(comparison());
        render(<FrontierComparisonPanel projectId={5} experimentId={7} />);
        await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(5, 7));
        expect(await screen.findByTestId('frontier-headline')).toHaveTextContent('92% as good as GPT-4o mini');
        expect(screen.getByText('f1')).toBeInTheDocument();
        expect(screen.getByText('92%')).toBeInTheDocument();
        // provenance badges for cost + latency.
        expect(screen.getAllByText('estimated').length).toBeGreaterThanOrEqual(2);
        expect(screen.getByText('13.3% of cost')).toBeInTheDocument();
        expect(screen.getByText('0.1× latency')).toBeInTheDocument();
    });

    it('shows the soft-fallback message when there is no frontier baseline eval', async () => {
        fetchMock.mockResolvedValueOnce(
            comparison({
                quality: { status: 'no_frontier_eval', metric_comparisons: [], headline_quality_pct: null, frontier_baseline_run_id: null, message: 'No GPT-4o mini baseline eval on this eval set.' },
                headline: 'Your model runs at 13.3% of the cost and 0.1× the latency vs GPT-4o mini.',
            }),
        );
        render(<FrontierComparisonPanel projectId={5} experimentId={7} />);
        expect(await screen.findByTestId('frontier-quality-fallback')).toHaveTextContent('No GPT-4o mini baseline eval');
        // No quality table rows.
        expect(screen.queryByText('f1')).not.toBeInTheDocument();
    });

    it('marks the SLM cost/latency unavailable without fabricating', async () => {
        fetchMock.mockResolvedValueOnce(
            comparison({
                cost: { frontier_usd_per_1m_tokens: 0.375, frontier_source: 'OpenAI', slm_usd_per_1m_tokens: null, cost_pct: null, provenance: 'unavailable', message: 'Run a benchmark sweep.' },
                latency: { frontier_latency_ms: 700, frontier_source: 'OpenAI', slm_latency_ms: null, latency_ratio: null, provenance: 'unavailable', message: 'Run a benchmark sweep.' },
            }),
        );
        render(<FrontierComparisonPanel projectId={5} experimentId={7} />);
        await screen.findByTestId('frontier-comparison');
        expect(screen.getAllByText('unavailable').length).toBeGreaterThanOrEqual(2);
        expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(2); // SLM cost + latency dashes
    });

    it('self-hides on error', async () => {
        fetchMock.mockRejectedValueOnce({ response: { data: { detail: 'boom' } } });
        const { container } = render(<FrontierComparisonPanel projectId={5} experimentId={7} />);
        await waitFor(() => expect(fetchMock).toHaveBeenCalled());
        await waitFor(() => expect(container.firstChild).toBeNull());
    });
});
