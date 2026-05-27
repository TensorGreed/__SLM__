import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

import TrainabilityForecastPanel from './TrainabilityForecastPanel';
import type { ForecastResult, ForecastSignal, SuggestedActionKind } from '../../api/trainabilityForecast';


function makeForecast(overrides: Partial<ForecastResult> = {}): ForecastResult {
    return {
        overall: 'likely_pass',
        confidence_pct: 72,
        signals: [
            {
                id: 'row_count_below_minimum',
                severity: 'ok',
                headline: '200 training rows — above recipe minimum.',
                detail: 'Recipe minimum is 50; you\'re comfortably above.',
                suggested_action: null,
            },
            {
                id: 'gate_pass_probability',
                severity: 'ok',
                headline: 'Predicted gate-pass probability: ~72%',
                detail: 'Most signals look healthy.',
                suggested_action: null,
            },
        ],
        computed_at: '2026-05-23T00:00:00Z',
        cache_key: 'abc123',
        cache_hit: false,
        ...overrides,
    };
}


describe('TrainabilityForecastPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders the verdict badge and confidence band on success', async () => {
        apiMock.get.mockResolvedValue({ data: makeForecast() });
        render(<TrainabilityForecastPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast')).toBeInTheDocument();
        });
        expect(screen.getByText('Likely to pass gates')).toBeInTheDocument();
        // Confidence chip in the header (one of two occurrences of "~72%"
        // — the other lives inside the gate_pass_probability signal row).
        const confidenceChip = screen.getByText('Predicted gate-pass:').parentElement;
        expect(confidenceChip?.textContent).toMatch(/~72%/);
    });

    it('renders each signal row with severity-coded marker', async () => {
        apiMock.get.mockResolvedValue({
            data: makeForecast({
                overall: 'likely_fail',
                confidence_pct: 28,
                signals: [
                    {
                        id: 'row_count_below_minimum',
                        severity: 'block',
                        headline: 'Only 16 training rows — recipe recommends at least 50.',
                        detail: 'You\'re 34 rows short.',
                        suggested_action: {
                            kind: 'synth_augment',
                            params: { target_rows: 50 },
                        },
                    },
                    {
                        id: 'gate_pass_probability',
                        severity: 'warn',
                        headline: 'Predicted gate-pass probability: ~28%',
                        detail: 'Multiple signals suggest under-resourcing.',
                        suggested_action: null,
                    },
                ],
            }),
        });
        render(<TrainabilityForecastPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast-signal-row_count_below_minimum')).toBeInTheDocument();
        });
        const blockSignal = screen.getByTestId('trainability-forecast-signal-row_count_below_minimum');
        expect(blockSignal).toHaveAttribute('data-severity', 'block');
        const warnSignal = screen.getByTestId('trainability-forecast-signal-gate_pass_probability');
        expect(warnSignal).toHaveAttribute('data-severity', 'warn');
    });

    it('fires onActionClicked when the suggested action button is clicked', async () => {
        apiMock.get.mockResolvedValue({
            data: makeForecast({
                overall: 'likely_fail',
                signals: [
                    {
                        id: 'row_count_below_minimum',
                        severity: 'block',
                        headline: 'Only 16 rows.',
                        detail: '',
                        suggested_action: {
                            kind: 'synth_augment',
                            params: { target_rows: 50 },
                        },
                    },
                ],
            }),
        });
        const onAction = vi.fn();
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={onAction} />);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Generate more training rows/i })).toBeInTheDocument();
        });
        await userEvent.click(screen.getByRole('button', { name: /Generate more training rows/i }));
        expect(onAction).toHaveBeenCalledWith('synth_augment', { target_rows: 50 });
    });

    it('refreshes the forecast and bypasses cache when Refresh is clicked', async () => {
        // After T2, every load() fires two GETs: the forecast itself
        // and the snapshot-history endpoint. We mock both with a
        // route-aware implementation + a mutable forecast slot so the
        // refresh click flips the served confidence.
        let nextForecast = makeForecast({ cache_hit: false });
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return { data: nextForecast };
            }
            if (url.endsWith('/training/forecast/history')) {
                return { data: { project_id: 5, snapshots: [] } };
            }
            return { data: {} };
        });

        render(<TrainabilityForecastPanel projectId={5} />);

        await waitFor(() => {
            const chip = screen.getByText('Predicted gate-pass:').parentElement;
            expect(chip?.textContent).toMatch(/~72%/);
        });
        // Forecast endpoint hit first with no refresh param.
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/training/forecast', { params: undefined });

        nextForecast = makeForecast({ cache_hit: false, confidence_pct: 80 });
        await userEvent.click(screen.getByRole('button', { name: /Refresh forecast/i }));
        await waitFor(() => {
            const chip = screen.getByText('Predicted gate-pass:').parentElement;
            expect(chip?.textContent).toMatch(/~80%/);
        });
        // Refresh path: forecast endpoint hit with refresh=true.
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/training/forecast', { params: { refresh: true } });
    });

    it('renders a cache-hit note when result was served from cache', async () => {
        apiMock.get.mockResolvedValue({ data: makeForecast({ cache_hit: true }) });
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByText(/Cached result/i)).toBeInTheDocument();
        });
    });

    it('returns null silently when the project has no recipe (400)', async () => {
        apiMock.get.mockRejectedValue({
            response: { status: 400, data: { detail: 'Project has no selected recipe' } },
        });
        const { container } = render(<TrainabilityForecastPanel projectId={99} />);
        // Wait for the loading state to clear.
        await waitFor(() => {
            expect(screen.queryByTestId('trainability-forecast-loading')).not.toBeInTheDocument();
        });
        // Panel collapses to nothing — nothing visible from the component.
        expect(container.querySelector('.trainability-forecast')).toBeNull();
    });

    it('surfaces a retry option when the request fails with a non-400 status', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'boom' } },
        });
        apiMock.get.mockResolvedValueOnce({ data: makeForecast() });
        render(<TrainabilityForecastPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText(/boom/)).toBeInTheDocument();
        });
        await userEvent.click(screen.getByRole('button', { name: /Retry/i }));
        await waitFor(() => {
            expect(screen.getByText('Likely to pass gates')).toBeInTheDocument();
        });
    });

    // ── T2 — sparkline + verdict-delta strip ───────────────────────
    // Mounted above the signal list once there are >=2 snapshots in
    // the history endpoint payload. A single point isn't a trend, so
    // the strip stays hidden until the second compute lands.

    function mockForecastWithHistory(snapshots: any[]) {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return { data: makeForecast() };
            }
            if (url.endsWith('/training/forecast/history')) {
                return { data: { project_id: 1, snapshots } };
            }
            return { data: {} };
        });
    }

    it('hides the history strip when there are fewer than 2 snapshots', async () => {
        mockForecastWithHistory([
            {
                id: 1,
                cache_key: 'a',
                computed_at: '2026-05-25T10:00:00Z',
                overall: 'likely_pass',
                confidence_pct: 72,
                signals: [],
            },
        ]);
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast')).toBeInTheDocument();
        });
        // History endpoint was called, but the strip isn't rendered.
        expect(apiMock.get).toHaveBeenCalledWith(
            '/projects/1/training/forecast/history',
            { params: { limit: 10 } },
        );
        expect(screen.queryByTestId('trainability-forecast-history')).not.toBeInTheDocument();
    });

    it('renders the sparkline with one dot per snapshot (chronological order)', async () => {
        // Newest-first from the API; the strip reverses for the chart.
        mockForecastWithHistory([
            { id: 3, cache_key: 'c', computed_at: '2026-05-25T12:00:00Z', overall: 'likely_pass', confidence_pct: 81, signals: [] },
            { id: 2, cache_key: 'b', computed_at: '2026-05-25T11:00:00Z', overall: 'borderline', confidence_pct: 55, signals: [] },
            { id: 1, cache_key: 'a', computed_at: '2026-05-25T10:00:00Z', overall: 'likely_fail', confidence_pct: 38, signals: [] },
        ]);
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast-history')).toBeInTheDocument();
        });
        // SVG with one circle per snapshot.
        const dots = screen.getAllByTestId(/^trainability-forecast-sparkline-dot-\d+$/);
        expect(dots).toHaveLength(3);
        // The strip plots chronologically left-to-right; dot 0 is the
        // oldest snapshot (verdict likely_fail).
        expect(dots[0].getAttribute('class')).toContain('--likely_fail');
        // Newest dot is the rightmost + carries the latest verdict.
        expect(dots[2].getAttribute('class')).toContain('--likely_pass');
    });

    it('renders verdict-delta chips for the last three transitions', async () => {
        // 4 snapshots → 3 deltas: +21, +17, -15.
        mockForecastWithHistory([
            { id: 4, cache_key: 'd', computed_at: '2026-05-25T13:00:00Z', overall: 'likely_pass', confidence_pct: 73, signals: [] },
            { id: 3, cache_key: 'c', computed_at: '2026-05-25T12:00:00Z', overall: 'borderline', confidence_pct: 52, signals: [] },
            { id: 2, cache_key: 'b', computed_at: '2026-05-25T11:00:00Z', overall: 'borderline', confidence_pct: 35, signals: [] },
            { id: 1, cache_key: 'a', computed_at: '2026-05-25T10:00:00Z', overall: 'likely_fail', confidence_pct: 50, signals: [] },
        ]);
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast-deltas')).toBeInTheDocument();
        });
        const chips = screen.getAllByTestId('trainability-forecast-delta-chip');
        expect(chips).toHaveLength(3);
        // Newest first: +21, +17, -15.
        expect(chips[0].textContent).toMatch(/\+21%/);
        expect(chips[1].textContent).toMatch(/\+17%/);
        expect(chips[2].textContent).toMatch(/-15%/);
        // Tone classes encode the direction so the UI can color-code.
        expect(chips[0].getAttribute('class')).toContain('--up');
        expect(chips[2].getAttribute('class')).toContain('--down');
    });

    it('renders a tooltip per dot with verdict + signal severities', async () => {
        mockForecastWithHistory([
            {
                id: 2,
                cache_key: 'b',
                computed_at: '2026-05-25T11:00:00Z',
                overall: 'borderline',
                confidence_pct: 55,
                signals: [
                    { id: 'row_count_below_minimum', severity: 'warn', headline: '', detail: '', suggested_action: null },
                    { id: 'class_imbalance', severity: 'ok', headline: '', detail: '', suggested_action: null },
                ],
            },
            {
                id: 1,
                cache_key: 'a',
                computed_at: '2026-05-25T10:00:00Z',
                overall: 'likely_fail',
                confidence_pct: 38,
                signals: [],
            },
        ]);
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('trainability-forecast-history')).toBeInTheDocument();
        });
        // The newer dot's <title> carries verdict + signal severities.
        const newestDot = screen.getByTestId('trainability-forecast-sparkline-dot-1');
        const titleEl = newestDot.querySelector('title');
        expect(titleEl?.textContent).toMatch(/borderline @ 55%/);
        expect(titleEl?.textContent).toMatch(/warn: row_count_below_minimum/);
        expect(titleEl?.textContent).toMatch(/ok: class_imbalance/);
    });

    // ── T1 — cost-of-fix chip + ROI hint ────────────────────────────

    /** Build a forecast with N actionable signals, each carrying a
     *  cost estimate. Used by the chip/ROI tests below so the fixture
     *  isn't repeated per case. */
    function makeForecastWithCosts(
        actions: Array<{
            id: string;
            kind: SuggestedActionKind;
            params: Record<string, unknown>;
            cost: ForecastSignal['cost_estimate'];
        }>,
    ): ForecastResult {
        return makeForecast({
            overall: 'borderline',
            confidence_pct: 55,
            signals: actions.map((a) => ({
                id: a.id,
                severity: 'warn',
                headline: `Headline for ${a.id}`,
                detail: '',
                suggested_action: { kind: a.kind, params: a.params },
                cost_estimate: a.cost,
            })),
        });
    }

    it('renders the cost chip next to the action button for an actionable signal', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return {
                    data: makeForecastWithCosts([{
                        id: 'row_count_below_minimum',
                        kind: 'synth_augment',
                        params: { target_rows: 50 },
                        cost: { time_minutes: 25, llm_cost_usd: 0.005, confidence: 'rough' },
                    }]),
                };
            }
            return { data: { project_id: 1, snapshots: [] } };
        });
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={() => {}} />);
        const chip = await screen.findByTestId('trainability-forecast-cost-row_count_below_minimum');
        // LLM cost under $0.01 collapses to "<$0.01".
        expect(chip.textContent).toMatch(/~25 min/);
        expect(chip.textContent).toMatch(/<\$0\.01/);
    });

    it('renders "no $" for manual fix_gold_rows actions (llm_cost_usd is null)', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return {
                    data: makeForecastWithCosts([{
                        id: 'label_vocab_fragmented',
                        kind: 'fix_gold_rows',
                        params: { fragment_groups: [['a', 'A'], ['b', 'B']] },
                        cost: { time_minutes: 4, llm_cost_usd: null, confidence: 'rough' },
                    }]),
                };
            }
            return { data: { project_id: 1, snapshots: [] } };
        });
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={() => {}} />);
        const chip = await screen.findByTestId('trainability-forecast-cost-label_vocab_fragmented');
        expect(chip.textContent).toMatch(/~4 min/);
        expect(chip.textContent).toMatch(/no \$/);
    });

    it('does NOT render the ROI hint when only one signal carries an action', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return {
                    data: makeForecastWithCosts([{
                        id: 'row_count_below_minimum',
                        kind: 'synth_augment',
                        params: { target_rows: 50 },
                        cost: { time_minutes: 25, llm_cost_usd: 0.005, confidence: 'rough' },
                    }]),
                };
            }
            return { data: { project_id: 1, snapshots: [] } };
        });
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={() => {}} />);
        await screen.findByTestId('trainability-forecast');
        expect(screen.queryByTestId('trainability-forecast-roi-hint')).not.toBeInTheDocument();
    });

    it('ranks the cheapest action first in the ROI hint when ≥2 carry actions', async () => {
        // 3 actionable signals — the 2-min fix beats the 4-min and 25-min
        // synth runs even though it has no LLM cost.
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return {
                    data: makeForecastWithCosts([
                        {
                            id: 'row_count_below_minimum',
                            kind: 'synth_augment',
                            params: { target_rows: 50 },
                            cost: { time_minutes: 25, llm_cost_usd: 0.005, confidence: 'rough' },
                        },
                        {
                            id: 'label_vocab_fragmented',
                            kind: 'fix_gold_rows',
                            params: { fragment_groups: [['a', 'A']] },
                            cost: { time_minutes: 2, llm_cost_usd: null, confidence: 'rough' },
                        },
                        {
                            id: 'goldset_diversity_low',
                            kind: 'synth_diversify',
                            params: { target_rows: 8 },
                            cost: { time_minutes: 4, llm_cost_usd: 0.0008, confidence: 'rough' },
                        },
                    ]),
                };
            }
            return { data: { project_id: 1, snapshots: [] } };
        });
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={() => {}} />);
        const hint = await screen.findByTestId('trainability-forecast-roi-hint');
        // Wall-clock time wins → label_vocab_fragmented is cheapest.
        expect(screen.getByTestId('trainability-forecast-roi-cheapest-id').textContent)
            .toBe('label_vocab_fragmented');
        expect(hint.textContent).toMatch(/~2 min/);
    });

    it('breaks time-tie ties on llm_cost_usd (lower cost wins)', async () => {
        // Two actions land at the same time_minutes — the one with
        // lower llm_cost_usd should rank first.
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return {
                    data: makeForecastWithCosts([
                        {
                            id: 'expensive',
                            kind: 'synth_augment',
                            params: { target_rows: 50 },
                            cost: { time_minutes: 25, llm_cost_usd: 0.05, confidence: 'rough' },
                        },
                        {
                            id: 'cheaper',
                            kind: 'synth_diversify',
                            params: { target_rows: 50 },
                            cost: { time_minutes: 25, llm_cost_usd: 0.005, confidence: 'rough' },
                        },
                    ]),
                };
            }
            return { data: { project_id: 1, snapshots: [] } };
        });
        render(<TrainabilityForecastPanel projectId={1} onActionClicked={() => {}} />);
        await screen.findByTestId('trainability-forecast-roi-hint');
        expect(screen.getByTestId('trainability-forecast-roi-cheapest-id').textContent)
            .toBe('cheaper');
    });

    it('survives a 5xx history fetch without breaking the live forecast render', async () => {
        // History endpoint flakes; the forecast read still succeeds.
        // The panel must show the verdict badge and just skip the strip.
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/training/forecast')) {
                return { data: makeForecast() };
            }
            if (url.endsWith('/training/forecast/history')) {
                throw { response: { status: 500, data: { detail: 'boom' } } };
            }
            return { data: {} };
        });
        render(<TrainabilityForecastPanel projectId={1} />);
        await waitFor(() => {
            expect(screen.getByText('Likely to pass gates')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('trainability-forecast-history')).not.toBeInTheDocument();
    });
});
