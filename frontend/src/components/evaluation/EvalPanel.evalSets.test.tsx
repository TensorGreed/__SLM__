/**
 * Eval sets in the run form: the Dataset list offers the project's eval sets (with version and prompt
 * format pickers) only once the form is open; the payload carries ``eval_set:<slug>@v<N>`` +
 * ``prompt_variant``; without a set the payload is unchanged; results show the set label.
 */

import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';
import { useJobsStore } from '../../stores/jobsStore';
import { act, StrictMode } from 'react';

const SETS = [{
    id: 1, slug: 'fresh', name: 'Fresh', current_version: 2, rows: 3, is_locked: false,
    versions: [{ version: 1, rows: 4 }, { version: 2, rows: 3 }],
    variants: [{ name: 'native', spec: { system: 'strip' }, label: 'system message stripped' }],
    dataset_name: 'eval_set:fresh@v2', last_evals: {},
}];

function mockApi(variantSupport: { supported: boolean; reason: string | null } = { supported: true, reason: null }) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/eval-sets/variant-support')) return { data: { experiment_id: 21, ...variantSupport } };
        if (url.includes('/eval-sets')) return { data: { eval_sets: SETS } };
        if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [{
                    id: 1002, dataset_name: 'eval_set:fresh@v2+native', eval_type: 'exact_match', pass_rate: 0.5,
                    metrics: { exact_match: 0.5, prompt_format_version: 2 },
                    details: { dataset: { content_sha256: 'abcdef1234567890', eval_set: { name: 'Fresh', rows: 3 } } },
                }, {
                    id: 1001, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.7,
                    metrics: { exact_match: 0.7 }, details: {},
                }],
            };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/gates/21')) {
            return { data: { captured_at: '2026-03-26T00:00:00Z', passed: true, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return { data: { experiment_id: 21, is_ship: true, decision: 'SHIP', reasons: [], failed_gates: [], missing_metrics: [],
                gate_report: { passed: true, checks: [], missing_required_metrics: [], failed_gate_ids: [] } } };
        }
        if (url.includes('failure-clusters')) {
            return { data: { eval_result_id: 1002, experiment_id: 21, dataset_name: 'test', eval_type: 'exact_match',
                total_failures_analyzed: 0, reason_code_totals: {}, dominant_reason_code: null, clusters: [], remediation_plans: [] } };
        }
        if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
        if (url.includes('/gold/review')) return { data: { total: 0, rows: [] } };
        return { data: {} };
    });
}

async function openForm() {
    const user = userEvent.setup();
    render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
    await user.click(await screen.findByRole('button', { name: /exp-21/ }));
    await user.click(await screen.findByRole('button', { name: /Score this run/ }));
    return user;
}

describe('EvalPanel eval-set picker', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
    });

    it('loads the eval sets once the run has an eval-set result (the Verdict card\'s cross-prompt lines read them)', async () => {
        mockApi();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await screen.findByRole('button', { name: /exp-21/ });
        await waitFor(() => expect(apiMock.get.mock.calls.some(([url]) => String(url).endsWith('/eval-sets'))).toBe(true));
    });

    it('sends the set, the picked version and the prompt-format variant', async () => {
        mockApi();
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        const user = await openForm();
        const select = screen.getByTestId('eval-dataset-select');
        await waitFor(() => expect(select).toHaveTextContent('Fresh · v2 · 3 rows'));
        await user.selectOptions(select, 'eval_set:fresh');
        await user.selectOptions(screen.getByTestId('eval-set-version-select'), '1');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
        await user.click(screen.getByTestId('eval-run-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        const body = apiMock.post.mock.calls[0][1];
        expect(body.dataset_name).toBe('eval_set:fresh@v1');
        expect(body.prompt_variant).toBe('native');
    });

    it('disables the prompt-format picker with the reason when the run renders no system message', async () => {
        const reason = 'Prompt-format variants change only the system message, and this is a classifier-head run.';
        mockApi({ supported: false, reason });
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        const user = await openForm();
        const select = screen.getByTestId('eval-dataset-select');
        await waitFor(() => expect(select).toHaveTextContent('Fresh · v2 · 3 rows'));
        await user.selectOptions(select, 'eval_set:fresh');
        expect(await screen.findByTestId('eval-set-variant-blocked')).toHaveTextContent('classifier-head run');
        expect(screen.getByTestId('eval-set-variant-select')).toBeDisabled();
        await user.click(screen.getByTestId('eval-run-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        const body = apiMock.post.mock.calls[0][1];
        expect(body.dataset_name).toBe('eval_set:fresh@v2');
        expect(body).not.toHaveProperty('prompt_variant');
    });

    it('defaults the reroute panel to the newest built-in-set result, not an eval-set result', async () => {
        mockApi();
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="failures" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        // The reroute panel sits under Failures › Advanced (kept mounted): it still fetches for the built-in result.
        await waitFor(() => expect(apiMock.get.mock.calls.some(([url]) => String(url).includes('/reroute') && String(url).includes('1001'))).toBe(true));
        const rerouteUrls = apiMock.get.mock.calls.map(([url]) => String(url)).filter((u) => u.includes('/reroute'));
        expect(rerouteUrls.every((u) => u.includes('1001') && !u.includes('1002'))).toBe(true);
    });

    it('keeps the payload unchanged on a built-in set', async () => {
        mockApi();
        // The project has a prepared test split (it has a result on it), so the one eval set is not prefilled (phase 7c).
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string) => (
            url.endsWith('/prepared-manifest') ? { data: { splits: { train: 20, test: 100 } } } : base(url)
        ));
        apiMock.post.mockResolvedValue({ data: { id: 7 } });
        const user = await openForm();
        expect(screen.queryByTestId('eval-set-variant-select')).toBeNull();
        // Eval-set management sits in Verdict › Test sets (a closed Advanced section), not in the form.
        expect(screen.getByTestId('advanced-verdict-test-sets-toggle')).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByTestId('advanced-verdict-test-sets-body')).toBeNull();
        await user.click(screen.getByTestId('eval-run-submit'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        const body = apiMock.post.mock.calls[0][1];
        expect(body.dataset_name).toBe('test');
        expect(body).not.toHaveProperty('prompt_variant');
    });

    it('Add an eval set (in the form) opens Verdict › Test sets with the sets listed', async () => {
        mockApi();
        const user = await openForm();
        expect(screen.queryByTestId('advanced-verdict-test-sets-body')).toBeNull();
        await user.click(screen.getByTestId('eval-run-add-eval-set'));
        expect(await screen.findByTestId('advanced-verdict-test-sets-body')).toBeInTheDocument();
        expect(screen.getByTestId('advanced-verdict-test-sets-toggle')).toHaveTextContent('Test sets (1)');
    });

    it('?open=score-run-advanced opens the Score this run form with its Advanced section', async () => {
        mockApi();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" openSection="score-run-advanced" /></MemoryRouter>);
        expect(await screen.findByTestId('eval-run-form')).toBeInTheDocument();
        expect(screen.getByTestId('advanced-score-run-advanced-toggle')).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByTestId('eval-run-metric')).toBeVisible();
    });

    it('labels eval-set results in the history with set, version, variant and provenance', async () => {
        mockApi();
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        const cell = await screen.findByTestId('eval-history-dataset-1002');
        expect(cell).toHaveTextContent('eval set fresh v2 (native) "Fresh" · 3 rows · content abcdef12 · prompt rendering v2');
        expect(screen.getByTestId('eval-history-dataset-1001')).toHaveTextContent('test');
    });

    it('phase 7k: the history column is "Score", named by the score of record and in its own scale', async () => {
        mockApi();
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/evaluation/results/21')) {
                return {
                    data: [{
                        id: 1003, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.184,
                        metrics: { exact_match: 0, f1: 0.184, total: 21, correct: 0 }, details: {},
                    }, {
                        id: 1001, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.7,
                        metrics: { exact_match: 0.7, total: 10, correct: 7 }, details: {},
                    }],
                };
            }
            return base(url);
        });
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await user.click(await screen.findByRole('button', { name: /exp-21/ }));
        await screen.findByTestId('eval-history-dataset-1003');
        expect(screen.getByText('Score', { selector: 'th' })).toBeInTheDocument();
        expect(screen.queryByText('Pass Rate', { selector: 'th' })).toBeNull();
        expect(screen.getByTestId('eval-score-label-1003')).toHaveTextContent('Token F1');
        expect(screen.getByTestId('eval-pass-rate-1003')).toHaveTextContent(/^0\.184$/);
        // A 0–1 mean has no pass threshold: no colour.
        expect(screen.getByTestId('eval-pass-rate-1003').className).toBe('');
        expect(screen.getByTestId('eval-score-label-1001')).toHaveTextContent('Exact match');
        expect(screen.getByTestId('eval-pass-rate-1001')).toHaveTextContent(/^70\.0%$/);
        expect(screen.getByTestId('eval-pass-rate-1001').className).toBe('u-text-warning');
    });

    it('phase 7b (N8): the Verdict refreshes when a background eval for the run finishes (the base leg files elsewhere)', async () => {
        mockApi();
        useJobsStore.setState({ jobs: [] });
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await screen.findByTestId('verdict-card-lifts');
        const liftCalls = () => apiMock.get.mock.calls.filter(([url]) => String(url).includes('sft-lift-summary')).length;
        const resultCalls = () => apiMock.get.mock.calls.filter(([url]) => String(url).includes('/evaluation/results/21')).length;
        await waitFor(() => expect(liftCalls()).toBeGreaterThan(0));
        const [lifts, results] = [liftCalls(), resultCalls()];
        // A polling tick that changes nothing reloads nothing.
        act(() => { useJobsStore.setState({ jobs: [] }); });
        expect(resultCalls()).toBe(results);
        act(() => {
            useJobsStore.setState({ jobs: [{
                id: 900, kind: 'heldout_evaluation', title: 'Eval', status: 'succeeded', project_id: 4,
                params: { experiment_id: 21, model_path: 'Org/Base' },
            } as never] });
        });
        await waitFor(() => expect(liftCalls()).toBeGreaterThan(lifts));
        expect(resultCalls()).toBeGreaterThan(results);
        useJobsStore.setState({ jobs: [] });
    });

    it('phase 7i: the Verdict and summary refresh when the automatic post-training lift check finishes', async () => {
        mockApi();
        useJobsStore.setState({ jobs: [] });
        render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
        await screen.findByTestId('verdict-card-lifts');
        const count = (part: string) => apiMock.get.mock.calls.filter(([url]) => String(url).includes(part)).length;
        await waitFor(() => expect(count('sft-lift-summary')).toBeGreaterThan(0));
        await waitFor(() => expect(count('/evaluation/summary')).toBeGreaterThan(0));
        const [lifts, results, summaries] = [count('sft-lift-summary'), count('/evaluation/results/21'), count('/evaluation/summary')];
        // Still running: nothing reloads.
        act(() => {
            useJobsStore.setState({ jobs: [{
                id: 950, kind: 'post_training_lift_eval', title: 'Did fine-tuning help?', status: 'running', project_id: 4,
                params: { experiment_id: 21 },
            } as never] });
        });
        expect(count('/evaluation/results/21')).toBe(results);
        act(() => {
            useJobsStore.setState({ jobs: [{
                id: 950, kind: 'post_training_lift_eval', title: 'Did fine-tuning help?', status: 'succeeded', project_id: 4,
                params: { experiment_id: 21 },
            } as never] });
        });
        await waitFor(() => expect(count('sft-lift-summary')).toBeGreaterThan(lifts));
        expect(count('/evaluation/results/21')).toBeGreaterThan(results);
        await waitFor(() => expect(count('/evaluation/summary')).toBeGreaterThan(summaries));
        // A lift check for another run changes nothing here.
        const settled = count('/evaluation/results/21');
        act(() => {
            useJobsStore.setState({ jobs: [
                { id: 950, kind: 'post_training_lift_eval', title: 'x', status: 'succeeded', project_id: 4, params: { experiment_id: 21 } },
                { id: 951, kind: 'post_training_lift_eval', title: 'x', status: 'succeeded', project_id: 4, params: { experiment_id: 99 } },
            ] as never });
        });
        expect(count('/evaluation/results/21')).toBe(settled);
        useJobsStore.setState({ jobs: [] });
    });

    it('7h follow-up (G1): the run summary refetches when an eval for the run finishes, once per finished Job (StrictMode)', async () => {
        mockApi();
        let basePairsIn = false;
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string, config?: unknown) => (
            url.includes('/evaluation/summary')
                ? { data: { experiment_id: 21, verdict: 'not_evaluated', headline: null, failures: [],
                    message: basePairsIn
                        ? 'Evaluated on eval sets only: Fresh. Base vs fine-tuned on each set is on the Verdict card above.'
                        : 'Evaluated on eval sets only: Fresh. The verdict compares the run with its base model on the built-in test examples: evaluate it there.' } }
                : base(url, config)
        ));
        useJobsStore.setState({ jobs: [] });
        render(<StrictMode><MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter></StrictMode>);
        expect(await screen.findByText(/evaluate it there/)).toBeInTheDocument();
        const summaryCalls = () => apiMock.get.mock.calls.filter(([url]) => String(url).includes('/evaluation/summary')).length;
        await waitFor(() => expect(screen.queryByTestId('eval-summary-loading')).toBeNull());
        const before = summaryCalls();
        // A polling tick that changes nothing refetches nothing.
        act(() => { useJobsStore.setState({ jobs: [] }); });
        expect(summaryCalls()).toBe(before);
        basePairsIn = true;
        act(() => {
            useJobsStore.setState({ jobs: [{
                id: 901, kind: 'heldout_evaluation', title: 'Eval', status: 'succeeded', project_id: 4,
                params: { experiment_id: 21, model_path: 'Org/Base' },
            } as never] });
        });
        expect(await screen.findByText(/on the Verdict card above/)).toBeInTheDocument();
        expect(screen.queryByText(/evaluate it there/)).toBeNull();
        // The card stayed mounted (no loading flash) and the refetch settles: no loop.
        expect(screen.queryByTestId('eval-summary-loading')).toBeNull();
        const after = summaryCalls();
        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(summaryCalls()).toBe(after);
        expect(after - before).toBeLessThanOrEqual(2);
        useJobsStore.setState({ jobs: [] });
    });
});
