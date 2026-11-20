/**
 * Results workflows (Epic U phase U3): the one-page panel splits by
 * ``workflow`` — each tab shows its blocks and none of the others'.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, stub } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    stub: (id: string) => ({ default: () => <div data-testid={id} /> }),
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./EvalSummaryCard', () => ({
    default: ({ hideTryBox, verdictShowsLift }: { hideTryBox?: boolean; verdictShowsLift?: boolean }) => (
        <div data-testid="summary-card" data-verdict-shows-lift={verdictShowsLift ? 'yes' : 'no'}>{hideTryBox ? 'no-try-box' : 'try-box'}</div>
    ),
}));
vi.mock('./ExternalBenchmarksPanel', () => stub('benchmarks-panel'));
vi.mock('./EvalGapsPanel', () => stub('eval-gaps'));
vi.mock('./ProbePackPanel', () => ({
    default: ({ onOpenRun }: { onOpenRun?: (id: number) => void }) => (
        <button type="button" data-testid="probe-pack" onClick={() => onOpenRun?.(21)}>open run</button>
    ),
}));
vi.mock('./GoldSetWorkbenchPanel', () => stub('workbench'));
vi.mock('../data/AnswerKeyReviewPanel', () => stub('answer-key-review'));
vi.mock('./ScorecardPanel', () => stub('scorecard'));
vi.mock('./FailureClustersPanel', () => stub('failure-clusters'));
vi.mock('./ActiveLearningPanel', () => stub('active-learning'));
vi.mock('./SftLiftPanel', () => stub('sft-lift'));
vi.mock('./StudentTeacherComparisonPanel', () => stub('student-teacher'));
vi.mock('./FrontierComparisonPanel', () => stub('frontier'));
vi.mock('./RerouteRecommendationPanel', () => stub('reroute'));
vi.mock('./AutoRagComparisonPanel', () => stub('auto-rag'));
vi.mock('./DriftReviewQueuePanel', () => stub('drift-queue'));
vi.mock('./RedactionTryBox', () => stub('redact-box'));
vi.mock('./ClassificationChartsPanel', () => stub('classification-charts'));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';

function installApi(scoringMode: string | null = 'span_set') {
    apiMock.get.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.endsWith('/training/experiments')) return { data: [{ id: 21, name: 'exp-21' }] };
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [{ id: 1001, dataset_name: 'gold_test', eval_type: 'f1', pass_rate: 0.5, metrics: { scoring_mode: scoringMode }, details: { predictions_preview: [] } }],
            };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return { data: { experiment_id: 21, decision: 'NO-SHIP', reasons: [], failed_gates: [], missing_metrics: [], gate_report: { passed: false, checks: [], missing_required_metrics: [], failed_gate_ids: [], captured_at: '2026-10-17T00:00:00Z' } } };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/gates/21')) {
            return { data: { captured_at: '2026-10-17T00:00:00Z', passed: false, checks: [], missing_required_metrics: [], failed_gate_ids: [] } };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) return { data: { preferred_pack_id: null, active_pack_id: null } };
        if (url.includes('/remediation')) return { data: { plans: [] } };
        return { data: {} };
    });
}

async function renderWorkflow(workflow: 'verdict' | 'failures' | 'compare' | 'try') {
    render(<MemoryRouter><EvalPanel projectId={1} workflow={workflow} /></MemoryRouter>);
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: /exp-21/ }));
    await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(expect.stringContaining('/evaluation/results/21')));
}

describe('EvalPanel — Results workflows', () => {
    beforeEach(() => {
        window.localStorage.clear();
        useProjectStore.setState({ activeProject: { id: 1, beginner_mode: true } as never });
        installApi();
    });

    it('Verdict: the Verdict card, summary card (without the try box), Scores, How it\'s scored; lift and scorecard under Details; no failures or comparisons', async () => {
        await renderWorkflow('verdict');
        expect(screen.getByTestId('verdict-card')).toBeInTheDocument();
        expect(screen.getByTestId('summary-card')).toHaveTextContent('no-try-box');
        expect(screen.getByText('Evaluate')).toBeInTheDocument();
        expect(await screen.findByTestId('how-scored')).toBeInTheDocument();
        // Scores: collapsed, kept mounted (the per-result panels hold their own state).
        expect(screen.getByTestId('advanced-verdict-scores-toggle')).toHaveTextContent('Scores (1)');
        expect(screen.getByTestId('advanced-verdict-scores-body')).toHaveAttribute('hidden');
        expect(within(screen.getByTestId('advanced-verdict-scores-body')).getByText('Metric History')).toBeInTheDocument();
        // Details: the full rule report, base vs fine-tuned and the scorecard, one click away.
        expect(screen.queryByTestId('scorecard')).toBeNull();
        await userEvent.setup().click(screen.getByTestId('advanced-verdict-details-toggle'));
        expect(await screen.findByTestId('scorecard')).toBeInTheDocument();
        expect(screen.getByTestId('sft-lift')).toBeInTheDocument();
        for (const id of ['benchmarks-panel', 'eval-gaps', 'probe-pack', 'failure-clusters', 'drift-queue', 'auto-rag', 'frontier', 'redact-box', 'answer-key-review']) {
            expect(screen.queryByTestId(id)).toBeNull();
        }
        expect(screen.queryByTestId('eval-advanced-toggle')).toBeNull();
        expect(screen.queryByRole('tab', { name: 'Workbench' })).toBeNull();
    });

    it('Failures: clusters and checks in view; every expert panel still mounts under the collapsed Advanced section', async () => {
        await renderWorkflow('failures');
        for (const id of ['eval-gaps', 'failure-clusters']) {
            expect(await screen.findByTestId(id)).toBeInTheDocument();
        }
        const body = screen.getByTestId('advanced-failures-tools-body');
        expect(body).toHaveAttribute('hidden');
        expect(screen.getByTestId('advanced-failures-tools-toggle')).toHaveTextContent('Advanced (10)');
        for (const id of ['probe-pack', 'active-learning', 'reroute', 'drift-queue', 'eval-answer-key-review', 'eval-workbench']) {
            expect(await within(body).findByTestId(id)).toBeInTheDocument();
        }
        expect(within(body).getByText('Pass/fail rules')).toBeInTheDocument();
        expect(within(body).getByText('Closed-Loop Remediation Planner')).toBeInTheDocument();
        expect(within(body).getByText('Skill Radar (LLM Judge)')).toBeInTheDocument();
        await userEvent.setup().click(screen.getByTestId('advanced-failures-tools-toggle'));
        expect(body).not.toHaveAttribute('hidden');
        // The score panels moved to Verdict › Scores.
        expect(screen.queryByText('Metric History')).toBeNull();
        for (const id of ['summary-card', 'verdict-card', 'scorecard', 'sft-lift', 'benchmarks-panel', 'auto-rag', 'frontier']) {
            expect(screen.queryByTestId(id)).toBeNull();
        }
    });

    it('the summary card is told the Verdict card shows base vs fine-tuned lines (no second headline)', async () => {
        await renderWorkflow('verdict');
        await waitFor(() => expect(screen.getByTestId('summary-card')).toHaveAttribute('data-verdict-shows-lift', 'yes'));
    });

    it('a failed gate-report fetch reads "could not be loaded" on the Verdict card', async () => {
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string, ...rest: unknown[]) => {
            if (url.includes('/evaluation/gates/21')) throw new Error('Network Error');
            return base(url, ...rest);
        });
        await renderWorkflow('verdict');
        await waitFor(() => expect(screen.getByTestId('verdict-rules-status')).toHaveTextContent('could not be loaded'));
    });

    it('the probe pack\'s "open run" goes to Verdict › Details (where the scorecard is), not a dead anchor', async () => {
        function Where() {
            const loc = useLocation();
            return <div data-testid="where">{loc.pathname + loc.search}</div>;
        }
        render(<MemoryRouter><EvalPanel projectId={1} workflow="failures" /><Where /></MemoryRouter>);
        await userEvent.setup().click(await screen.findByRole('button', { name: /exp-21/ }));
        await userEvent.setup().click(await screen.findByTestId('probe-pack'));
        await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/project/1/results?open=verdict-details&tab=verdict'));
    });

    it('?open= opens the targeted Advanced section (the palette path)', async () => {
        render(<MemoryRouter><EvalPanel projectId={1} workflow="failures" openSection="failures-tools" /></MemoryRouter>);
        await userEvent.setup().click(await screen.findByRole('button', { name: /exp-21/ }));
        expect(await screen.findByTestId('advanced-failures-tools-body')).not.toHaveAttribute('hidden');
    });

    it('Compare: benchmarks, frontier, RAG and student/teacher only', async () => {
        await renderWorkflow('compare');
        for (const id of ['benchmarks-panel', 'frontier', 'auto-rag', 'student-teacher']) {
            expect(await screen.findByTestId(id)).toBeInTheDocument();
        }
        for (const id of ['summary-card', 'scorecard', 'failure-clusters', 'eval-gaps', 'drift-queue']) {
            expect(screen.queryByTestId(id)).toBeNull();
        }
    });

    it('Try it: the redaction box for a span run, a note otherwise', async () => {
        await renderWorkflow('try');
        expect(await screen.findByTestId('redact-box')).toBeInTheDocument();
        expect(screen.queryByTestId('summary-card')).toBeNull();
        installApi('field_match');
        render(<MemoryRouter><EvalPanel projectId={2} workflow="try" /></MemoryRouter>);
        const user = userEvent.setup();
        await user.click((await screen.findAllByRole('button', { name: /exp-21/ })).at(-1)!);
        expect((await screen.findAllByText(/Available for a span-tagging run/)).length).toBeGreaterThan(0);
    });
});
