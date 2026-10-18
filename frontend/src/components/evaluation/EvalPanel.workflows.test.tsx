/**
 * Results workflows (Epic U phase U3): the one-page panel splits by
 * ``workflow`` — each tab shows its blocks and none of the others'.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, stub } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    stub: (id: string) => ({ default: () => <div data-testid={id} /> }),
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./EvalSummaryCard', () => ({ default: ({ hideTryBox }: { hideTryBox?: boolean }) => <div data-testid="summary-card">{hideTryBox ? 'no-try-box' : 'try-box'}</div> }));
vi.mock('./ExternalBenchmarksPanel', () => stub('benchmarks-panel'));
vi.mock('./EvalGapsPanel', () => stub('eval-gaps'));
vi.mock('./ProbePackPanel', () => stub('probe-pack'));
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
    await user.click(await screen.findByRole('button', { name: 'exp-21' }));
    await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(expect.stringContaining('/evaluation/results/21')));
}

describe('EvalPanel — Results workflows', () => {
    beforeEach(() => {
        useProjectStore.setState({ activeProject: { id: 1, beginner_mode: true } as never });
        installApi();
    });

    it('Verdict: summary card (without the try box), lift and scorecard; no failures or comparisons, no Advanced toggle', async () => {
        await renderWorkflow('verdict');
        expect(screen.getByTestId('summary-card')).toHaveTextContent('no-try-box');
        expect(await screen.findByTestId('scorecard')).toBeInTheDocument();
        expect(screen.getByTestId('sft-lift')).toBeInTheDocument();
        expect(screen.getByText('Evaluate')).toBeInTheDocument();
        for (const id of ['benchmarks-panel', 'eval-gaps', 'probe-pack', 'failure-clusters', 'drift-queue', 'auto-rag', 'frontier', 'redact-box', 'answer-key-review']) {
            expect(screen.queryByTestId(id)).toBeNull();
        }
        expect(screen.queryByTestId('eval-advanced-toggle')).toBeNull();
        expect(screen.queryByRole('tab', { name: 'Workbench' })).toBeNull();
    });

    it('Failures: checks, clusters, reroute, drift, active learning, the rule set and the collapsed answer-key sections', async () => {
        await renderWorkflow('failures');
        for (const id of ['eval-gaps', 'probe-pack', 'failure-clusters', 'active-learning', 'reroute', 'drift-queue', 'eval-answer-key-review', 'eval-workbench']) {
            expect(await screen.findByTestId(id)).toBeInTheDocument();
        }
        expect(screen.getByText('Pass/fail rules')).toBeInTheDocument();
        expect(screen.getByText('Closed-Loop Remediation Planner')).toBeInTheDocument();
        for (const id of ['summary-card', 'scorecard', 'sft-lift', 'benchmarks-panel', 'auto-rag', 'frontier']) {
            expect(screen.queryByTestId(id)).toBeNull();
        }
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
        await user.click((await screen.findAllByRole('button', { name: 'exp-21' })).at(-1)!);
        expect((await screen.findAllByText(/Available for a span-tagging run/)).length).toBeGreaterThan(0);
    });
});
