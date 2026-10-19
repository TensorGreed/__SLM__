/**
 * One selected run across the Results tabs (picked-data provenance): the
 * default is the summary's run — the latest completed non-baseline run —
 * never the auto lift check's baseline, and every Verdict card names it.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, stub } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    stub: (id: string) => ({ default: () => <div data-testid={id} /> }),
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./EvalSummaryCard', () => ({
    default: ({ experimentId, onResolved }: { experimentId: number | null; onResolved?: (id: number) => void }) => (
        <div data-testid="summary-card">
            <span data-testid="summary-run">{String(experimentId)}</span>
            <button type="button" onClick={() => onResolved?.(1)}>resolve-to-1</button>
        </div>
    ),
}));
vi.mock('./ScorecardPanel', () => ({
    default: ({ experimentId, runLabel }: { experimentId: number; runLabel?: string }) => (
        <div data-testid="scorecard">{experimentId}|{runLabel}</div>
    ),
}));
vi.mock('./ExternalBenchmarksPanel', () => stub('benchmarks-panel'));
vi.mock('./EvalGapsPanel', () => stub('eval-gaps'));
vi.mock('./ProbePackPanel', () => stub('probe-pack'));
vi.mock('./GoldSetWorkbenchPanel', () => stub('workbench'));
vi.mock('../data/AnswerKeyReviewPanel', () => stub('answer-key-review'));
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
import { defaultRunId, runLabel } from './runSelection';
import { useProjectStore } from '../../stores/projectStore';

// The bug's shape: trained run #1, then the lift check's baseline #2.
const RUNS = [
    { id: 2, name: 'Baseline · SmolLM2-135M-Instruct', status: 'completed', config: { is_baseline: true, source: 'post_training.auto_lift' } },
    { id: 1, name: 'v09-narrated-run', status: 'completed', config: {} },
];

function installApi() {
    apiMock.get.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.endsWith('/training/experiments')) return { data: RUNS };
        if (/\/evaluation\/results\/\d+$/.test(url)) return { data: [] };
        if (url.includes('/evaluation/safety-scorecard/')) return { data: { overall_risk: 'unknown', red_flags: [] } };
        const gate = url.match(/\/evaluation\/gates\/(\d+)/);
        if (gate) {
            const failing = gate[1] === '1' ? ['min_f1'] : ['min_exact_match'];
            return {
                data: {
                    captured_at: '2026-10-19T00:00:00Z', passed: false, checks: [], missing_required_metrics: [],
                    failed_gate_ids: failing, pack: { pack_id: 'evalpack.general.default' },
                },
            };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) return { data: { preferred_pack_id: null, active_pack_id: null } };
        if (url.includes('/remediation')) return { data: { plans: [] } };
        return { data: {} };
    });
}

function requested(fragment: string): boolean {
    return apiMock.get.mock.calls.some(([url]) => String(url).includes(fragment));
}

describe('EvalPanel — one selected run', () => {
    beforeEach(() => {
        useProjectStore.setState({ activeProject: { id: 1, beginner_mode: true } as never });
        installApi();
    });

    it('Verdict opens on the trained run, not the newer baseline, and every card names it', async () => {
        render(<MemoryRouter><EvalPanel projectId={1} workflow="verdict" /></MemoryRouter>);
        const gate = await screen.findByTestId('eval-verdict-gate');
        expect(within(gate).getByTestId('eval-verdict-gate-run')).toHaveTextContent('Run #1 · v09-narrated-run');
        expect(gate).toHaveTextContent('Failed required rules: min_f1');
        expect(gate).toHaveTextContent('evalpack.general.default');
        expect(screen.getByTestId('summary-run')).toHaveTextContent('1');
        expect(screen.getByTestId('scorecard')).toHaveTextContent('1|Run #1 · v09-narrated-run');
        expect(requested('/evaluation/gates/2')).toBe(false);
        // The baseline stays pickable, labelled as such.
        expect(screen.getByTestId('eval-run-2')).toHaveTextContent('#2 · Baseline · SmolLM2-135M-Instruct');
        expect(within(screen.getByTestId('eval-run-2')).getByText('baseline')).toBeInTheDocument();
        expect(screen.getByTestId('eval-run-1')).toHaveClass('btn-primary');
        // No "Compare to #2" — the baseline is not a prior run.
        expect(screen.queryByTestId('eval-panel-compare-to-prior')).toBeNull();
    });

    it('a picked run sticks: the summary card resolving its default does not override it', async () => {
        const user = userEvent.setup();
        render(<MemoryRouter><EvalPanel projectId={1} workflow="verdict" /></MemoryRouter>);
        await screen.findByTestId('eval-verdict-gate');
        await user.click(screen.getByTestId('eval-run-2'));
        await waitFor(() => expect(screen.getByTestId('eval-verdict-gate-run')).toHaveTextContent('Run #2 · Baseline · SmolLM2-135M-Instruct · baseline'));
        await user.click(screen.getByRole('button', { name: 'resolve-to-1' }));
        expect(screen.getByTestId('summary-run')).toHaveTextContent('2');
        expect(screen.getByTestId('scorecard')).toHaveTextContent('2|Run #2');
    });

    it('Failures (no summary card mounted) opens on the same run', async () => {
        render(<MemoryRouter><EvalPanel projectId={1} workflow="failures" /></MemoryRouter>);
        await waitFor(() => expect(requested('/evaluation/results/1')).toBe(true));
        expect(screen.getByTestId('eval-run-1')).toHaveClass('btn-primary');
        expect(requested('/evaluation/results/2')).toBe(false);
        expect(screen.queryByTestId('eval-verdict-gate')).toBeNull();
    });
});

describe('defaultRunId / runLabel', () => {
    it('picks the newest completed run that is neither a baseline nor a seed child', () => {
        expect(defaultRunId(RUNS)).toBe(1);
        expect(defaultRunId([
            { id: 5, name: 'seed-1', status: 'completed', seed_group_id: 'g', seed_value: 1 },
            { id: 4, name: 'leader', status: 'completed', seed_group_id: 'g', seed_value: null },
            { id: 3, name: 'older', status: 'completed' },
        ])).toBe(4);
        expect(defaultRunId([
            { id: 7, name: 'running', status: 'running' },
            { id: 6, name: 'done', status: 'completed' },
        ])).toBe(6);
        expect(defaultRunId([{ id: 8, name: 'still running', status: 'running' }])).toBe(8);
        expect(defaultRunId([RUNS[0]])).toBeNull();
    });

    it('labels a run with its id and name, and says baseline', () => {
        expect(runLabel(RUNS, 1)).toBe('Run #1 · v09-narrated-run');
        expect(runLabel(RUNS, 2)).toBe('Run #2 · Baseline · SmolLM2-135M-Instruct · baseline');
        expect(runLabel(RUNS, 9)).toBe('Run #9');
        expect(runLabel(RUNS, null)).toBe('');
    });
});
