import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { contextState, evalPanelProps } = vi.hoisted(() => ({
    contextState: { pipelineStatus: null as unknown },
    evalPanelProps: { workflows: [] as string[] },
}));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({ projectId: 31, project: { id: 31 }, pipelineStatus: contextState.pipelineStatus, refreshPipelineStatus: vi.fn() }),
    };
});
vi.mock('../components/evaluation/EvalPanel', () => ({
    default: ({ workflow }: { workflow: string }) => {
        evalPanelProps.workflows.push(workflow);
        return <div data-testid="eval-panel">{workflow}</div>;
    },
}));
vi.mock('../components/training/ChatPlaygroundPanel', () => ({ default: () => <div data-testid="playground" /> }));
vi.mock('./ProjectEvalComparePage', () => ({ default: () => <div data-testid="compare-page" /> }));

import ProjectResultsPage, { resolveResultsTab } from './ProjectResultsPage';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname + location.search}</div>;
}

function renderAt(path: string) {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/project/:id/results" element={<><ProjectResultsPage /><LocationProbe /></>} />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectResultsPage', () => {
    beforeEach(() => {
        evalPanelProps.workflows = [];
        contextState.pipelineStatus = null;
    });

    it('opens on Verdict with four workflow tabs and one evaluation panel', () => {
        renderAt('/project/31/results');
        expect(screen.getAllByRole('tab')).toHaveLength(4);
        expect(screen.getByTestId('results-tab-verdict')).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByTestId('eval-panel')).toHaveTextContent('verdict');
        expect(screen.queryByTestId('playground')).toBeNull();
    });

    it('switches tabs by URL and keeps the compare pair / playground run across tabs', async () => {
        const user = userEvent.setup();
        renderAt('/project/31/results?tab=compare&a=1&b=2');
        expect(screen.getByTestId('compare-page')).toBeInTheDocument();
        expect(screen.getByTestId('eval-panel')).toHaveTextContent('compare');
        await user.click(screen.getByTestId('results-tab-try'));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/31/results?a=1&b=2&tab=try');
        expect(screen.getByTestId('playground')).toBeInTheDocument();
        expect(screen.getByTestId('eval-panel')).toHaveTextContent('try');
    });

    it('Compare without a pair shows a hint instead of the side-by-side page', () => {
        renderAt('/project/31/results?tab=compare');
        expect(screen.queryByTestId('compare-page')).toBeNull();
        expect(screen.getByTestId('results-compare-hint')).toBeInTheDocument();
    });

    it('shows the pass/fail rules card on Verdict only', () => {
        contextState.pipelineStatus = {
            current_stage: 'evaluation',
            auto_gate: { experiment_id: 9, pack_id: 'auto', passed: false, failed_gate_ids: ['f1'], missing_required_metrics: [], captured_at: null },
        };
        renderAt('/project/31/results');
        expect(screen.getByTestId('results-auto-gate')).toHaveTextContent('Failed required rules: f1');
        renderAt('/project/31/results?tab=failures');
        expect(screen.getAllByTestId('results-auto-gate')).toHaveLength(1);
    });

    it('resolveResultsTab maps the old eval anchors onto workflows', () => {
        expect(resolveResultsTab('', '')).toBe('verdict');
        expect(resolveResultsTab('?tab=failures', '')).toBe('failures');
        expect(resolveResultsTab('', '#probe-pack-panel')).toBe('failures');
        expect(resolveResultsTab('', '#external-benchmarks')).toBe('compare');
        expect(resolveResultsTab('?tab=nope', '#experiment-scorecard')).toBe('verdict');
    });
});
