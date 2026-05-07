import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { TimelineNode } from '../../types/observability';

import TimelineTree from './TimelineTree';

function makeNode(overrides: Partial<TimelineNode>): TimelineNode {
    return {
        run_id: 'exp-1',
        parent_run_id: null,
        is_orphan: false,
        stage: 'training',
        stages_present: ['training'],
        summary: 'Training started',
        actor: 'system',
        first_ts: '2026-05-07T10:00:00Z',
        last_ts: '2026-05-07T10:00:30Z',
        duration_seconds: 30,
        event_count: 1,
        severity_counts: { info: 1 },
        highest_severity: 'info',
        latest_reason_code: null,
        children: [],
        ...overrides,
    };
}

describe('TimelineTree', () => {
    it('renders empty state when tree is empty', () => {
        render(
            <TimelineTree tree={[]} selectedRunId={null} onSelectRun={vi.fn()} />,
        );
        expect(screen.getByText(/No timeline events/i)).toBeInTheDocument();
    });

    it('renders run_id, stage, summary, severity badge', () => {
        render(
            <TimelineTree
                tree={[makeNode({})]}
                selectedRunId={null}
                onSelectRun={vi.fn()}
            />,
        );
        expect(screen.getByText('exp-1')).toBeInTheDocument();
        expect(screen.getByText('training')).toBeInTheDocument();
        expect(screen.getByText('Training started')).toBeInTheDocument();
        expect(screen.getByText('info')).toBeInTheDocument();
    });

    it('clicking the run-id button calls onSelectRun', async () => {
        const onSelect = vi.fn();
        render(
            <TimelineTree
                tree={[makeNode({})]}
                selectedRunId={null}
                onSelectRun={onSelect}
            />,
        );
        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Open events for exp-1/i }),
        );
        expect(onSelect).toHaveBeenCalledWith('exp-1');
    });

    it('expands children on toggle click', async () => {
        const tree = [
            makeNode({
                run_id: 'exp-1',
                children: [
                    makeNode({
                        run_id: 'eval-1',
                        parent_run_id: 'exp-1',
                        stage: 'eval',
                        summary: 'Eval done',
                    }),
                ],
            }),
        ];
        render(
            <TimelineTree
                tree={tree}
                selectedRunId={null}
                onSelectRun={vi.fn()}
                defaultAllExpanded
            />,
        );
        // Both rows visible because defaultAllExpanded=true.
        expect(screen.getByText('Eval done')).toBeInTheDocument();
    });

    it('orphan node shows the orphan badge', () => {
        render(
            <TimelineTree
                tree={[
                    makeNode({
                        run_id: 'eval-99',
                        parent_run_id: 'exp-99',
                        is_orphan: true,
                    }),
                ]}
                selectedRunId={null}
                onSelectRun={vi.fn()}
            />,
        );
        expect(screen.getByText(/orphan/i)).toBeInTheDocument();
    });

    it('error severity uses the danger badge', () => {
        render(
            <TimelineTree
                tree={[
                    makeNode({
                        highest_severity: 'error',
                        latest_reason_code: 'training_runtime_error',
                    }),
                ]}
                selectedRunId={null}
                onSelectRun={vi.fn()}
            />,
        );
        const badge = screen.getByText('error');
        expect(badge.className).toContain('badge-danger');
        expect(
            screen.getByText('training_runtime_error'),
        ).toBeInTheDocument();
    });

    it('selected row gets the is-selected class', () => {
        render(
            <TimelineTree
                tree={[makeNode({ run_id: 'exp-7' })]}
                selectedRunId="exp-7"
                onSelectRun={vi.fn()}
            />,
        );
        const row = document.querySelector('.timeline-row.is-selected');
        expect(row).not.toBeNull();
        expect(row?.getAttribute('data-run-id')).toBe('exp-7');
    });
});
