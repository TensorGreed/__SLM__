import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { TimelineFilters } from '../../types/observability';

import RunTimelineFilters from './RunTimelineFilters';

function defaultFilters(): TimelineFilters {
    return {
        stage: '',
        severity: '',
        run_id: '',
        since: '',
        until: '',
        limit: 500,
    };
}

describe('RunTimelineFilters', () => {
    it('emits onChange when stage selection changes', async () => {
        const onChange = vi.fn();
        render(
            <RunTimelineFilters
                value={defaultFilters()}
                onChange={onChange}
                onRefresh={vi.fn()}
            />,
        );
        const user = userEvent.setup();
        await user.selectOptions(
            screen.getByLabelText(/Filter by stage/i),
            'training',
        );
        expect(onChange).toHaveBeenLastCalledWith(
            expect.objectContaining({ stage: 'training' }),
        );
    });

    it('emits onChange when severity selection changes', async () => {
        const onChange = vi.fn();
        render(
            <RunTimelineFilters
                value={defaultFilters()}
                onChange={onChange}
                onRefresh={vi.fn()}
            />,
        );
        const user = userEvent.setup();
        await user.selectOptions(
            screen.getByLabelText(/Filter by severity/i),
            'error',
        );
        expect(onChange).toHaveBeenLastCalledWith(
            expect.objectContaining({ severity: 'error' }),
        );
    });

    it('emits onChange when run_id text input changes', async () => {
        const onChange = vi.fn();
        render(
            <RunTimelineFilters
                value={defaultFilters()}
                onChange={onChange}
                onRefresh={vi.fn()}
            />,
        );
        const user = userEvent.setup();
        await user.type(screen.getByLabelText(/Anchor on run id/i), 'a');
        // Each keystroke fires one onChange.
        expect(onChange).toHaveBeenCalled();
        expect(onChange.mock.calls[0][0].run_id).toBe('a');
    });

    it('refresh button calls onRefresh and is disabled while loading', async () => {
        const onRefresh = vi.fn();
        const { rerender } = render(
            <RunTimelineFilters
                value={defaultFilters()}
                onChange={vi.fn()}
                onRefresh={onRefresh}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Refresh/i }));
        expect(onRefresh).toHaveBeenCalledTimes(1);

        rerender(
            <RunTimelineFilters
                value={defaultFilters()}
                onChange={vi.fn()}
                onRefresh={onRefresh}
                loading
            />,
        );
        expect(
            screen.getByRole('button', { name: /Loading…/i }),
        ).toBeDisabled();
    });

    it('shows truncated badge when truncated=true', () => {
        render(
            <RunTimelineFilters
                value={defaultFilters()}
                onChange={vi.fn()}
                onRefresh={vi.fn()}
                truncated
            />,
        );
        expect(screen.getByText(/truncated/i)).toBeInTheDocument();
    });
});
