import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import GuidedLearningRail, { guidedTipDismissKey } from './GuidedLearningRail';
import type { Project } from '../../types';

function makeProject(overrides: Partial<Project> = {}): Project {
    return {
        id: 7,
        name: 'Demo',
        description: null,
        status: 'active' as Project['status'],
        pipeline_stage: 'ingestion',
        base_model_name: null,
        domain_pack_id: null,
        domain_profile_id: null,
        beginner_mode: true,
        created_at: '2026-06-20T00:00:00Z',
        updated_at: '2026-06-20T00:00:00Z',
        ...overrides,
    };
}

describe('GuidedLearningRail', () => {
    beforeEach(() => {
        window.localStorage.clear();
    });
    afterEach(() => {
        window.localStorage.clear();
    });

    it('renders nothing when the project is not in beginner mode', () => {
        const { container } = render(
            <GuidedLearningRail
                project={makeProject({ beginner_mode: false })}
                activeTab="data"
                onNextStep={() => {}}
            />,
        );
        expect(container.querySelector('.guided-rail')).toBeNull();
    });

    it('renders nothing when there is no active project', () => {
        const { container } = render(
            <GuidedLearningRail project={null} activeTab="data" onNextStep={() => {}} />,
        );
        expect(container.querySelector('.guided-rail')).toBeNull();
    });

    it('shows the checkpoint step line and a next-step CTA in beginner mode', async () => {
        const onNext = vi.fn();
        render(
            <GuidedLearningRail project={makeProject()} activeTab="data" onNextStep={onNext} />,
        );
        // data is tab 1 of 10; next tab is Cleaning.
        expect(screen.getByText(/Step 1 of 10/i)).toBeInTheDocument();
        const next = screen.getByRole('button', { name: /Next: Cleaning/i });
        const user = userEvent.setup();
        await user.click(next);
        expect(onNext).toHaveBeenCalledTimes(1);
    });

    it('shows a first-visit tip and persists dismissal to localStorage', async () => {
        render(
            <GuidedLearningRail project={makeProject()} activeTab="goldset" onNextStep={() => {}} />,
        );
        expect(await screen.findByText(/trusted answer key/i)).toBeInTheDocument();
        const dismiss = screen.getByRole('button', { name: /Dismiss the Gold Set tip/i });
        const user = userEvent.setup();
        await user.click(dismiss);
        expect(window.localStorage.getItem(guidedTipDismissKey(7, 'goldset'))).toBe('1');
        expect(screen.queryByText(/trusted answer key/i)).not.toBeInTheDocument();
    });

    it('hides the tip when its dismissed flag is already set', () => {
        window.localStorage.setItem(guidedTipDismissKey(7, 'goldset'), '1');
        render(
            <GuidedLearningRail project={makeProject()} activeTab="goldset" onNextStep={() => {}} />,
        );
        // The rail still renders (step line), but the tip is suppressed.
        expect(screen.getByText(/Step 3 of 10/i)).toBeInTheDocument();
        expect(screen.queryByText(/trusted answer key/i)).not.toBeInTheDocument();
    });

    it('marks the export tab as the final step with no next CTA', () => {
        render(
            <GuidedLearningRail project={makeProject()} activeTab="export" onNextStep={() => {}} />,
        );
        expect(screen.getByText(/Step 10 of 10/i)).toBeInTheDocument();
        expect(screen.getByText(/Final step/i)).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /^Next:/i })).not.toBeInTheDocument();
    });
});
