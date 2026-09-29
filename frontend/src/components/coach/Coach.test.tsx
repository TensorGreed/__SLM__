import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../video/TabVideoLink', () => ({
    default: ({ tabKey }: { tabKey: string }) => <div data-testid={`video-${tabKey}`} />,
}));

import Coach from './Coach';
import { coachLocationFor } from './coachContext';
import { _resetCoachModeStoreForTests, useCoachModeStore } from '../../stores/coachModeStore';
import type { PipelineStatusResponse, Project } from '../../types';

const PROJECT = {
    id: 7,
    name: 'P',
    description: null,
    status: 'active',
    pipeline_stage: 'ingestion',
    base_model_name: null,
    domain_pack_id: null,
    domain_profile_id: null,
    beginner_mode: true,
} as unknown as Project;

const STATUS = {
    project_id: 7,
    current_stage: 'ingestion',
    progress_percent: 0,
    stages: [],
} as unknown as PipelineStatusResponse;

function renderAt(path: string, project: Project = PROJECT) {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Coach projectId={7} project={project} pipelineStatus={STATUS} />
        </MemoryRouter>,
    );
}

describe('Coach (the one guidance surface)', () => {
    beforeEach(() => {
        window.localStorage.clear();
        _resetCoachModeStoreForTests();
        apiMock.get.mockReset();
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/gamification')) return { data: { level: 9 } }; // "senior" user
            if (url.endsWith('/coach/data')) {
                return {
                    data: {
                        project_id: 7,
                        stage: 'data',
                        handler_available: true,
                        suggestions: [
                            {
                                id: 's1', title: 'Only 12 rows', body: 'Add more examples.', severity: 'warning',
                                action: { kind: 'navigate', label: 'Open synthetic', params: { target: 'synthetic-review-queue' } },
                            },
                        ],
                    },
                };
            }
            return { data: {} };
        });
    });

    it('shows where you are, the next step, a tab tip, the video and stage suggestions', async () => {
        renderAt('/project/7/pipeline/data');
        expect(screen.getByTestId('coach-where').textContent).toContain('0% done');
        // Beginner projects never get sent to (hidden) domain packs first.
        expect(screen.getByTestId('coach-next').textContent).toContain('Import source data');
        // Already on the Data tab → no "Continue" button.
        expect(screen.queryByTestId('coach-continue')).toBeNull();
        expect(screen.getByTestId('coach-tip')).toBeInTheDocument();
        expect(screen.getByTestId('video-data')).toBeInTheDocument();
        expect(await screen.findByText('Only 12 rows')).toBeInTheDocument();
    });

    it('is on by default for beginner projects even for a high-level user', () => {
        renderAt('/project/7/pipeline/data');
        expect(screen.getByTestId('coach-bar')).toBeInTheDocument();
    });

    it('offers Continue off-target and remembers a dismissed tip', async () => {
        const user = userEvent.setup();
        const { unmount } = renderAt('/project/7/pipeline/cleaning');
        expect(screen.getByTestId('coach-continue')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Dismiss tip' }));
        expect(screen.queryByTestId('coach-tip')).toBeNull();
        unmount();
        renderAt('/project/7/pipeline/cleaning');
        expect(screen.queryByTestId('coach-tip')).toBeNull();
    });

    it('stays out of the way on the plan page and when switched off', async () => {
        renderAt('/project/7/guide');
        expect(screen.queryByTestId('coach-bar')).toBeNull();
        useCoachModeStore.getState().setOverride(7, 'off');
        renderAt('/project/7/pipeline/data');
        await waitFor(() => expect(screen.queryByTestId('coach-bar')).toBeNull());
    });

    it('maps workspace URLs to the tab + coach stage', () => {
        expect(coachLocationFor('/project/1/pipeline/goldset')).toEqual({ tab: 'goldset', stage: 'gold_set' });
        expect(coachLocationFor('/project/1/pipeline/dataprep')).toEqual({ tab: 'dataprep', stage: null });
        expect(coachLocationFor('/project/1/training-config')).toEqual({ tab: 'training', stage: 'training' });
        expect(coachLocationFor('/project/1/playground')).toEqual({ tab: null, stage: null });
    });
});
