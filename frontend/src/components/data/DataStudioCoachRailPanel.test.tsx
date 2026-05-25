import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioCoachRailPanel from './DataStudioCoachRailPanel';

const coachPayload = {
    project_id: 1,
    verdict: 'blocked',
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_data_studio_checks',
    summary: {
        blocker_count: 2,
        warning_count: 1,
        info_count: 1,
        section_count: 10,
        ready_section_count: 2,
        empty_section_count: 1,
        next_action_target: 'data',
    },
    next_action: {
        id: 'overview:missing_recipe',
        section_id: 'overview',
        section_label: 'Overview',
        severity: 'blocker',
        priority: 'high',
        title: 'Recipe not selected',
        message: 'Pick a task recipe so BrewSLM knows the training shape.',
        action_label: 'Choose recipe',
        target_tab: 'data',
        requires_user_confirmation: true,
    },
    next_steps: [
        {
            id: 'overview:missing_recipe',
            section_id: 'overview',
            section_label: 'Overview',
            severity: 'blocker',
            priority: 'high',
            title: 'Recipe not selected',
            message: 'Pick a task recipe so BrewSLM knows the training shape.',
            action_label: 'Choose recipe',
            target_tab: 'data',
            requires_user_confirmation: true,
        },
        {
            id: 'mapping:no_mapping_source',
            section_id: 'mapping',
            section_label: 'Mapping',
            severity: 'blocker',
            priority: 'high',
            title: 'No previewable rows',
            message: 'Add an accepted raw document or row-backed dataset.',
            action_label: 'Add sources',
            target_tab: 'data',
            requires_user_confirmation: true,
        },
    ],
    checks: [
        {
            id: 'overview',
            label: 'Overview',
            status: 'blocked',
            verdict: 'blocked',
            target_tab: 'data',
            action_label: 'Open Data',
            message: 'Recipe not selected',
            blocker_count: 1,
            warning_count: 0,
            info_count: 0,
        },
        {
            id: 'mapping',
            label: 'Mapping',
            status: 'blocked',
            verdict: 'empty',
            target_tab: 'dataprep',
            action_label: 'Review Mapping',
            message: 'No previewable rows',
            blocker_count: 1,
            warning_count: 0,
            info_count: 0,
        },
        {
            id: 'dataset_versions',
            label: 'Dataset Versions',
            status: 'empty',
            verdict: 'empty',
            target_tab: 'dataprep',
            action_label: 'Open Versions',
            message: 'Prepared dataset versions are not available yet.',
            blocker_count: 0,
            warning_count: 0,
            info_count: 1,
        },
    ],
    issues: [],
    entry_points: [
        {
            label: 'Open Data Prep',
            target_tab: 'dataprep',
            reason: 'Review mapping, split, manifest, and version checks.',
            requires_confirmation: true,
        },
    ],
    power_details: {},
};

describe('DataStudioCoachRailPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders a beginner next action and compact power checks', async () => {
        apiMock.get.mockResolvedValueOnce({ data: coachPayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioCoachRailPanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-coach-rail')).toBeInTheDocument();
        });

        expect(screen.getByText('Blocked')).toBeInTheDocument();
        expect(screen.getAllByText('Recipe not selected').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Mapping').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Dataset Versions').length).toBeGreaterThan(0);
        expect(screen.getByText('10')).toBeInTheDocument();

        fireEvent.click(screen.getAllByRole('button', { name: /Choose recipe/i })[0]);
        expect(onOpenTarget).toHaveBeenCalledWith('data', 'overview');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/coach');
    });

    it('routes clean projects toward training without mutating', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...coachPayload,
                verdict: 'ready',
                summary: {
                    blocker_count: 0,
                    warning_count: 0,
                    info_count: 0,
                    section_count: 10,
                    ready_section_count: 10,
                    empty_section_count: 0,
                    next_action_target: 'training',
                },
                next_action: {
                    id: 'coach_open_training',
                    section_id: 'dataset_versions',
                    section_label: 'Dataset Versions',
                    severity: 'info',
                    priority: 'medium',
                    title: 'Launch training from the prepared dataset',
                    message: 'Prepared versions are ready for training.',
                    action_label: 'Open Training',
                    target_tab: 'training',
                    requires_user_confirmation: true,
                },
                next_steps: [
                    {
                        id: 'coach_open_training',
                        section_id: 'dataset_versions',
                        section_label: 'Dataset Versions',
                        severity: 'info',
                        priority: 'medium',
                        title: 'Launch training from the prepared dataset',
                        message: 'Prepared versions are ready for training.',
                        action_label: 'Open Training',
                        target_tab: 'training',
                        requires_user_confirmation: true,
                    },
                ],
                checks: coachPayload.checks.map((check) => ({
                    ...check,
                    status: 'ready',
                    verdict: 'ready',
                    blocker_count: 0,
                    warning_count: 0,
                    info_count: 0,
                    message: `${check.label} is ready.`,
                })),
            },
        });
        const onOpenTarget = vi.fn();

        render(<DataStudioCoachRailPanel projectId={2} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-coach-rail')).toBeInTheDocument();
        });

        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.getAllByText('Launch training from the prepared dataset').length).toBeGreaterThan(0);

        fireEvent.click(screen.getAllByRole('button', { name: /Open Training/i })[0]);
        expect(onOpenTarget).toHaveBeenCalledWith('training', 'dataset_versions');
    });
});
