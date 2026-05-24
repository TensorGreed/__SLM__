import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioOverviewPanel from './DataStudioOverviewPanel';

const readyOverview = {
    project_id: 1,
    verdict: 'ready',
    recipe: {
        id: 'classification',
        name: 'Classification',
        task_profile: 'classification',
    },
    domain: {
        profile_id: 'support-domain-v1',
        profile_source: 'project',
        pack_id: 'support-pack-v1',
        pack_source: 'project',
        display_name: 'Support FAQ',
    },
    row_counts: {
        trainable: 184,
        raw: 0,
        cleaned: 120,
        gold: 60,
        synthetic_total: 4,
        synthetic_pending: 0,
        synthetic_accepted: 4,
        prepared: 184,
        train: 148,
        validation: 18,
        test: 18,
    },
    source_summary: {
        dataset_count: 4,
        documents_total: 1,
        documents_accepted: 1,
        documents_processing: 0,
        documents_pending: 0,
        documents_error: 0,
    },
    issues: [],
    primary_action: {
        label: 'Open training',
        target_tab: 'training',
        reason: 'A prepared dataset is available.',
    },
};

describe('DataStudioOverviewPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders overview metrics and routes the primary action', async () => {
        apiMock.get.mockResolvedValueOnce({ data: readyOverview });
        const onOpenTab = vi.fn();

        render(<DataStudioOverviewPanel projectId={1} onOpenTab={onOpenTab} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-overview')).toBeInTheDocument();
        });

        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.getByText('184')).toBeInTheDocument();
        expect(screen.getByText('Classification')).toBeInTheDocument();
        expect(screen.getByText('Support FAQ')).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Open training' }));
        expect(onOpenTab).toHaveBeenCalledWith('training');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/overview');
    });

    it('surfaces blockers from the overview endpoint', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...readyOverview,
                verdict: 'blocked',
                recipe: null,
                row_counts: {
                    ...readyOverview.row_counts,
                    trainable: 0,
                    prepared: 0,
                },
                issues: [
                    {
                        id: 'missing_recipe',
                        severity: 'blocker',
                        title: 'Recipe not selected',
                        message: 'Pick a task recipe so BrewSLM knows the training shape.',
                        action_label: 'Choose recipe',
                        target_tab: 'data',
                    },
                ],
                primary_action: {
                    label: 'Choose recipe',
                    target_tab: 'data',
                    reason: 'Recipe not selected',
                },
            },
        });

        render(<DataStudioOverviewPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getAllByText('Recipe not selected').length).toBeGreaterThan(0);
        });
        expect(screen.getByText('Blocked')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Choose recipe' })).toBeInTheDocument();
    });
});
