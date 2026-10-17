import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, contextState } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    contextState: {
        project: {
            id: 31, name: 'Tagger', description: null, status: 'active', pipeline_stage: 'evaluation',
            base_model_name: 'microsoft/deberta-v3-base', domain_pack_id: null, domain_profile_id: null, beginner_mode: true,
        } as unknown,
        pipelineStatus: { project_id: 31, current_stage: 'evaluation', progress_percent: 70, stages: [] } as unknown,
    },
}));

vi.mock('../api/client', () => ({ default: apiMock }));
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({
            projectId: 31,
            project: contextState.project,
            pipelineStatus: contextState.pipelineStatus,
            refreshPipelineStatus: vi.fn(),
        }),
    };
});
vi.mock('../components/guide/QuickstartCard', () => ({ default: () => <div data-testid="quickstart" /> }));

import ProjectOverviewPage from './ProjectOverviewPage';
import { latestRunText } from './overviewText';
import { _resetCoachModeStoreForTests } from '../stores/coachModeStore';
import type { EvalSummary } from '../api/evalSummary';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname + location.search}</div>;
}

function renderOverview() {
    return render(
        <MemoryRouter initialEntries={['/project/31']}>
            <Routes>
                <Route path="/project/:id" element={<><ProjectOverviewPage /><LocationProbe /></>} />
                <Route path="*" element={<LocationProbe />} />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectOverviewPage', () => {
    beforeEach(() => {
        _resetCoachModeStoreForTests();
        apiMock.get.mockReset();
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.endsWith('/evaluation/summary')) {
                return {
                    data: {
                        project_id: 31, experiment_id: 143, kind: 'span_eval', verdict: 'evaluated', headline: null, failures: [],
                        answer_key: { eval_result_id: 1, dataset_name: 'gold_test', precision: 0.9, recall: 0.9, f1: 0.778, rows: 44, weakest_types: [] },
                    },
                };
            }
            if (url.endsWith('/export/list')) {
                return {
                    data: [
                        { id: 36, status: 'completed', export_format: 'huggingface', completed_at: '2026-10-17T13:00:00Z', experiment_id: 135 },
                        { id: 37, status: 'completed', export_format: 'huggingface', completed_at: '2026-10-17T15:00:00Z', experiment_id: 143,
                          hub_publish: { repo_id: 'TensorGreed/brewslm-sensitive-data-tagger', url: 'https://huggingface.co/TensorGreed/brewslm-sensitive-data-tagger' } },
                    ],
                };
            }
            if (url.includes('/coach/eval')) {
                return {
                    data: {
                        project_id: 31, stage: 'eval', handler_available: true,
                        suggestions: [
                            { id: 'i', title: 'Info one', body: '', severity: 'info', action: { kind: 'navigate', label: 'Go', params: { target: 'recipe-picker' } } },
                            { id: 'c', title: 'Critical one', body: '', severity: 'critical', action: { kind: 'navigate', label: 'Go', params: { target: 'recipe-picker' } } },
                            { id: 'w', title: 'Warning one', body: '', severity: 'warning', action: { kind: 'navigate', label: 'Go', params: { target: 'recipe-picker' } } },
                            { id: 'i2', title: 'Info two', body: '', severity: 'info', action: { kind: 'navigate', label: 'Go', params: { target: 'recipe-picker' } } },
                        ],
                    },
                };
            }
            return { data: {} };
        });
    });

    it('shows the stage, the next step, the latest run and the published model', async () => {
        renderOverview();
        expect(screen.getByTestId('overview-stage')).toHaveTextContent('Evaluation');
        expect(screen.getByTestId('overview-next')).toHaveTextContent('Evaluate model quality');
        await waitFor(() => expect(screen.getByTestId('overview-latest-run')).toHaveTextContent('Run #143'));
        expect(screen.getByTestId('overview-latest-run')).toHaveTextContent('span F1 0.778 on the answer key');
        await waitFor(() => expect(screen.getByTestId('overview-shipped')).toHaveTextContent('Export #37'));
        expect(screen.getByRole('link', { name: 'TensorGreed/brewslm-sensitive-data-tagger' })).toBeInTheDocument();
    });

    it('lists at most three nudges, most severe first', async () => {
        renderOverview();
        await screen.findByText('Critical one');
        const nudges = screen.getByTestId('overview-nudges');
        expect(nudges).toHaveTextContent('Critical one');
        expect(nudges).toHaveTextContent('Warning one');
        expect(nudges).toHaveTextContent('Info one');
        expect(screen.queryByText('Info two')).toBeNull();
        expect(nudges.textContent!.indexOf('Critical one')).toBeLessThan(nudges.textContent!.indexOf('Info one'));
    });

    it('the next-step button navigates to the new stage URL', async () => {
        const user = userEvent.setup();
        renderOverview();
        await user.click(screen.getByTestId('overview-next'));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/31/results');
    });

    it('opens the full plan on request', async () => {
        const user = userEvent.setup();
        renderOverview();
        expect(screen.queryByTestId('quickstart')).toBeNull();
        await user.click(screen.getByTestId('overview-plan-toggle'));
        expect(screen.getByTestId('quickstart')).toBeInTheDocument();
        expect(screen.getByText('Train')).toBeInTheDocument();
    });

    it('latestRunText reads the headline kinds', () => {
        expect(latestRunText(null)).toBeNull();
        expect(latestRunText({ verdict: 'no_trained_run', experiment_id: null } as unknown as EvalSummary)).toBeNull();
        expect(latestRunText({
            verdict: 'better', experiment_id: 5, trained: { experiment_id: 5, experiment_name: 'x' },
            headline: { metric_id: 'f1', baseline_value: 0.5, trained_value: 0.7, absolute_delta: 0.2, relative_delta_pct: 40, direction: 'improved' },
        } as unknown as EvalSummary)).toEqual({ title: 'Run #5', detail: 'beats its base model · f1 0.500 → 0.700' });
        expect(latestRunText({ verdict: 'not_evaluated', experiment_id: 6, headline: null } as unknown as EvalSummary))
            .toEqual({ title: 'Run #6', detail: 'not evaluated yet' });
    });
});
