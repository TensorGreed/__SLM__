import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import CoachStrip from './CoachStrip';
import { _resetCoachModeStoreForTests, useCoachModeStore } from '../../stores/coachModeStore';


function newbieProgressionResponse() {
    return {
        data: {
            xp_balance: 50,
            level: 1,
            level_title: 'Intern',
            xp_into_level: 50,
            xp_to_next_level: 50,
            achievements_unlocked: [],
            milestones: {},
            recent_unlocks: [],
            counters: {
                base_models_trained: [],
                import_sources_used: [],
                successful_training_runs: 0,
            },
        },
    };
}


function installGetRouter(coachStageResponse: any) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/gamification')) {
            return newbieProgressionResponse();
        }
        if (url.includes('/coach/')) {
            return coachStageResponse;
        }
        return { data: {} };
    });
}


describe('CoachStrip', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        _resetCoachModeStoreForTests();
    });

    it('renders nothing when Coach Mode is off for the project', async () => {
        installGetRouter({
            data: { project_id: 1, stage: 'data', suggestions: [], handler_available: true },
        });
        // Flip the override to "off" before mount so the strip never
        // renders content.
        useCoachModeStore.getState().setOverride(1, 'off');
        const { container } = render(<CoachStrip projectId={1} stage="data" />);
        // Give the gamification fetch a beat to land — even after
        // that, the strip must stay empty.
        await new Promise((r) => setTimeout(r, 20));
        expect(container.firstChild).toBeNull();
    });

    it('renders a healthy pill when the backend returns zero suggestions', async () => {
        installGetRouter({
            data: { project_id: 1, stage: 'data', suggestions: [], handler_available: true },
        });
        render(<CoachStrip projectId={1} stage="data" />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-strip-data-healthy')).toBeInTheDocument();
        });
        expect(screen.getByTestId('coach-strip-data-healthy').textContent).toMatch(/healthy/i);
    });

    it('renders one suggestion card per backend suggestion', async () => {
        installGetRouter({
            data: {
                project_id: 1,
                stage: 'data',
                handler_available: true,
                suggestions: [
                    {
                        id: 'data:gold-row-count',
                        title: 'Your gold set has 30 rows',
                        body: 'Most useful first models need at least 100 rows.',
                        severity: 'critical',
                        action: {
                            kind: 'run_playbook',
                            label: 'Generate 70 synthetic positives',
                            params: { mode: 'positives_paraphrase', target_count: 70, target_class: null },
                        },
                        context: { gold_row_count: 30 },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={1} stage="data" />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-suggestion-data:gold-row-count')).toBeInTheDocument();
        });
        expect(screen.getByText(/Your gold set has 30 rows/)).toBeInTheDocument();
        expect(screen.getByTestId('coach-suggestion-action-data:gold-row-count').textContent).toMatch(/Generate 70/);
    });

    it('clicking the run_playbook action posts to /synthetic/run-playbook', async () => {
        installGetRouter({
            data: {
                project_id: 1,
                stage: 'data',
                handler_available: true,
                suggestions: [
                    {
                        id: 'data:gold-row-count',
                        title: 'Your gold set is thin',
                        body: 'Add more rows.',
                        severity: 'critical',
                        action: {
                            kind: 'run_playbook',
                            label: 'Generate 50 positives',
                            params: { mode: 'positives_paraphrase', target_count: 50, target_class: null },
                        },
                    },
                ],
            },
        });
        apiMock.post.mockResolvedValue({ data: { rows: [{ payload: {} }], backend_used: 'ollama', elapsed_sec: 1.2, prompt_snippet: '...' } });

        render(<CoachStrip projectId={1} stage="data" />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-suggestion-action-data:gold-row-count')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('coach-suggestion-action-data:gold-row-count'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/run-playbook',
                expect.objectContaining({
                    mode: 'positives_paraphrase',
                    target_count: 50,
                }),
            );
        });
    });

    it('shows an error fallback when the coach endpoint fails', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/gamification')) {
                return newbieProgressionResponse();
            }
            return Promise.reject({ response: { status: 500, data: { detail: 'boom' } } });
        });
        render(<CoachStrip projectId={1} stage="data" />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-strip-data').textContent).toMatch(/unavailable/i);
        });
    });
});
