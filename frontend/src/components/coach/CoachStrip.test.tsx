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

    it('renders the auto-pinned backend caption with the schema-aware badge (Phase 5c)', async () => {
        // The caption surfaces under the action button so users see
        // the constrained-decoding upgrade happening, not silently
        // applied. Caption renders iff params.backend is set; the
        // schema-aware chip renders iff context.schema_aware_backend
        // matches params.backend (both stamped by coach_service).
        installGetRouter({
            data: {
                project_id: 7,
                stage: 'gold_set',
                handler_available: true,
                suggestions: [
                    {
                        id: 'gold_set:class-imbalance',
                        title: 'Class distribution is skewed',
                        body: 'Generate examples for the minority class.',
                        severity: 'critical',
                        action: {
                            kind: 'run_playbook',
                            label: "Generate 50 examples for 'technical'",
                            params: {
                                mode: 'class_balance_fill',
                                target_count: 50,
                                target_class: 'technical',
                                backend: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
                            },
                        },
                        context: {
                            schema_aware_backend: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
                        },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={7} stage="gold_set" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-gold_set:class-imbalance'),
            ).toBeInTheDocument();
        });
        // Caption visible + names the pinned backend.
        const caption = screen.getByTestId(
            'coach-suggestion-backend-gold_set:class-imbalance',
        );
        expect(caption.textContent).toMatch(/will run on/i);
        expect(caption.textContent).toContain('vllm:meta-llama/Meta-Llama-3.1-8B-Instruct');
        // Schema-aware chip renders.
        expect(
            screen.getByTestId('coach-suggestion-schema-badge-gold_set:class-imbalance'),
        ).toBeInTheDocument();
    });

    it('omits the backend caption when Coach did not pin a backend (Phase 5c)', async () => {
        // Ollama-only install — coach_service omits params.backend so
        // the orchestrator auto-picks. No caption should render; the
        // card should look identical to its pre-5c shape.
        installGetRouter({
            data: {
                project_id: 8,
                stage: 'gold_set',
                handler_available: true,
                suggestions: [
                    {
                        id: 'gold_set:class-imbalance',
                        title: 'Class distribution is skewed',
                        body: 'Generate examples for the minority class.',
                        severity: 'critical',
                        action: {
                            kind: 'run_playbook',
                            label: "Generate 50 examples for 'technical'",
                            params: {
                                mode: 'class_balance_fill',
                                target_count: 50,
                                target_class: 'technical',
                                // no backend pin
                            },
                        },
                        context: { schema_aware_backend: null },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={8} stage="gold_set" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-gold_set:class-imbalance'),
            ).toBeInTheDocument();
        });
        expect(
            screen.queryByTestId('coach-suggestion-backend-gold_set:class-imbalance'),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByTestId('coach-suggestion-schema-badge-gold_set:class-imbalance'),
        ).not.toBeInTheDocument();
    });

    it('renders the backend caption without the schema-aware chip when the pin is not schema-aware (Phase 5c)', async () => {
        // Defensive: if Coach ever pins a non-schema-aware backend
        // (no contract guarantees this won't happen in a future
        // phase), the caption still surfaces the pin so the user
        // can see it — but without the green chip, because
        // context.schema_aware_backend doesn't match.
        installGetRouter({
            data: {
                project_id: 9,
                stage: 'gold_set',
                handler_available: true,
                suggestions: [
                    {
                        id: 'gold_set:class-imbalance',
                        title: 'Class distribution is skewed',
                        body: 'Generate examples for the minority class.',
                        severity: 'critical',
                        action: {
                            kind: 'run_playbook',
                            label: 'Generate 50 examples',
                            params: {
                                mode: 'class_balance_fill',
                                target_count: 50,
                                target_class: 'technical',
                                backend: 'ollama:llama3.1:8b',
                            },
                        },
                        context: { schema_aware_backend: null },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={9} stage="gold_set" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-gold_set:class-imbalance'),
            ).toBeInTheDocument();
        });
        const caption = screen.getByTestId(
            'coach-suggestion-backend-gold_set:class-imbalance',
        );
        expect(caption.textContent).toContain('ollama:llama3.1:8b');
        expect(
            screen.queryByTestId('coach-suggestion-schema-badge-gold_set:class-imbalance'),
        ).not.toBeInTheDocument();
    });

    it('forwards a schema-aware backend pin on run_playbook actions (Phase 5c)', async () => {
        // Coach Mode stamps params.backend on class_balance_fill
        // suggestions when a schema-aware backend (vllm > nemo) is
        // reachable. The click-to-execute flow must forward that pin
        // to runPlaybook so the orchestrator routes to the schema-
        // honoring backend instead of auto-picking Ollama.
        installGetRouter({
            data: {
                project_id: 1,
                stage: 'gold_set',
                handler_available: true,
                suggestions: [
                    {
                        id: 'gold_set:class-imbalance',
                        title: 'Class distribution is skewed',
                        body: 'Generate examples for the minority class.',
                        severity: 'critical',
                        action: {
                            kind: 'run_playbook',
                            label: "Generate 50 examples for 'technical'",
                            params: {
                                mode: 'class_balance_fill',
                                target_count: 50,
                                target_class: 'technical',
                                backend: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
                            },
                        },
                    },
                ],
            },
        });
        apiMock.post.mockResolvedValue({
            data: {
                rows: [{ payload: {} }],
                backend_used: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
                elapsed_sec: 1.2,
                prompt_snippet: '...',
            },
        });

        render(<CoachStrip projectId={1} stage="gold_set" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-gold_set:class-imbalance'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-gold_set:class-imbalance'),
        );
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/run-playbook',
                expect.objectContaining({
                    mode: 'class_balance_fill',
                    target_count: 50,
                    target_class: 'technical',
                    backend: 'vllm:meta-llama/Meta-Llama-3.1-8B-Instruct',
                }),
            );
        });
    });

    it('clicking the augment_from_cluster action posts to /evaluation/.../augment (Phase 3)', async () => {
        installGetRouter({
            data: {
                project_id: 1,
                stage: 'eval',
                handler_available: true,
                suggestions: [
                    {
                        id: 'eval:top-failure-cluster',
                        title: 'Top failure cluster: 12 hallucination failures (45%)',
                        body: 'Augmenting this cluster bridges the gap.',
                        severity: 'critical',
                        action: {
                            kind: 'augment_from_cluster',
                            label: 'Augment 30 rows for this cluster',
                            params: {
                                eval_result_id: 42,
                                cluster_id: 'cluster-1',
                                target_count: 30,
                            },
                        },
                    },
                ],
            },
        });
        apiMock.post.mockResolvedValue({ data: { rows: [{ payload: {} }], backend_used: 'ollama', elapsed_sec: 0.5, prompt_snippet: '...' } });

        render(<CoachStrip projectId={1} stage="eval" />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-suggestion-action-eval:top-failure-cluster')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('coach-suggestion-action-eval:top-failure-cluster'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/evaluation/42/clusters/cluster-1/augment',
                null,
                expect.objectContaining({
                    params: expect.objectContaining({ target_count: 30 }),
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
