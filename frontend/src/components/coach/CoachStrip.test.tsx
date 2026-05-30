import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    navigateMock: vi.fn(),
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

// CoachSuggestion uses `useNavigate` for Phase 5c navigate actions
// that route to a concrete URL (e.g. synthetic-review-queue). The
// strip is rendered without a Router in these tests, so we mock the
// hook directly.
vi.mock('react-router-dom', () => ({
    useNavigate: () => navigateMock,
}));

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
        navigateMock.mockReset();
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
            // Hardening — Coach run_playbook actions fire the
            // async-job variant of the synth endpoint, not the
            // blocking sync one. The bell takes over progress.
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/run-playbook?async_job=true',
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

    it('navigates to the Synthetic tab review queue with focus_synth_source query (Phase 5c)', async () => {
        // Phase 5c — known navigate targets route to a concrete URL
        // instead of falling through to the toast-hint fallback. The
        // gold_set:synth-review-pending suggestion lands on the
        // Synthetic pipeline tab (where SynthReviewQueue is mounted),
        // forwards synth_source as ?focus_synth_source=..., and uses
        // the #synth-review-queue hash so SyntheticPanel scrolls.
        installGetRouter({
            data: {
                project_id: 42,
                stage: 'gold_set',
                handler_available: true,
                suggestions: [
                    {
                        id: 'gold_set:synth-review-pending',
                        title: '7 synthetic rows pending review',
                        body: 'Generated rows are excluded from training until accepted.',
                        severity: 'warning',
                        action: {
                            kind: 'navigate',
                            label: 'Open review queue',
                            params: {
                                target: 'synthetic-review-queue',
                                synth_source:
                                    'playbook:classification:class_balance_fill:class=technical',
                            },
                        },
                        context: { total_pending: 7 },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={42} stage="gold_set" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-gold_set:synth-review-pending'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-gold_set:synth-review-pending'),
        );
        await waitFor(() => {
            expect(navigateMock).toHaveBeenCalledWith(
                '/project/42/pipeline/synthetic?focus_synth_source=playbook%3Aclassification%3Aclass_balance_fill%3Aclass%3Dtechnical#synth-review-queue',
            );
        });
    });

    it('navigates without focus_synth_source when the action omits it (Phase 5c)', async () => {
        // When the queue's top group has no synth_source (legacy
        // rows), coach_service omits params.synth_source — URL still
        // routes to the Synthetic tab but without the focus query.
        installGetRouter({
            data: {
                project_id: 42,
                stage: 'gold_set',
                handler_available: true,
                suggestions: [
                    {
                        id: 'gold_set:synth-review-pending',
                        title: '3 synthetic rows pending review',
                        body: '…',
                        severity: 'info',
                        action: {
                            kind: 'navigate',
                            label: 'Open review queue',
                            params: { target: 'synthetic-review-queue' },
                        },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={42} stage="gold_set" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-gold_set:synth-review-pending'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-gold_set:synth-review-pending'),
        );
        await waitFor(() => {
            expect(navigateMock).toHaveBeenCalledWith(
                '/project/42/pipeline/synthetic#synth-review-queue',
            );
        });
    });

    it('navigates to the task-shape recipe-picker page on the recipe-picker target', async () => {
        // The recipe-picker maps to the standalone task-shape recipe
        // picker (/project/{id}/recipe-picker) — distinct from
        // /recipes, which is the pipeline-DAG recipes page. Emitted
        // by every "Pick a recipe before doing X" Coach suggestion
        // on data / gold_set / training stages. Pointing at /recipes
        // (the prior URL) was a bug — users would land on the
        // pipeline-DAG page, pick a pipeline recipe, and the
        // "no recipe selected" signals would persist because
        // Project.selected_recipe was never populated.
        installGetRouter({
            data: {
                project_id: 1,
                stage: 'data',
                handler_available: true,
                suggestions: [
                    {
                        id: 'data:no-recipe',
                        title: 'Pick a recipe',
                        body: 'Coach needs the recipe to score.',
                        severity: 'warning',
                        action: {
                            kind: 'navigate',
                            label: 'Open recipe picker',
                            params: { target: 'recipe-picker' },
                        },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={1} stage="data" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-data:no-recipe'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-data:no-recipe'),
        );
        await waitFor(() => {
            expect(navigateMock).toHaveBeenCalledWith('/project/1/recipe-picker');
        });
    });

    it('navigates to the training-config page on the training-config target', async () => {
        // Phase 6d's curriculum nudge + Phase 9d's auto-RAG nudge both
        // emit ``target: 'training-config'`` — same standalone-page
        // problem the recipe-picker had (toast-only on a route that
        // really exists). Maps to /project/{id}/training-config, the
        // same path ProjectSidebar uses.
        installGetRouter({
            data: {
                project_id: 7,
                stage: 'training',
                handler_available: true,
                suggestions: [
                    {
                        id: 'training:curriculum-learning-available',
                        title: 'Curriculum recommended',
                        body: 'Body.',
                        severity: 'info',
                        action: {
                            kind: 'navigate',
                            label: 'Open Training Config',
                            params: { target: 'training-config' },
                        },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={7} stage="training" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-training:curriculum-learning-available'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-training:curriculum-learning-available'),
        );
        await waitFor(() => {
            expect(navigateMock).toHaveBeenCalledWith('/project/7/training-config');
        });
    });

    it('unknown navigate targets still fall back to the toast hint (Phase 1 backwards-compat)', async () => {
        // Any target that isn't in NAVIGATE_TARGET_URLS keeps the
        // toast-hint behavior — so partial / future / unwired targets
        // don't crash the click.
        installGetRouter({
            data: {
                project_id: 1,
                stage: 'data',
                handler_available: true,
                suggestions: [
                    {
                        id: 'data:future-target',
                        title: 'Imaginary future suggestion',
                        body: 'Some body.',
                        severity: 'info',
                        action: {
                            kind: 'navigate',
                            label: 'Do the thing',
                            params: { target: 'this-target-isnt-wired-yet' },
                        },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={1} stage="data" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-data:future-target'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-data:future-target'),
        );
        // Did NOT navigate — the toast-hint path ran instead.
        expect(navigateMock).not.toHaveBeenCalled();
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
        // Async-job variant returns a Job stub (202), not a
        // PlaybookResult — the bell takes over progress + outcome.
        apiMock.post.mockResolvedValue({
            data: {
                id: 99,
                kind: 'synth_playbook',
                title: 'Synth · class_balance_fill · 50 rows',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 1,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
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
            // Hardening — async-job variant of the synth endpoint;
            // the backend pin still flows through verbatim.
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/synthetic/run-playbook?async_job=true',
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
        // Async-job variant returns a Job stub (202) — bell takes
        // over progress + the 0-rows-as-FAILED guard.
        apiMock.post.mockResolvedValue({
            data: {
                id: 101,
                kind: 'synth_augment_from_cluster',
                title: 'Augment cluster cluster-1 · 30 rows',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 1,
                user_id: null,
                params: {},
                result: null,
                error: null,
                queued_at: '2026-05-26T12:00:00Z',
                started_at: null,
                completed_at: null,
                dismissed_at: null,
            },
        });

        render(<CoachStrip projectId={1} stage="eval" />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-suggestion-action-eval:top-failure-cluster')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('coach-suggestion-action-eval:top-failure-cluster'));
        await waitFor(() => {
            // Hardening — same endpoint, with the async_job param
            // flipped on so it spawns a Job and returns 202 instead
            // of blocking for 30-180s on the LLM call.
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/evaluation/42/clusters/cluster-1/augment',
                null,
                expect.objectContaining({
                    params: expect.objectContaining({
                        target_count: 30,
                        async_job: true,
                    }),
                }),
            );
        });
    });

    it('navigates to the observability FailureClusterList on the failure-clusters-panel target', async () => {
        // Sweep-inconclusive nudge target — coach surfaces it on the
        // training stage when the latest sweep's verdict is
        // inconclusive. The deep-link lands the user on the
        // observability page with #failure-clusters anchoring the
        // FailureClusterList section so they see why each cell missed
        // rather than promoting a sub-gate model.
        installGetRouter({
            data: {
                project_id: 7,
                stage: 'training',
                handler_available: true,
                suggestions: [
                    {
                        id: 'training:sweep-inconclusive',
                        title: 'Sweep inconclusive — 4/4 cells, none cleared gate',
                        body: 'Sweep sweepabc finished with no cell clearing the project gate.',
                        severity: 'warning',
                        action: {
                            kind: 'navigate',
                            label: 'Open Failure clusters',
                            params: { target: 'failure-clusters-panel' },
                        },
                        context: { sweep_id: 'sweepabc' },
                    },
                ],
            },
        });
        render(<CoachStrip projectId={7} stage="training" />);
        await waitFor(() => {
            expect(
                screen.getByTestId('coach-suggestion-action-training:sweep-inconclusive'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('coach-suggestion-action-training:sweep-inconclusive'),
        );
        await waitFor(() => {
            expect(navigateMock).toHaveBeenCalledWith(
                '/project/7/observability#failure-clusters',
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
