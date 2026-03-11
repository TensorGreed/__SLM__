import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import WorkflowRunMonitor from './WorkflowRunMonitor';

const DEFAULT_RUNTIME = {
    execution_modes: ['local'],
    required_services: [],
    required_env: [],
    required_settings: [],
    requires_gpu: false,
    min_vram_gb: 0,
};

function makeNode(stage: string, stepType: string, config: Record<string, unknown>) {
    return {
        id: `step:${stage}`,
        stage,
        display_name: stage,
        index: 0,
        kind: 'template_step',
        status: 'pending',
        step_type: stepType,
        description: `${stage} step`,
        input_artifacts: [],
        output_artifacts: [],
        config_schema_ref: `slm.step.${stage}/v1`,
        config,
        runtime_requirements: DEFAULT_RUNTIME,
        position: { x: 0, y: 0 },
    };
}

const AUTOPILOT_GRAPH = {
    project_id: 1,
    graph_id: 'template.autopilot_chat',
    graph_label: 'Autopilot Chat Pipeline',
    graph_version: '1.0.0',
    mode: 'readonly_preview',
    current_stage: 'ingestion',
    nodes: [
        makeNode('ingestion', 'core.ingestion', {}),
        makeNode('synthetic_conversation', 'core.synthetic_conversation', { mode: 'noop' }),
        makeNode('semantic_curation', 'core.semantic_curation', { mode: 'noop' }),
        makeNode('cloud_burst', 'core.cloud_burst_plan', { mode: 'plan' }),
        makeNode('distillation', 'core.distillation_training', { mode: 'noop' }),
        makeNode('model_merge', 'core.model_merge', { mode: 'noop' }),
    ],
    edges: [],
};

const AUTOPILOT_TEMPLATE = {
    template_id: 'template.autopilot_chat',
    display_name: 'Autopilot Chat Pipeline',
    description: 'Autopilot graph',
    graph: AUTOPILOT_GRAPH,
};

type MockPostCall = [url: string, payload?: Record<string, unknown>];
type ScorecardResponse = {
    project_id: number;
    template_id: string;
    generated_at: string;
    run_window_limit: number;
    run_window_count: number;
    autopilot_run_count: number;
    latest_run_id: number | null;
    latest_profile: 'safe' | 'guided' | 'full' | null;
    recommended_profile: 'safe' | 'guided' | 'full';
    promotion_available: boolean;
    demotion_suggested: boolean;
    reason: string;
    profiles: Array<'safe' | 'guided' | 'full'>;
    by_profile: Record<'safe' | 'guided' | 'full', {
        runs: number;
        completed_runs: number;
        failed_runs: number;
        blocked_runs: number;
        cancelled_runs: number;
        pending_runs: number;
        running_runs: number;
        preflight_checks: number;
        preflight_passed: number;
        success_rate: number | null;
        blocked_or_failed_rate: number | null;
        preflight_pass_rate: number | null;
        last_run_id: number | null;
        last_run_at: string | null;
    }>;
    recent_runs: Array<Record<string, unknown>>;
};

function makeScorecard(recommendedProfile: 'safe' | 'guided' | 'full' = 'safe'): ScorecardResponse {
    return {
        project_id: 1,
        template_id: 'template.autopilot_chat',
        generated_at: '2026-03-10T00:00:00.000Z',
        run_window_limit: 30,
        run_window_count: 0,
        autopilot_run_count: 0,
        latest_run_id: null,
        latest_profile: null,
        recommended_profile: recommendedProfile,
        promotion_available: recommendedProfile !== 'safe',
        demotion_suggested: false,
        reason: 'Autopilot scorecard baseline',
        profiles: ['safe', 'guided', 'full'],
        by_profile: {
            safe: {
                runs: 2,
                completed_runs: 2,
                failed_runs: 0,
                blocked_runs: 0,
                cancelled_runs: 0,
                pending_runs: 0,
                running_runs: 0,
                preflight_checks: 2,
                preflight_passed: 2,
                success_rate: 1,
                blocked_or_failed_rate: 0,
                preflight_pass_rate: 1,
                last_run_id: 21,
                last_run_at: '2026-03-10T00:00:00.000Z',
            },
            guided: {
                runs: 0,
                completed_runs: 0,
                failed_runs: 0,
                blocked_runs: 0,
                cancelled_runs: 0,
                pending_runs: 0,
                running_runs: 0,
                preflight_checks: 0,
                preflight_passed: 0,
                success_rate: null,
                blocked_or_failed_rate: null,
                preflight_pass_rate: null,
                last_run_id: null,
                last_run_at: null,
            },
            full: {
                runs: 0,
                completed_runs: 0,
                failed_runs: 0,
                blocked_runs: 0,
                cancelled_runs: 0,
                pending_runs: 0,
                running_runs: 0,
                preflight_checks: 0,
                preflight_passed: 0,
                success_rate: null,
                blocked_or_failed_rate: null,
                preflight_pass_rate: null,
                last_run_id: null,
                last_run_at: null,
            },
        },
        recent_runs: [],
    };
}

function compileResponse(overrides?: Record<string, unknown>) {
    return {
        project_id: 1,
        current_stage: 'ingestion',
        valid_graph: true,
        fallback_used: false,
        requested_source: 'request_override',
        effective_source: 'request_override',
        has_saved_override: false,
        errors: [],
        warnings: [],
        checks: {
            active_stage_present: true,
            active_stage_node_id: 'step:ingestion',
            active_stage_missing_inputs: [],
            active_stage_runtime_requirements: DEFAULT_RUNTIME,
            active_stage_missing_runtime_requirements: [],
            active_stage_runtime_ready: true,
            active_stage_ready_now: true,
        },
        available_artifacts: [],
        graph: AUTOPILOT_GRAPH,
        ...(overrides || {}),
    };
}

describe('WorkflowRunMonitor autopilot gate', () => {
    beforeEach(() => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/pipeline/graph/workflow-runs')) {
                return { data: { project_id: 1, limit: 20, count: 0, runs: [] } };
            }
            if (url.includes('/pipeline/graph/templates')) {
                return { data: { project_id: 1, templates: [AUTOPILOT_TEMPLATE] } };
            }
            if (url.includes('/pipeline/graph/autopilot/scorecard')) {
                return { data: makeScorecard('safe') };
            }
            return { data: {} };
        });
        apiMock.post.mockResolvedValue({ data: {} });
        apiMock.put.mockResolvedValue({ data: {} });
    });

    it('blocks autopilot run when preflight gate fails', async () => {
        const user = userEvent.setup();
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.includes('/pipeline/graph/compile')) {
                return {
                    data: compileResponse({
                        checks: {
                            active_stage_present: true,
                            active_stage_node_id: 'step:ingestion',
                            active_stage_missing_inputs: ['dataset.raw'],
                            active_stage_runtime_requirements: DEFAULT_RUNTIME,
                            active_stage_missing_runtime_requirements: [],
                            active_stage_runtime_ready: true,
                            active_stage_ready_now: false,
                        },
                    }),
                };
            }
            if (url.includes('/pipeline/graph/run-async')) {
                return {
                    data: {
                        project_id: 1,
                        queued: true,
                        run_id: 99,
                        run: {
                            id: 99,
                            project_id: 1,
                            graph_id: 'template.autopilot_chat',
                            graph_version: '1.0.0',
                            execution_backend: 'local',
                            status: 'pending',
                            run_config: {},
                            summary: { node_counts: {}, total_nodes: 0 },
                            started_at: null,
                            finished_at: null,
                            created_at: null,
                            updated_at: null,
                            nodes: [],
                        },
                    },
                };
            }
            return { data: {} };
        });

        render(<WorkflowRunMonitor projectId={1} currentStage="ingestion" />);

        const graphSourceSelect = await screen.findByLabelText('Graph Source');
        await user.selectOptions(graphSourceSelect, 'template.autopilot_chat');
        expect(await screen.findByText('Autopilot Chat Flow')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Run Autopilot Path' }));

        await waitFor(() => {
            expect(screen.getByText(/Autopilot run blocked by preflight gate/i)).toBeInTheDocument();
        });
        expect(screen.getByText(/Preflight: BLOCKED/i)).toBeInTheDocument();

        const runAsyncCall = (apiMock.post.mock.calls as MockPostCall[]).find(([url]) =>
            String(url).includes('/pipeline/graph/run-async'),
        );
        expect(runAsyncCall).toBeUndefined();
    });

    it('queues autopilot run with profile metadata when preflight passes', async () => {
        const user = userEvent.setup();
        apiMock.post.mockImplementation(async (url: string, payload?: Record<string, unknown>) => {
            const payloadGraph = (
                payload?.graph && typeof payload.graph === 'object'
                    ? payload.graph as { graph_id?: string }
                    : undefined
            );
            const payloadConfig = (
                payload?.config && typeof payload.config === 'object'
                    ? payload.config as Record<string, unknown>
                    : {}
            );
            if (url.includes('/pipeline/graph/compile')) {
                return {
                    data: compileResponse({
                        checks: {
                            active_stage_present: true,
                            active_stage_node_id: 'step:ingestion',
                            active_stage_missing_inputs: ['source.file', 'source.remote_dataset'],
                            active_stage_runtime_requirements: DEFAULT_RUNTIME,
                            active_stage_missing_runtime_requirements: [],
                            active_stage_runtime_ready: true,
                            active_stage_ready_now: false,
                        },
                    }),
                };
            }
            if (url.includes('/pipeline/graph/run-async')) {
                return {
                    data: {
                        project_id: 1,
                        queued: true,
                        run_id: 77,
                        run: {
                            id: 77,
                            project_id: 1,
                            graph_id: payloadGraph?.graph_id || 'template.autopilot_chat',
                            graph_version: '1.0.0',
                            execution_backend: 'local',
                            status: 'pending',
                            run_config: payloadConfig,
                            summary: { node_counts: {}, total_nodes: 0 },
                            started_at: null,
                            finished_at: null,
                            created_at: null,
                            updated_at: null,
                            nodes: [],
                        },
                    },
                };
            }
            return { data: {} };
        });

        render(<WorkflowRunMonitor projectId={1} currentStage="ingestion" />);

        const graphSourceSelect = await screen.findByLabelText('Graph Source');
        await user.selectOptions(graphSourceSelect, 'template.autopilot_chat');
        expect(await screen.findByText('Autopilot Chat Flow')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Run Autopilot Path' }));

        await waitFor(() => {
            expect(screen.getByText(/Autopilot run #77 queued/i)).toBeInTheDocument();
        });

        const runAsyncCall = (apiMock.post.mock.calls as MockPostCall[]).find(([url]) =>
            String(url).includes('/pipeline/graph/run-async'),
        );
        expect(runAsyncCall).toBeTruthy();
        expect(runAsyncCall?.[1]).toEqual(
            expect.objectContaining({
                execution_backend: 'local',
                config: expect.objectContaining({
                    bootstrap_source_artifacts: true,
                    autopilot_template_id: 'template.autopilot_chat',
                    autopilot: expect.objectContaining({
                        profile: 'safe',
                    }),
                }),
            }),
        );
    });

    it('uses promoted scorecard recommendation for autopilot profile payload', async () => {
        const user = userEvent.setup();
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/pipeline/graph/workflow-runs')) {
                return { data: { project_id: 1, limit: 20, count: 0, runs: [] } };
            }
            if (url.includes('/pipeline/graph/templates')) {
                return { data: { project_id: 1, templates: [AUTOPILOT_TEMPLATE] } };
            }
            if (url.includes('/pipeline/graph/autopilot/scorecard')) {
                return { data: makeScorecard('guided') };
            }
            return { data: {} };
        });
        apiMock.post.mockImplementation(async (url: string, payload?: Record<string, unknown>) => {
            const payloadGraph = (
                payload?.graph && typeof payload.graph === 'object'
                    ? payload.graph as { graph_id?: string; nodes?: Array<Record<string, unknown>> }
                    : undefined
            );
            const payloadConfig = (
                payload?.config && typeof payload.config === 'object'
                    ? payload.config as Record<string, unknown>
                    : {}
            );
            if (url.includes('/pipeline/graph/compile')) {
                return {
                    data: compileResponse({
                        checks: {
                            active_stage_present: true,
                            active_stage_node_id: 'step:ingestion',
                            active_stage_missing_inputs: ['source.file', 'source.remote_dataset'],
                            active_stage_runtime_requirements: DEFAULT_RUNTIME,
                            active_stage_missing_runtime_requirements: [],
                            active_stage_runtime_ready: true,
                            active_stage_ready_now: false,
                        },
                    }),
                };
            }
            if (url.includes('/pipeline/graph/run-async')) {
                return {
                    data: {
                        project_id: 1,
                        queued: true,
                        run_id: 101,
                        run: {
                            id: 101,
                            project_id: 1,
                            graph_id: payloadGraph?.graph_id || 'template.autopilot_chat',
                            graph_version: '1.0.0',
                            execution_backend: 'local',
                            status: 'pending',
                            run_config: payloadConfig,
                            summary: { node_counts: {}, total_nodes: 0 },
                            started_at: null,
                            finished_at: null,
                            created_at: null,
                            updated_at: null,
                            nodes: [],
                        },
                    },
                };
            }
            return { data: {} };
        });

        render(<WorkflowRunMonitor projectId={1} currentStage="ingestion" />);

        const graphSourceSelect = await screen.findByLabelText('Graph Source');
        await user.selectOptions(graphSourceSelect, 'template.autopilot_chat');
        expect(await screen.findByText('Autopilot Chat Flow')).toBeInTheDocument();
        expect(await screen.findByText(/recommended: guided/i)).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Run Autopilot Path' }));

        await waitFor(() => {
            expect(screen.getByText(/Autopilot run #101 queued/i)).toBeInTheDocument();
        });

        const runAsyncCall = (apiMock.post.mock.calls as MockPostCall[]).find(([url]) =>
            String(url).includes('/pipeline/graph/run-async'),
        );
        expect(runAsyncCall).toBeTruthy();
        const payload = (runAsyncCall?.[1] || {}) as Record<string, unknown>;
        expect(payload.config).toEqual(
            expect.objectContaining({
                autopilot: expect.objectContaining({
                    profile: 'guided',
                }),
            }),
        );
        const payloadGraph = payload.graph as { nodes?: Array<Record<string, unknown>> };
        const guidedSyntheticNode = (payloadGraph.nodes || []).find(
            (node) => String(node.stage) === 'synthetic_conversation',
        );
        expect(guidedSyntheticNode?.config).toEqual(
            expect.objectContaining({
                mode: 'generate_and_save',
            }),
        );
    });
});
