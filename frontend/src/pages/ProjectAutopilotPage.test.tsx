import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
    },
}));

vi.mock('../api/client', () => ({
    default: apiMock,
}));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({
            projectId: 77,
            project: { id: 77, name: 'Test' },
            pipelineStatus: null,
        }),
    };
});

import ProjectAutopilotPage from './ProjectAutopilotPage';

describe('ProjectAutopilotPage', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    function buildPreviewResponse(overrides: Partial<Record<string, unknown>> = {}) {
        return {
            data: {
                preview: {
                    id: 1,
                    plan_token: 'plan-abcdef0123456789',
                    project_id: 77,
                    intent: 'Train a support assistant',
                    expires_at: '2026-04-22T14:30:00+00:00',
                    created_at: '2026-04-22T14:15:00+00:00',
                    applied_at: null,
                    state_hash: 'aabbccddeeff0011',
                },
                config_diff: {
                    summary: 'Will create a new training experiment.',
                    would_create_experiment: true,
                    selected_profile: 'balanced',
                    effective_target_profile_id: 'edge_gpu',
                    preflight_ok: true,
                    repairs_planned: [
                        { kind: 'intent_rewrite', applied: true, original_intent: 'help me', rewritten_intent: 'Train' },
                        { kind: 'target_fallback', applied: false, reason: 'not required' },
                    ],
                    safe_config_preview: { base_model: 'microsoft/phi-2', task_profile: 'qa' },
                    guardrails: { can_run: true, blockers: [], warnings: [], reason_codes: [] },
                    decision_log_preview: [
                        { step: 'strict_mode_policy', status: 'inactive', summary: 'Strict off.' },
                        { step: 'initial_planning', status: 'completed', summary: 'Built initial plan.' },
                    ],
                    strict_mode: false,
                },
                dry_run_response: {
                    project_id: 77,
                    run_id: 'run-dry-abc',
                    dry_run: true,
                    started: false,
                    strict_mode: false,
                    intent: 'Train a support assistant',
                },
                state_hash: 'aabbccddeeff0011',
                ...overrides,
            },
        };
    }

    it('posts to /autopilot/repair-preview and renders the plan + repairs', async () => {
        apiMock.post.mockResolvedValue(buildPreviewResponse());
        const user = userEvent.setup();

        render(<ProjectAutopilotPage />);

        const previewBtn = screen.getByRole('button', { name: /preview plan/i });
        await user.click(previewBtn);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/autopilot/repair-preview',
                expect.objectContaining({ project_id: 77, plan_profile: 'balanced' }),
            );
        });

        const tokenMatches = await screen.findAllByText(/plan-abcdef0123456789/);
        expect(tokenMatches.length).toBeGreaterThan(0);
        expect(screen.getByText(/Will create a new training experiment/)).toBeInTheDocument();
        // Repairs table rows.
        expect(screen.getAllByText(/Intent rewrite/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/Target fallback/i).length).toBeGreaterThan(0);
        // Apply button should be enabled because the plan is runnable.
        const applyBtn = screen.getByRole('button', { name: /apply plan/i });
        expect(applyBtn).not.toBeDisabled();
    });

    it('renders strict-mode refusal as a risk card and disables Apply without force', async () => {
        apiMock.post.mockResolvedValue(
            buildPreviewResponse({
                config_diff: {
                    summary: 'Apply will refuse until blockers clear.',
                    would_create_experiment: false,
                    selected_profile: 'balanced',
                    effective_target_profile_id: 'mobile_cpu',
                    preflight_ok: false,
                    repairs_planned: [
                        {
                            kind: 'target_fallback',
                            applied: false,
                            strict_mode_blocked: true,
                            reason_code: 'STRICT_MODE_REFUSED_TARGET_FALLBACK',
                            from_target_profile_id: 'mobile_cpu',
                        },
                    ],
                    safe_config_preview: {},
                    guardrails: {
                        can_run: false,
                        blockers: [
                            'Strict mode refused to swap target profile. Choose a compatible target explicitly or disable strict mode.',
                        ],
                        warnings: [],
                        reason_codes: ['STRICT_MODE_REFUSED_TARGET_FALLBACK'],
                    },
                    decision_log_preview: [],
                    strict_mode: true,
                },
            }),
        );

        const user = userEvent.setup();
        render(<ProjectAutopilotPage />);

        await user.click(screen.getByRole('button', { name: /preview plan/i }));
        const strictRefusal = await screen.findAllByText(/STRICT_MODE_REFUSED_TARGET_FALLBACK/);
        expect(strictRefusal.length).toBeGreaterThan(0);
        expect(screen.getAllByText(/Strict mode refusal/i).length).toBeGreaterThan(0);

        // can_run=false → Apply disabled without force.
        const applyBtn = screen.getByRole('button', { name: /apply plan/i });
        expect(applyBtn).toBeDisabled();

        // Toggling force re-enables Apply.
        await user.click(screen.getByLabelText(/force apply/i));
        expect(applyBtn).not.toBeDisabled();
    });

    it('apply button calls /autopilot/repair-apply with the plan token', async () => {
        apiMock.post.mockImplementation(async (url: string) => {
            if (url === '/autopilot/repair-preview') {
                return buildPreviewResponse();
            }
            if (url === '/autopilot/repair-apply') {
                return {
                    data: {
                        ok: true,
                        preview: {
                            id: 1,
                            plan_token: 'plan-abcdef0123456789',
                            project_id: 77,
                            intent: 'Train a support assistant',
                            state_hash: 'aabbccddeeff0011',
                            applied_at: '2026-04-22T14:16:00+00:00',
                            applied_run_id: 'run-live-xyz',
                            applied_by: 'api',
                        },
                        response: {
                            project_id: 77,
                            run_id: 'run-live-xyz',
                            dry_run: false,
                            started: true,
                            experiment: { id: 42, name: 'Autopilot Balanced' },
                            decision_log: [
                                { step: 'start_training', status: 'completed', summary: 'Training started.' },
                            ],
                        },
                    },
                };
            }
            throw new Error(`Unexpected POST ${url}`);
        });

        const user = userEvent.setup();
        render(<ProjectAutopilotPage />);

        await user.click(screen.getByRole('button', { name: /preview plan/i }));
        await screen.findAllByText(/plan-abcdef0123456789/);

        await user.click(screen.getByRole('button', { name: /apply plan/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/autopilot/repair-apply',
                expect.objectContaining({ plan_token: 'plan-abcdef0123456789', force: false }),
            );
        });

        // Post-apply confirmation.
        const liveMatches = await screen.findAllByText(/run_id run-live-xyz/);
        expect(liveMatches.length).toBeGreaterThan(0);
        expect(screen.getAllByText(/training started/i).length).toBeGreaterThan(0);
    });

    it('Run Directly posts to /projects/:id/training/autopilot/v2/orchestrate/run with dry_run=false', async () => {
        apiMock.post.mockResolvedValue({
            data: {
                project_id: 77,
                run_id: 'run-direct-xyz',
                dry_run: false,
                started: true,
                experiment: { id: 99, name: 'direct' },
                decision_log: [],
            },
        });

        const user = userEvent.setup();
        render(<ProjectAutopilotPage />);

        await user.click(screen.getByRole('button', { name: /run directly/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/77/training/autopilot/v2/orchestrate/run',
                expect.objectContaining({ dry_run: false, plan_profile: 'balanced' }),
            );
        });
        const directMatches = await screen.findAllByText(/run_id run-direct-xyz/);
        expect(directMatches.length).toBeGreaterThan(0);
    });

    it('surfaces API errors from preview', async () => {
        apiMock.post.mockRejectedValue({
            response: { data: { detail: { reason: 'invalid_request', message: 'bad intent' } } },
        });
        const user = userEvent.setup();
        render(<ProjectAutopilotPage />);
        await user.click(screen.getByRole('button', { name: /preview plan/i }));

        const alert = await screen.findByText(/bad intent/i);
        expect(alert).toBeInTheDocument();
    });

    it('shows guardrail badges and decision log preview', async () => {
        apiMock.post.mockResolvedValue(buildPreviewResponse());
        const user = userEvent.setup();
        render(<ProjectAutopilotPage />);
        await user.click(screen.getByRole('button', { name: /preview plan/i }));

        // Guardrail row has profile and target badges + can_run=true badge.
        const guardrailsSection = (await screen.findByText(/Guardrails/i)).closest('section');
        expect(guardrailsSection).not.toBeNull();
        const scoped = within(guardrailsSection as HTMLElement);
        expect(scoped.getByText(/can_run: true/i)).toBeInTheDocument();
        expect(scoped.getByText(/profile: balanced/i)).toBeInTheDocument();
        expect(scoped.getByText(/target: edge_gpu/i)).toBeInTheDocument();

        // Decision log preview rendered.
        expect(screen.getByText(/Built initial plan/)).toBeInTheDocument();
    });
});
