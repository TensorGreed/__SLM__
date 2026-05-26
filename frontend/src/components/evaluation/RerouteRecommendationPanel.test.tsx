import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, toastMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    toastMock: {
        success: vi.fn(),
        error: vi.fn(),
        info: vi.fn(),
        warning: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../stores/toastStore', () => ({ toast: toastMock }));

// Navigation is via window.location.assign — spy on that so we can
// assert the post-clone URL without requiring a Router context.
const locationAssignMock = vi.fn();
Object.defineProperty(window, 'location', {
    value: { assign: locationAssignMock, href: 'http://localhost/' },
    writable: true,
});

import RerouteRecommendationPanel from './RerouteRecommendationPanel';


function makeAnalysis(overrides: Partial<{
    kind: 'try_rag' | 'try_prompt_engineering' | 'expand_data' | 'stay_the_course';
    pass_rate: number | null;
    firedSignalIds: string[];
}>) {
    const kind = overrides.kind ?? 'try_rag';
    const pass_rate = overrides.pass_rate ?? 0.35;
    const firedSignalIds = overrides.firedSignalIds ?? ['brief_mentions_retrieval', 'goldset_answer_diversity_high'];
    const allSignals = [
        {
            id: 'brief_mentions_retrieval',
            fired: firedSignalIds.includes('brief_mentions_retrieval'),
            detail: 'Your brief mentions retrieval-style language',
            evidence: { matched_keywords: ['answer questions about'] },
        },
        {
            id: 'goldset_answer_diversity_high',
            fired: firedSignalIds.includes('goldset_answer_diversity_high'),
            detail: 'Gold-set answers are highly diverse (mean Jaccard 0.05)',
            evidence: { mean_pairwise_jaccard: 0.05 },
        },
        {
            id: 'input_output_density_low',
            fired: firedSignalIds.includes('input_output_density_low'),
            detail: 'Output is a tiny slice of input (mean ratio 0.02)',
            evidence: { mean_density: 0.02 },
        },
    ];
    return {
        eval_result_id: 101,
        project_id: 4,
        pass_rate,
        signals: allSignals,
        recommendation: {
            kind,
            confidence: kind === 'try_rag' ? 0.85 : 0.55,
            rationale: 'A RAG-first project would retrieve from your gold set at inference instead of relying on what the model memorized during fine-tuning.',
        },
        computed_at: '2026-05-26T12:00:00Z',
    };
}


function renderPanel(props: { projectId?: number; evalResultId?: number | null } = {}) {
    return render(
        <RerouteRecommendationPanel
            projectId={props.projectId ?? 4}
            evalResultId={props.evalResultId === undefined ? 101 : props.evalResultId}
        />,
    );
}


describe('RerouteRecommendationPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        locationAssignMock.mockReset();
        toastMock.success.mockReset();
        toastMock.error.mockReset();
    });

    it('renders nothing when evalResultId is null (no eval yet)', async () => {
        const { container } = renderPanel({ evalResultId: null });
        // No fetch fired.
        expect(apiMock.get).not.toHaveBeenCalled();
        expect(container.querySelector('.reroute-card')).toBeNull();
    });

    it('renders the try_rag card with fired signals + amber styling', async () => {
        apiMock.get.mockResolvedValueOnce({ data: makeAnalysis({ kind: 'try_rag' }) });
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-try-rag')).toBeInTheDocument();
        });
        // Two signals fired in the default fixture.
        expect(screen.getByTestId('reroute-card-signal-brief_mentions_retrieval')).toBeInTheDocument();
        expect(screen.getByTestId('reroute-card-signal-goldset_answer_diversity_high')).toBeInTheDocument();
        // Confidence chip shows the 85% from the fixture.
        expect(screen.getByTestId('reroute-card-confidence')).toHaveTextContent('85%');
        // Primary CTA reads "Switch to RAG…"
        expect(screen.getByTestId('reroute-card-cta')).toHaveTextContent(/switch to rag/i);
    });

    it('self-hides on stay_the_course', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: makeAnalysis({ kind: 'stay_the_course', pass_rate: 0.92, firedSignalIds: [] }),
        });
        const { container } = renderPanel();
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        // Wait a tick for the effect to flush.
        await new Promise((r) => setTimeout(r, 0));
        expect(container.querySelector('.reroute-card')).toBeNull();
    });

    it('renders try_prompt_engineering card with Playground CTA', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: makeAnalysis({
                kind: 'try_prompt_engineering',
                firedSignalIds: ['input_output_density_low'],
            }),
        });
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-try-prompt')).toBeInTheDocument();
        });
        expect(screen.getByTestId('reroute-card-cta')).toHaveTextContent(/playground/i);
    });

    it('renders expand_data card as the catch-all', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: makeAnalysis({ kind: 'expand_data', firedSignalIds: [] }),
        });
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-expand-data')).toBeInTheDocument();
        });
        // No fired signals → the signals list is omitted entirely.
        expect(screen.queryByTestId('reroute-card-signals')).toBeNull();
    });

    it('silently renders nothing when the fetch fails (404 / 400 / network)', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { status: 404 } });
        const { container } = renderPanel();
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        await new Promise((r) => setTimeout(r, 0));
        // Advisory panel — failures self-hide rather than render a noisy error.
        expect(container.querySelector('.reroute-card')).toBeNull();
    });

    it('opens the confirmation modal when "Switch to RAG" is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: makeAnalysis({ kind: 'try_rag' }) });
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-cta')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('reroute-card-cta'));
        expect(screen.getByTestId('reroute-modal')).toBeInTheDocument();
        // Modal body mentions the source project id.
        expect(screen.getByTestId('reroute-modal')).toHaveTextContent('#4');
    });

    it('cancels the modal without firing the clone POST', async () => {
        apiMock.get.mockResolvedValueOnce({ data: makeAnalysis({ kind: 'try_rag' }) });
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-cta')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('reroute-card-cta'));
        await userEvent.click(screen.getByTestId('reroute-modal-cancel'));
        expect(screen.queryByTestId('reroute-modal')).toBeNull();
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('confirms the modal → POSTs async reroute-to-rag → fires info toast (Hardening Phase H1)', async () => {
        apiMock.get.mockResolvedValueOnce({ data: makeAnalysis({ kind: 'try_rag' }) });
        // Async path returns a Job stub (202) — no direct navigate
        // since the work runs in the background. The notification
        // bell takes over once the job completes.
        apiMock.post.mockResolvedValueOnce({
            data: {
                id: 7,
                kind: 'reroute_to_rag',
                title: 'Clone to RAG · project #4',
                status: 'queued',
                progress: null,
                progress_message: null,
                project_id: 4,
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
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-cta')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('reroute-card-cta'));
        await userEvent.click(screen.getByTestId('reroute-modal-confirm'));
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/4/reroute-to-rag?async_job=true',
                {},
            );
        });
        // Async flow uses an info toast, not success — and no direct
        // navigate (the bell does it when the job completes).
        await waitFor(() => {
            expect(toastMock.info).toHaveBeenCalledWith(
                expect.stringContaining('Cloning started'),
                4000,
            );
        });
        expect(locationAssignMock).not.toHaveBeenCalled();
    });

    it('surfaces a toast.error when the clone POST fails', async () => {
        apiMock.get.mockResolvedValueOnce({ data: makeAnalysis({ kind: 'try_rag' }) });
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'source_recipe_not_eligible:classification' } },
        });
        renderPanel();
        await waitFor(() => {
            expect(screen.getByTestId('reroute-card-cta')).toBeInTheDocument();
        });
        await userEvent.click(screen.getByTestId('reroute-card-cta'));
        await userEvent.click(screen.getByTestId('reroute-modal-confirm'));
        await waitFor(() => {
            expect(toastMock.error).toHaveBeenCalledWith(
                expect.stringContaining('source_recipe_not_eligible:classification'),
            );
        });
        // No navigate fired.
        expect(locationAssignMock).not.toHaveBeenCalled();
    });
});
