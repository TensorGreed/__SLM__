/**
 * ArchetypeComparisonPanel tests (USER-SUCCESS Epic 8 Phase 8b).
 *
 * Covers:
 *   * Self-hide on the healthy + cold-start (no user projects) path.
 *   * Renders the table with per-feature status badges + cohort line.
 *   * run_playbook suggested-action button calls runPlaybookAsync +
 *     fires the info toast + kicks the jobs store.
 *   * navigate suggested-action button calls window.location.assign.
 *   * 4xx / network error → silent (no panel rendered).
 */

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

const refreshSpy = vi.fn();
vi.mock('../../stores/jobsStore', () => ({
    useJobsStore: {
        getState: () => ({ refreshAfterLocalChange: refreshSpy }),
    },
}));

const locationAssignMock = vi.fn();
Object.defineProperty(window, 'location', {
    value: { assign: locationAssignMock, href: 'http://localhost/' },
    writable: true,
});

import ArchetypeComparisonPanel from './ArchetypeComparisonPanel';


function comparisonFixture(overrides: {
    summary?: 'healthy' | 'below_cohort' | 'above_cohort' | 'mixed';
    nUserProjects?: number;
    features?: Array<{
        feature_id: string;
        status: 'ok' | 'below' | 'above' | 'missing';
        suggestion?: string | null;
        suggested_action?: {
            kind: 'run_playbook' | 'navigate';
            params: Record<string, unknown>;
        } | null;
    }>;
} = {}) {
    const summary = overrides.summary ?? 'below_cohort';
    const nUser = overrides.nUserProjects ?? 0;
    const features = (overrides.features ?? [
        {
            feature_id: 'row_count',
            status: 'below',
            suggestion: 'Generate 80 more rows.',
            suggested_action: {
                kind: 'run_playbook',
                params: {
                    mode: 'positives_paraphrase',
                    target_count: 80,
                },
            },
        },
    ]).map((f) => ({
        feature_id: f.feature_id,
        label: `Feature ${f.feature_id}`,
        unit: 'rows',
        your_value: 50,
        archetype_p25: 100,
        archetype_p50: 200,
        archetype_p75: 300,
        status: f.status,
        suggestion: f.suggestion ?? null,
        suggested_action: f.suggested_action ?? null,
    }));
    return {
        project_id: 4,
        recipe_id: 'classification',
        archetype: {
            recipe_id: 'classification',
            n_passing_projects: 2 + nUser,
            n_user_projects: nUser,
            n_template_seeds: 2,
            computed_at: '2026-05-26T12:00:00Z',
            features: [],
            cohort_provenance: [
                { id: -1, name: 'Template · ticket-router', source: 'template', pass_rate: null },
                { id: -2, name: 'Template · log-triage', source: 'template', pass_rate: null },
            ],
        },
        features,
        summary,
    };
}


describe('ArchetypeComparisonPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        toastMock.info.mockReset();
        toastMock.error.mockReset();
        refreshSpy.mockReset();
        locationAssignMock.mockReset();
    });

    it('self-hides on healthy + cold-start (n_user_projects = 0)', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: comparisonFixture({
                summary: 'healthy',
                nUserProjects: 0,
                features: [{ feature_id: 'row_count', status: 'ok' }],
            }),
        });
        const { container } = render(<ArchetypeComparisonPanel projectId={4} />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/4/archetype-comparison',
            );
        });
        // Wait for the loading effect to settle.
        await new Promise((r) => setTimeout(r, 0));
        expect(container.querySelector('.archetype-cmp')).toBeNull();
    });

    it('does NOT self-hide on healthy when the user has shipped projects', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: comparisonFixture({
                summary: 'healthy',
                nUserProjects: 2,
                features: [{ feature_id: 'row_count', status: 'ok' }],
            }),
        });
        render(<ArchetypeComparisonPanel projectId={4} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('archetype-comparison-panel'),
            ).toBeInTheDocument();
        });
    });

    it('renders the feature table with status badges + suggestion text', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: comparisonFixture({
                summary: 'below_cohort',
                features: [
                    {
                        feature_id: 'row_count',
                        status: 'below',
                        suggestion: 'Generate 80 more rows.',
                        suggested_action: {
                            kind: 'run_playbook',
                            params: {
                                mode: 'positives_paraphrase',
                                target_count: 80,
                            },
                        },
                    },
                ],
            }),
        });
        render(<ArchetypeComparisonPanel projectId={4} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('archetype-comparison-panel'),
            ).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('archetype-comparison-row-row_count'),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId('archetype-comparison-status-row_count'),
        ).toHaveTextContent('below cohort');
        // Generate-via-playbook CTA is rendered for run_playbook actions.
        expect(
            screen.getByTestId('archetype-comparison-action-row_count'),
        ).toHaveTextContent(/generate via playbook/i);
    });

    it('fires runPlaybookAsync + info toast + jobs refresh on run_playbook action', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: comparisonFixture({
                summary: 'below_cohort',
                features: [
                    {
                        feature_id: 'row_count',
                        status: 'below',
                        suggestion: 'Generate 80 more rows.',
                        suggested_action: {
                            kind: 'run_playbook',
                            params: {
                                mode: 'positives_paraphrase',
                                target_count: 80,
                            },
                        },
                    },
                ],
            }),
        });
        // runPlaybookAsync POSTs to /projects/4/synthetic/run-playbook?async_job=true
        apiMock.post.mockResolvedValueOnce({
            data: {
                id: 55,
                kind: 'synth_playbook',
                title: 'Synth · positives_paraphrase · 80 rows',
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
        render(<ArchetypeComparisonPanel projectId={4} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('archetype-comparison-action-row_count'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('archetype-comparison-action-row_count'),
        );
        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/4/synthetic/run-playbook?async_job=true',
                expect.objectContaining({
                    mode: 'positives_paraphrase',
                    target_count: 80,
                }),
            );
        });
        expect(toastMock.info).toHaveBeenCalledWith(
            expect.stringContaining('#55'),
            4000,
        );
        expect(refreshSpy).toHaveBeenCalled();
    });

    it('navigates on a navigate suggested_action', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: comparisonFixture({
                summary: 'below_cohort',
                features: [
                    {
                        feature_id: 'goldset_diversity',
                        status: 'below',
                        suggestion: 'Diversify your gold set.',
                        suggested_action: {
                            kind: 'navigate',
                            params: { target: 'data-studio-diversity' },
                        },
                    },
                ],
            }),
        });
        render(<ArchetypeComparisonPanel projectId={4} />);
        await waitFor(() => {
            expect(
                screen.getByTestId('archetype-comparison-action-goldset_diversity'),
            ).toBeInTheDocument();
        });
        await userEvent.click(
            screen.getByTestId('archetype-comparison-action-goldset_diversity'),
        );
        expect(locationAssignMock).toHaveBeenCalledWith(
            '/project/4/data-studio#diversity',
        );
    });

    it('silently renders nothing on 4xx / network failure', async () => {
        apiMock.get.mockRejectedValueOnce({ response: { status: 404 } });
        const { container } = render(<ArchetypeComparisonPanel projectId={4} />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        await new Promise((r) => setTimeout(r, 0));
        expect(container.querySelector('.archetype-cmp')).toBeNull();
    });

    it('renders the shared "pick a recipe first" CTA on RECIPE_REQUIRED 400', async () => {
        // Legacy NULL-recipe project: backend now returns a
        // structured 400 with error_code=RECIPE_REQUIRED. Panel must
        // mount the shared NoRecipeEmptyState (instead of silently
        // hiding like it does for other 4xx — empty cohort, network
        // failure, etc.).
        apiMock.get.mockRejectedValueOnce({
            response: {
                status: 400,
                data: {
                    detail: {
                        error_code: 'RECIPE_REQUIRED',
                        message:
                            'Project has no selected recipe — '
                            + "can't compare to an archetype.",
                    },
                },
            },
        });
        render(<ArchetypeComparisonPanel projectId={9} />);
        const cta = await screen.findByTestId(
            'archetype-comparison-recipe-required',
        );
        expect(cta.textContent).toMatch(/Pick a recipe first/);
        const link = cta.querySelector('a') as HTMLAnchorElement;
        const href = link.getAttribute('href') || '';
        expect(href.startsWith('/project/9/recipe-picker?')).toBe(true);
        expect(href).toMatch(/return_to=/);
    });

    it('keeps silent-hide on non-RECIPE_REQUIRED 400 (empty cohort)', async () => {
        // Cohort empty for this recipe — backend returns plain-string
        // detail like "empty_cohort:code-review". Panel keeps the
        // legacy silent-hide behavior.
        apiMock.get.mockRejectedValueOnce({
            response: {
                status: 400,
                data: { detail: 'empty_cohort:code-review' },
            },
        });
        const { container } = render(<ArchetypeComparisonPanel projectId={9} />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalled();
        });
        await new Promise((r) => setTimeout(r, 0));
        expect(
            screen.queryByTestId('archetype-comparison-recipe-required'),
        ).not.toBeInTheDocument();
        expect(container.querySelector('.archetype-cmp')).toBeNull();
    });
});
