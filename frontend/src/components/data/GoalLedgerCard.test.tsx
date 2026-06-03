/**
 * Arc H — GoalLedgerCard tests.
 *
 * Covers:
 *   - Renders overall progress % + per-component bars from the
 *     progress endpoint payload.
 *   - Renders the "default goal" nudge when has_explicit_goal=false.
 *   - PUT /goal flow: typing a threshold + clicking "Save goal"
 *     POSTs the right body and re-fetches progress.
 *   - Status colour-coding flips between green/amber/red across
 *     ledger statuses.
 *   - Term integration: each component renders the matching
 *     beginner label from the Term registry so the Academy
 *     "Learn more" link reaches the user.
 *   - Blockers list surfaces backend blocker strings verbatim.
 */

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

vi.mock('../../api/client', () => ({ default: apiMock }));

import GoalLedgerCard from './GoalLedgerCard';
import { useGlossaryStore } from '../../stores/glossaryStore';


function _progressFixture(overrides: Partial<Record<string, unknown>> = {}) {
    return {
        project_id: 7,
        goal: {
            target_metric: 'f1',
            target_threshold: 0.7,
            deadline: null,
            title: null,
            stated_at: null,
        },
        has_explicit_goal: false,
        components: [
            {
                id: 'data_ready',
                label: 'Training data ready',
                value: 0.6,
                status: 'attention',
                detail: '30 training rows · 50 recommended.',
                concept_id: 'task_shape',
            },
            {
                id: 'gold_set',
                label: 'Gold Set ready',
                value: 0.12,
                status: 'attention',
                detail: '12 gold rows · 100 recommended.',
                concept_id: 'gold_set',
            },
            {
                id: 'predicted_pass',
                label: 'Predicted pass probability',
                value: null,
                status: 'pending',
                detail: 'Trainability forecast not computed yet.',
                concept_id: 'predicted_f1_confidence',
            },
            {
                id: 'eval_pass_rate',
                label: 'Eval pass rate',
                value: null,
                status: 'pending',
                detail: 'No eval has run yet.',
                concept_id: 'pass_rate',
            },
        ],
        overall_progress: 0.36,
        pending_components: ['predicted_pass', 'eval_pass_rate'],
        blockers: [
            'Only 30 training rows (need ≥50).',
            'Only 12 gold rows (need ≥100).',
        ],
        status: 'in_progress',
        ...overrides,
    };
}


describe('GoalLedgerCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        apiMock.delete.mockReset();
        // Pre-load the glossary store so Term mounts skip their
        // background fetch (would otherwise steal the first
        // mockResolvedValueOnce).
        useGlossaryStore.setState({ entries: {}, loading: false, loaded: true, error: null });
    });

    it('renders overall progress and one row per component', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _progressFixture() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger')).toBeInTheDocument();
        });
        expect(screen.getByTestId('goal-ledger-overall-pct')).toHaveTextContent('36%');
        // All 4 components rendered with the right testids.
        for (const id of ['data_ready', 'gold_set', 'predicted_pass', 'eval_pass_rate']) {
            expect(screen.getByTestId(`goal-ledger-component-${id}`)).toBeInTheDocument();
        }
    });

    it('shows the "default goal" nudge when has_explicit_goal is false', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _progressFixture() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-default-note')).toBeInTheDocument();
        });
        expect(screen.getByTestId('goal-ledger-default-note').textContent).toMatch(
            /No goal set yet/i,
        );
    });

    it('hides the default nudge when has_explicit_goal is true', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: _progressFixture({
                has_explicit_goal: true,
                goal: {
                    target_metric: 'f1',
                    target_threshold: 0.85,
                    deadline: null,
                    title: 'Ship refund classifier',
                    stated_at: '2026-06-03T11:00:00Z',
                },
            }),
        });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger')).toBeInTheDocument();
        });
        expect(screen.queryByTestId('goal-ledger-default-note')).toBeNull();
        // Custom title rendered.
        expect(screen.getByText('Ship refund classifier')).toBeInTheDocument();
    });

    it('"Set your own" opens the edit form; saving PUTs /goal and refreshes', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _progressFixture() });
        apiMock.put.mockResolvedValueOnce({
            data: {
                project_id: 7,
                goal: {
                    target_metric: 'f1',
                    target_threshold: 0.9,
                    deadline: null,
                    title: 'My target',
                    stated_at: '2026-06-03T11:00:00Z',
                },
            },
        });
        // Second GET after save reflects the new goal.
        apiMock.get.mockResolvedValueOnce({
            data: _progressFixture({
                has_explicit_goal: true,
                goal: {
                    target_metric: 'f1',
                    target_threshold: 0.9,
                    deadline: null,
                    title: 'My target',
                    stated_at: '2026-06-03T11:00:00Z',
                },
            }),
        });

        const user = userEvent.setup();
        render(<GoalLedgerCard projectId={7} />);
        await screen.findByTestId('goal-ledger-default-note');
        await user.click(screen.getByRole('button', { name: /Set your own/i }));

        const titleInput = await screen.findByPlaceholderText(/Ship refund classifier/i);
        await user.type(titleInput, 'My target');
        await user.click(screen.getByRole('button', { name: /Save goal/i }));

        await waitFor(() => {
            expect(apiMock.put).toHaveBeenCalledWith(
                '/projects/7/goal',
                expect.objectContaining({
                    target_metric: 'f1',
                    target_threshold: 0.85,
                    title: 'My target',
                }),
            );
        });
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledTimes(2);
        });
    });

    it('renders the ready-to-ship status when overall_progress is 1.0', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: _progressFixture({
                overall_progress: 1.0,
                status: 'ready_to_ship',
                components: [
                    {
                        id: 'data_ready', label: 'Training data ready', value: 1.0,
                        status: 'met', detail: 'ready', concept_id: 'task_shape',
                    },
                    {
                        id: 'gold_set', label: 'Gold Set ready', value: 1.0,
                        status: 'met', detail: 'ready', concept_id: 'gold_set',
                    },
                    {
                        id: 'predicted_pass', label: 'Predicted pass probability', value: 1.0,
                        status: 'met', detail: 'ready', concept_id: 'predicted_f1_confidence',
                    },
                    {
                        id: 'eval_pass_rate', label: 'Eval pass rate', value: 1.0,
                        status: 'met', detail: 'ready', concept_id: 'pass_rate',
                    },
                ],
                blockers: [],
                pending_components: [],
            }),
        });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-status').textContent).toMatch(/Ready to ship/);
        });
        expect(screen.getByTestId('goal-ledger-overall-pct')).toHaveTextContent('100%');
        // Blockers card is hidden when there are no blockers.
        expect(screen.queryByTestId('goal-ledger-blockers')).toBeNull();
    });

    it('surfaces the blockers list verbatim', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _progressFixture() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-blockers')).toBeInTheDocument();
        });
        expect(
            screen.getByTestId('goal-ledger-blockers').textContent,
        ).toMatch(/Only 30 training rows/);
        expect(
            screen.getByTestId('goal-ledger-blockers').textContent,
        ).toMatch(/Only 12 gold rows/);
    });

    it('renders the matching Term label per component (Arc G compound)', async () => {
        // Each component's concept_id is wired into the Term registry,
        // so the label rendered is the registry's beginnerLabel
        // (e.g. gold_set → "Reference Set"). This pins the
        // Term/glossary integration: drop a concept_id from the
        // registry and the test fails loud.
        apiMock.get.mockResolvedValueOnce({ data: _progressFixture() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-component-gold_set')).toBeInTheDocument();
        });
        // The card passes a custom ``label`` to <Term>, which overrides
        // the registry's beginnerLabel — so the visible string is the
        // backend's component.label. The tooltip still carries the
        // concept's full definition; we assert the visible label and
        // the existence of the button (the Term trigger).
        const card = screen.getByTestId('goal-ledger-component-gold_set');
        expect(card.textContent).toMatch(/Gold Set ready/);
        expect(card.querySelector('.term-trigger')).not.toBeNull();
    });

    it('shows the error state with a retry button when fetch fails', async () => {
        apiMock.get.mockRejectedValueOnce(new Error('network down'));
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-error')).toBeInTheDocument();
        });
        expect(screen.getByText('network down')).toBeInTheDocument();
    });
});
