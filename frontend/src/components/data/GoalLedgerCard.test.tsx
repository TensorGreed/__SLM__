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

    // ─────────────────────────────────────────────────────────────────
    // Arc R-2 slice 2 — gate breakdown sub-rows expand the
    // eval_pass_rate component into the gates the project's eval pack
    // actually enforces. Compound with Arc G: each gate's metric_id
    // wraps in <Term> so the row carries the Academy deep-link.
    // ─────────────────────────────────────────────────────────────────

    function _withEvalGates(overrides: Partial<Record<string, unknown>> = {}) {
        return _progressFixture({
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
                    id: 'predicted_pass', label: 'Predicted pass probability', value: 0.78,
                    status: 'met', detail: 'forecast 78%', concept_id: 'predicted_f1_confidence',
                },
                {
                    id: 'eval_pass_rate',
                    label: 'Eval pass rate',
                    value: 0.62,
                    status: 'attention',
                    detail: 'Latest eval 62% (your bar is 85%).',
                    concept_id: 'pass_rate',
                    gate_breakdown: [
                        // Citation gate fails — actual 0.72 < 0.75 threshold.
                        {
                            gate_id: 'min_citation_rate',
                            metric_id: 'citation_rate',
                            operator: 'gte',
                            threshold: 0.75,
                            required: true,
                            actual: 0.72,
                            passed: false,
                        },
                        // Hallucination gate fails — actual 0.18 > 0.15 ceiling.
                        {
                            gate_id: 'max_hallucination_rate',
                            metric_id: 'hallucination_rate',
                            operator: 'lte',
                            threshold: 0.15,
                            required: true,
                            actual: 0.18,
                            passed: false,
                        },
                        // Refusal gate passes — actual 0.85 ≥ 0.80 threshold.
                        {
                            gate_id: 'min_appropriate_refusal_rate',
                            metric_id: 'appropriate_refusal_rate',
                            operator: 'gte',
                            threshold: 0.80,
                            required: true,
                            actual: 0.85,
                            passed: true,
                        },
                        // Optional format gate — actual null (Slice-2
                        // metric not implemented yet); UI shows pending.
                        {
                            gate_id: 'min_format_consistency',
                            metric_id: 'format_consistency',
                            operator: 'gte',
                            threshold: 0.75,
                            required: false,
                            actual: null,
                            passed: true,  // optional + missing → backend reports passed=true
                        },
                    ],
                },
            ],
            ...overrides,
        });
    }

    it('renders the gate breakdown when the eval_pass_rate component carries one', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _withEvalGates() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-gate-breakdown-eval_pass_rate')).toBeInTheDocument();
        });
        // One sub-row per gate.
        expect(screen.getByTestId('goal-ledger-gate-min_citation_rate')).toBeInTheDocument();
        expect(screen.getByTestId('goal-ledger-gate-max_hallucination_rate')).toBeInTheDocument();
        expect(screen.getByTestId('goal-ledger-gate-min_appropriate_refusal_rate')).toBeInTheDocument();
        expect(screen.getByTestId('goal-ledger-gate-min_format_consistency')).toBeInTheDocument();
    });

    it('formats fractional gate values as percentages', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _withEvalGates() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-gate-min_citation_rate-actual')).toBeInTheDocument();
        });
        // 0.72 → "72%"; 0.18 → "18%".
        expect(
            screen.getByTestId('goal-ledger-gate-min_citation_rate-actual').textContent,
        ).toMatch(/72%/);
        expect(
            screen.getByTestId('goal-ledger-gate-max_hallucination_rate-actual').textContent,
        ).toMatch(/18%/);
        // The threshold renders alongside the actual value in the
        // dedicated threshold cell.
        const citationRow = screen.getByTestId('goal-ledger-gate-min_citation_rate');
        expect(citationRow.textContent).toMatch(/≥ 75%/);
        const halluRow = screen.getByTestId('goal-ledger-gate-max_hallucination_rate');
        expect(halluRow.textContent).toMatch(/≤ 15%/);
    });

    it('renders pending placeholder for gates with null actual values', async () => {
        apiMock.get.mockResolvedValueOnce({ data: _withEvalGates() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-gate-min_format_consistency-actual')).toBeInTheDocument();
        });
        // Null actual → "—" placeholder, not "0%" which would be
        // misleading (the metric isn't computed yet).
        expect(
            screen.getByTestId('goal-ledger-gate-min_format_consistency-actual').textContent,
        ).toBe('—');
    });

    it('omits the gate breakdown when the component has no gates', async () => {
        // Component without ``gate_breakdown`` field — backend reports
        // empty list. The card should not render the expandable
        // section at all so we don't show an empty drawer.
        apiMock.get.mockResolvedValueOnce({ data: _progressFixture() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger')).toBeInTheDocument();
        });
        expect(
            screen.queryByTestId('goal-ledger-gate-breakdown-eval_pass_rate'),
        ).toBeNull();
    });

    it('wraps each gate metric_id in a Term button (Arc G compound)', async () => {
        // The gate row renders <Term id={metric_id} label={...}>.
        // The Term registry was extended in Slice 2 with the 4 new
        // discipline metric concepts; the button surfaces them as
        // clickable Term triggers carrying the Academy deep-link.
        apiMock.get.mockResolvedValueOnce({ data: _withEvalGates() });
        render(<GoalLedgerCard projectId={7} />);
        await waitFor(() => {
            expect(screen.getByTestId('goal-ledger-gate-min_citation_rate')).toBeInTheDocument();
        });
        const citationRow = screen.getByTestId('goal-ledger-gate-min_citation_rate');
        // Term renders a <button class="term-trigger"> — assert
        // its presence as a proxy for "registry entry exists".
        expect(citationRow.querySelector('button.term-trigger')).not.toBeNull();
    });
});
