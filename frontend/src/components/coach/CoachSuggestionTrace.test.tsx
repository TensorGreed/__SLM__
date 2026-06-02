/**
 * Arc 4 — Coach decision-trace disclosure.
 *
 * Pre-Arc-4 the suggestion's ``context`` field was a backend-emitted
 * dict that the strip ignored. This component renders a "why?"
 * disclosure that expands to show the signal values + (when set)
 * the ``rule_id`` of the decision rule that matched. Tests pin:
 *
 *   - Renders nothing when neither rule_id nor context is set
 *     (pre-enriched suggestions don't get a misleading "Trace"
 *     button on an empty trace).
 *   - Shows the "why?" toggle by default; expands inline on click.
 *   - Renders signal entries as key/value pairs, formatting
 *     numbers / booleans / arrays / nested objects readably.
 *   - rule_id renders as the trace's first row when present.
 *   - schema_aware_backend is filtered out (already rendered as
 *     its own chip on the action column).
 *   - Close button collapses back to the toggle.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import CoachSuggestionTrace from './CoachSuggestionTrace';
import type { CoachSuggestion } from '../../api/coach';


function _suggestion(overrides: Partial<CoachSuggestion> = {}): CoachSuggestion {
    return {
        id: 'test-id',
        title: 'A suggestion',
        body: 'do the thing',
        severity: 'info',
        action: {
            kind: 'navigate',
            label: 'Open',
            params: { target: 'data' },
        },
        ...overrides,
    };
}


describe('CoachSuggestionTrace', () => {
    it('renders nothing when neither rule_id nor context is set', () => {
        // Pre-enriched suggestions (no decision-trace fields) must
        // not show an empty "why?" affordance. Hide entirely.
        const { container } = render(
            <CoachSuggestionTrace suggestion={_suggestion()} />,
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders the "why?" toggle when context is present', async () => {
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: { gold_row_count: 12, threshold: 50 },
                })}
            />,
        );
        expect(
            screen.getByTestId('coach-suggestion-test-id-why'),
        ).toBeInTheDocument();
        // Trace not expanded yet — signal entries hidden.
        expect(
            screen.queryByTestId('coach-suggestion-test-id-trace'),
        ).toBeNull();
    });

    it('expands the trace on click and lists signal entries', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: {
                        gold_row_count: 12,
                        comfortable_threshold: 200,
                        thin_threshold: 50,
                    },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        const trace = screen.getByTestId('coach-suggestion-test-id-trace');
        expect(trace).toBeInTheDocument();
        // Each context key gets a row keyed for selector targeting.
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-gold_row_count',
            ),
        ).toHaveTextContent('12');
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-comfortable_threshold',
            ),
        ).toHaveTextContent('200');
    });

    it('renders rule_id at the top of the trace when present', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    rule_id: 'gold-row-count.thin',
                    context: { gold_row_count: 12 },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        const ruleRow = screen.getByTestId(
            'coach-suggestion-test-id-rule',
        );
        expect(ruleRow).toHaveTextContent('gold-row-count.thin');
    });

    it('filters out schema_aware_backend from the trace listing', async () => {
        // The schema-aware backend pin is already rendered as a
        // pill chip on the action column. Surfacing it again in
        // the trace would double up the same signal.
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: {
                        underrepresented_classes: ['minority_a'],
                        schema_aware_backend: 'vllm',
                    },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        // The other signal renders.
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-underrepresented_classes',
            ),
        ).toBeInTheDocument();
        // schema_aware_backend NOT rendered in the trace.
        expect(
            screen.queryByTestId(
                'coach-suggestion-test-id-signal-schema_aware_backend',
            ),
        ).toBeNull();
    });

    it('formats numeric values with up to 3 decimal places', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: { drift_score: 0.123456 },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-drift_score',
            ),
        ).toHaveTextContent('0.123');
    });

    it('formats arrays with element previews + tail count', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: {
                        below_feature_ids: ['a', 'b', 'c', 'd', 'e', 'f'],
                    },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        // 6 elements → first 4 shown + "…+2" tail count.
        const cell = screen.getByTestId(
            'coach-suggestion-test-id-signal-below_feature_ids',
        );
        expect(cell.textContent).toMatch(/a.*b.*c.*d.*\+2/);
    });

    it('renders booleans readably', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: { is_failing: true, has_clusters: false },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-is_failing',
            ),
        ).toHaveTextContent('true');
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-has_clusters',
            ),
        ).toHaveTextContent('false');
    });

    it('handles null / undefined values without crashing', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: {
                        dominant_feature_id: null,
                        sweep_run_id: undefined,
                    },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        // ``—`` is the en-dash placeholder for null/undefined.
        expect(
            screen.getByTestId(
                'coach-suggestion-test-id-signal-dominant_feature_id',
            ),
        ).toHaveTextContent('—');
    });

    it('collapses back to the toggle when close is clicked', async () => {
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    context: { gold_row_count: 12 },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        expect(
            screen.getByTestId('coach-suggestion-test-id-trace'),
        ).toBeInTheDocument();
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-trace-close'),
        );
        // Back to the toggle, trace gone.
        expect(
            screen.getByTestId('coach-suggestion-test-id-why'),
        ).toBeInTheDocument();
        expect(
            screen.queryByTestId('coach-suggestion-test-id-trace'),
        ).toBeNull();
    });

    it('renders trace with only rule_id (no context entries)', async () => {
        // Edge case — a suggestion enriched with rule_id but
        // whose context is empty after filtering. The trace
        // should still surface with just the rule row, no
        // signals dl.
        const user = userEvent.setup();
        render(
            <CoachSuggestionTrace
                suggestion={_suggestion({
                    rule_id: 'standalone-rule',
                    context: { schema_aware_backend: 'vllm' },
                })}
            />,
        );
        await user.click(
            screen.getByTestId('coach-suggestion-test-id-why'),
        );
        expect(
            screen.getByTestId('coach-suggestion-test-id-rule'),
        ).toHaveTextContent('standalone-rule');
    });
});
