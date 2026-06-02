/**
 * Decision-trace disclosure for a Coach suggestion (Arc 4).
 *
 * Pre-Arc-4 the Coach emitted suggestions with a ``context`` field
 * carrying signal values, but the strip never rendered it — the
 * field's docstring literally said "Currently unused by the strip
 * but reserved." Users debugged Coach by reading the
 * ``coach_service.py`` source. This component closes that gap with
 * a small "Why?" disclosure that expands to show the signals the
 * rule observed and (when the backend emits it) the rule_id that
 * matched.
 *
 * Renders nothing when neither rule_id nor context is set.
 */

import { useState } from 'react';

import type { CoachSuggestion } from '../../api/coach';


interface CoachSuggestionTraceProps {
    suggestion: CoachSuggestion;
}


// Format helper — render a context value as a readable string,
// handling the common shapes (number, boolean, string, list,
// nested object) without crashing on anything weird. Coach
// signal values are typically scalars or short arrays; nested
// objects exist (e.g., archetype drift carries below_feature
// arrays of objects) but are rare.
function _formatValue(value: unknown): string {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'number') {
        // Round numeric values to keep the trace readable. Coach
        // signals are typically counts (integers) or ratios
        // (floats); 3 decimal places is enough for either.
        if (Number.isInteger(value)) return String(value);
        return value.toFixed(3);
    }
    if (typeof value === 'string') {
        // Truncate long strings to keep the trace tight. Full
        // strings still available in dev tools if the user really
        // needs them.
        if (value.length > 80) return `${value.slice(0, 77)}…`;
        return value;
    }
    if (Array.isArray(value)) {
        if (value.length === 0) return '[]';
        if (value.length > 4) {
            return `[${value.slice(0, 4).map(_formatValue).join(', ')}, …+${value.length - 4}]`;
        }
        return `[${value.map(_formatValue).join(', ')}]`;
    }
    if (typeof value === 'object') {
        // Compact JSON for nested objects. Truncated if long.
        try {
            const json = JSON.stringify(value);
            if (json.length > 80) return `${json.slice(0, 77)}…`;
            return json;
        } catch {
            return '<object>';
        }
    }
    return String(value);
}


export default function CoachSuggestionTrace({
    suggestion,
}: CoachSuggestionTraceProps) {
    const [expanded, setExpanded] = useState(false);

    const ruleId = suggestion.rule_id;
    const context = suggestion.context ?? {};
    const contextEntries = Object.entries(context).filter(
        // Skip schema_aware_backend — already rendered as its own
        // pill chip on the action column (CoachSuggestion.tsx
        // line ~454). Surfacing it in the trace too would
        // double-up the same signal.
        ([key]) => key !== 'schema_aware_backend',
    );
    const hasTrace = Boolean(ruleId) || contextEntries.length > 0;
    if (!hasTrace) return null;

    if (!expanded) {
        return (
            <button
                type="button"
                className="coach-suggestion-trace-toggle"
                onClick={() => setExpanded(true)}
                data-testid={`coach-suggestion-${suggestion.id}-why`}
                title={
                    'Show the signal values that triggered this '
                    + 'Coach suggestion'
                }
            >
                why?
            </button>
        );
    }

    return (
        <div
            className="coach-suggestion-trace"
            data-testid={`coach-suggestion-${suggestion.id}-trace`}
        >
            <div className="coach-suggestion-trace-header">
                <span className="coach-suggestion-trace-label">
                    Decision trace
                </span>
                <button
                    type="button"
                    className="coach-suggestion-trace-close"
                    onClick={() => setExpanded(false)}
                    data-testid={
                        `coach-suggestion-${suggestion.id}-trace-close`
                    }
                    aria-label="Hide decision trace"
                >
                    ×
                </button>
            </div>
            {ruleId && (
                <div
                    className="coach-suggestion-trace-rule"
                    data-testid={
                        `coach-suggestion-${suggestion.id}-rule`
                    }
                >
                    <span className="coach-suggestion-trace-key">
                        rule
                    </span>
                    <span className="coach-suggestion-trace-value">
                        {ruleId}
                    </span>
                </div>
            )}
            {contextEntries.length > 0 && (
                <dl className="coach-suggestion-trace-signals">
                    {contextEntries.map(([key, value]) => (
                        <div
                            key={key}
                            className="coach-suggestion-trace-signal"
                        >
                            <dt className="coach-suggestion-trace-key">
                                {key}
                            </dt>
                            <dd
                                className="coach-suggestion-trace-value"
                                data-testid={
                                    `coach-suggestion-${suggestion.id}`
                                    + `-signal-${key}`
                                }
                            >
                                {_formatValue(value)}
                            </dd>
                        </div>
                    ))}
                </dl>
            )}
        </div>
    );
}
