import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import FieldMatchMetricsPanel, { hasFieldMatchBreakdown, type FieldMatchMetrics } from './FieldMatchMetricsPanel';

const METRICS: FieldMatchMetrics = {
    scoring_mode: 'field_match',
    null_precision: 0.5,
    null_recall: 1,
    hallucinated: 3,
    abstained_wrongly: 1,
    schema_valid_rate: 0.9,
    per_field: {
        damage: {
            em: 0.75,
            null_precision: 0.5,
            null_recall: null,
            hallucinated: 2,
            out_of_vocab_rate: 0.125,
            type_violation_rate: null,
        },
        injuries_fatal: {
            em: 1,
            null_precision: 1,
            null_recall: 1,
            hallucinated: 0,
            out_of_vocab_rate: null,
            type_violation_rate: 0.25,
        },
    },
};

describe('FieldMatchMetricsPanel', () => {
    it('renders aggregate and per-field metrics with provenance', () => {
        render(<FieldMatchMetricsPanel metrics={METRICS} provenance="Run #21 · gold_dev · result 1001" />);
        expect(screen.getByTestId('field-match-provenance')).toHaveTextContent('Run #21 · gold_dev · result 1001');
        expect(screen.getByTestId('field-match-schema-valid')).toHaveTextContent('90.0%');
        expect(screen.getByTestId('field-match-null-precision')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('field-match-hallucinated')).toHaveTextContent('3');
        expect(screen.getByTestId('field-match-damage-oov')).toHaveTextContent('12.5%');
        expect(screen.getByTestId('field-match-injuries_fatal-type')).toHaveTextContent('25.0%');
        expect(screen.getByTestId('field-match-damage-hallucinated')).toHaveTextContent('2');
    });

    it('shows a dash, not 0%, for undefined rates', () => {
        render(<FieldMatchMetricsPanel metrics={METRICS} provenance="x" />);
        expect(screen.getByTestId('field-match-damage-null-recall')).toHaveTextContent('—');
        expect(screen.getByTestId('field-match-damage-type')).toHaveTextContent('—');
        expect(screen.getByTestId('field-match-injuries_fatal-oov')).toHaveTextContent('—');
    });

    it('shows a dash for exact match when a field has no support', () => {
        const metrics: FieldMatchMetrics = {
            ...METRICS,
            per_field: { ghost: { em: null, f1: null, support: 0, null_precision: null, null_recall: null } },
        };
        render(<FieldMatchMetricsPanel metrics={metrics} provenance="x" />);
        expect(screen.getByTestId('field-match-row-ghost').querySelectorAll('td')[0]).toHaveTextContent('—');
    });

    it('detects a typed field_match breakdown', () => {
        expect(hasFieldMatchBreakdown(METRICS)).toBe(true);
        expect(hasFieldMatchBreakdown({ scoring_mode: 'span_set', per_field: {} })).toBe(false);
        // Older results (no null metrics) fall back to the existing view.
        expect(hasFieldMatchBreakdown({ per_field: { a: { em: 1, f1: 1, support: 3 } } })).toBe(false);
        expect(hasFieldMatchBreakdown(undefined)).toBe(false);
    });
});
