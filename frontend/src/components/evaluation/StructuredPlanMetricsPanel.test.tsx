import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import StructuredPlanMetricsPanel, { isStructuredPlanMetrics, isStructuredPlanRow } from './StructuredPlanMetricsPanel';
import { metricDisplayName } from './liftEvidence';

describe('StructuredPlanMetricsPanel', () => {
    it('detects the scoring mode', () => {
        expect(isStructuredPlanMetrics({ scoring_mode: 'structured_plan' })).toBe(true);
        expect(isStructuredPlanMetrics({ scoring_mode: 'remediation' })).toBe(false);
        expect(isStructuredPlanRow({ row_structured_plan_pass: false })).toBe(true);
        expect(isStructuredPlanRow({ scoring_mode: 'field_match' })).toBe(false);
    });

    it('says what was not evaluated when no schema, pointers or rubric exist', () => {
        render(<StructuredPlanMetricsPanel
            metrics={{ scoring_mode: 'structured_plan', schema_missing: true, structured_plan_pass_rate: 0.5, total: 4, correct: 2, json_validity_rate: 1, schema_valid_rate: 1 }}
            provenance="Run #3"
        />);
        expect(screen.getByTestId('sp-pass')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('sp-schema-valid')).toHaveTextContent('not evaluated');
        expect(screen.getByTestId('sp-field-match')).toHaveTextContent('not evaluated');
        expect(screen.getByTestId('sp-rubric-pass')).toHaveTextContent('not run');
        expect(screen.getByTestId('sp-provenance')).toHaveTextContent('no schema · no rubric · no judge');
    });

    it('flags a configured rubric that had no judge', () => {
        render(<StructuredPlanMetricsPanel
            metrics={{ scoring_mode: 'structured_plan', rubric: { status: 'no_judge', id: 'plan-quality', sha256: 'ffeeddccbbaa' } }}
            provenance="Run #3"
        />);
        expect(screen.getByTestId('sp-no-judge')).toBeInTheDocument();
        expect(screen.getByTestId('sp-rubric-pass')).toHaveTextContent('no judge');
        expect(screen.queryByTestId('sp-rubric')).not.toBeInTheDocument();
    });

    it('warns when the schema changed after the split was prepared', () => {
        const { rerender } = render(<StructuredPlanMetricsPanel
            metrics={{ scoring_mode: 'structured_plan', schema_changed_since_prep: true, prep_schema_sha256: 'aabbccddeeff0011' }}
            provenance="Run #3"
        />);
        expect(screen.getByTestId('sp-schema-changed')).toHaveTextContent('aabbccdd');
        rerender(<StructuredPlanMetricsPanel metrics={{ scoring_mode: 'structured_plan' }} provenance="Run #3" />);
        expect(screen.queryByTestId('sp-schema-changed')).not.toBeInTheDocument();
    });

    it('labels the headline metric for lift', () => {
        expect(metricDisplayName('structured_plan_pass_rate')).toBe('plan pass rate');
    });
});
