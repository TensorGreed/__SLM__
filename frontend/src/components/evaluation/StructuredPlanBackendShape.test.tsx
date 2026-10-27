/**
 * Feeds the panel and the row view a payload shaped by the G6 contract: the metrics follow
 * `structured_plan_scoring.score_structured_plan` + `finalize_with_rubric` and the rows follow the
 * `predictions_preview` key whitelist. Reconcile the fixture with a captured backend payload when
 * either side changes shape.
 */
import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import fixture from './structuredPlanBackendShape.fixture.json';
import StructuredPlanMetricsPanel, { StructuredPlanRowView, type StructuredPlanMetrics, type StructuredPlanRow } from './StructuredPlanMetricsPanel';

const METRICS = fixture.metrics as unknown as StructuredPlanMetrics;
const ROWS = fixture.predictions_preview as unknown as StructuredPlanRow[];

describe('structured-plan components against the contract payload shape', () => {
    it('the metrics panel reads schema errors, pointers and the rubric block', () => {
        render(<StructuredPlanMetricsPanel metrics={METRICS} provenance="Run #1" />);
        expect(screen.getByTestId('sp-pass')).toHaveTextContent('33.3%');
        expect(screen.getByTestId('sp-schema-valid')).toHaveTextContent('66.7%');
        expect(screen.getByTestId('sp-provenance')).toHaveTextContent('schema plan.schema.json (sha abcdef01)');
        expect(screen.getByTestId('sp-provenance')).toHaveTextContent('rubric plan-quality (sha 12345678)');
        expect(screen.getByTestId('sp-provenance')).toHaveTextContent('judge ollama:llama3.3:70b');
        expect(screen.getByTestId('sp-error-0')).toHaveTextContent('/steps/*/action');
        expect(screen.getByTestId('sp-pointer-/environment')).toHaveTextContent('66.7%');
        const crit = screen.getByTestId('sp-criterion-safe_order');
        expect(within(crit).getByText('critical')).toBeInTheDocument();
        expect(crit).toHaveTextContent('3.00');
        expect(screen.getByTestId('sp-rubric-line')).toHaveTextContent('2 judged, 1 unjudged');
    });

    it('a passing row shows pointer results and rubric scores', () => {
        render(<StructuredPlanRowView row={ROWS[0]} />);
        expect(screen.getByTestId('sp-chip-pass')).toHaveTextContent('pass');
        expect(screen.getByTestId('sp-chip-rubric')).toHaveTextContent('rubric passed');
        expect(screen.getByTestId('sp-row-pointer-/steps/*/action')).toHaveTextContent('create, update');
        expect(screen.getByTestId('sp-row-criterion-clarity')).toHaveTextContent('5');
    });

    it('a failing row shows schema errors and the critical criterion', () => {
        render(<StructuredPlanRowView row={ROWS[1]} />);
        expect(screen.getByTestId('sp-row-errors')).toHaveTextContent('/steps/0/action');
        expect(screen.getByTestId('sp-row-critical')).toHaveTextContent('safe_order');
        expect(screen.getByTestId('sp-chip-schema')).toHaveTextContent('schema invalid');
    });

    it('an unjudged row says so and an acceptable-set pointer is labelled', () => {
        render(<StructuredPlanRowView row={ROWS[2]} />);
        expect(screen.getByTestId('sp-chip-rubric')).toHaveTextContent('rubric unjudged');
        expect(screen.getByTestId('sp-row-pointer-/environment')).toHaveTextContent('(any of)');
    });
});
