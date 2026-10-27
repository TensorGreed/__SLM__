import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ToolCallMetricsPanel, { hasToolsBlock, isToolCallMetrics, isToolCallRow } from './ToolCallMetricsPanel';
import { metricDisplayName, metricUnit } from './liftEvidence';

describe('ToolCallMetricsPanel', () => {
    it('shows the headline and says so when no catalog was set', () => {
        render(<ToolCallMetricsPanel metrics={{ scoring_mode: 'tool_call', tool_call_pass_rate: 0.8, tool_selection_accuracy: 0.9, total: 10, correct: 8 }} provenance="Run #3 · gold_dev · result 5" />);
        expect(screen.getByTestId('tcm-headline')).toHaveTextContent('80.0%');
        expect(screen.getByTestId('tcm-headline')).toHaveTextContent('8 of 10 rows');
        expect(screen.getByTestId('tcm-selection')).toHaveTextContent('90.0%');
        expect(screen.getByTestId('tcm-no-catalog')).toBeInTheDocument();
    });

    it('renders not applicable for null rates and the dry-run-off line', () => {
        render(<ToolCallMetricsPanel metrics={{ scoring_mode: 'tool_call', tool_call_pass_rate: 1, total: 1, correct: 1, tools: { catalog_version: 'abc', tool_call_valid_rate: 1, plan_order_valid_rate: null, dry_run: { enabled: false } } }} provenance="Run" />);
        expect(screen.getByTestId('tcm-order')).toHaveTextContent('not applicable');
        expect(screen.getByTestId('tcm-dry-run')).toHaveTextContent('Dry run was off');
    });

    it('type guards and lift labels', () => {
        expect(isToolCallMetrics({ scoring_mode: 'tool_call' })).toBe(true);
        expect(isToolCallMetrics({ scoring_mode: 'structured_plan' })).toBe(false);
        expect(hasToolsBlock({ tools: { catalog_version: 'x' } })).toBe(true);
        expect(hasToolsBlock({})).toBe(false);
        expect(isToolCallRow({ row_tool_results: { calls: [] } })).toBe(true);
        expect(isToolCallRow({})).toBe(false);
        expect(metricDisplayName('tool_call_pass_rate')).toBe('tool-call pass rate');
        expect(metricUnit('tool_call_valid_rate')).toBe('valid tool-call rate');
    });
});
