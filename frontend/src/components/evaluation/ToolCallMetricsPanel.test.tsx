import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ToolCallMetricsPanel, { ToolCallRowView, hasToolsBlock, isToolCallMetrics, isToolCallRow, rowsScoredText, toolFlagsText } from './ToolCallMetricsPanel';
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
        expect(metricDisplayName('tool_call_strict_pass_rate')).toBe('strict tool-call pass rate');
    });

    it('shows the strict score beside the default one only when it was computed', () => {
        const { rerender } = render(<ToolCallMetricsPanel metrics={{ scoring_mode: 'tool_call', tool_call_pass_rate: 1, total: 3, correct: 3 }} provenance="Run" />);
        expect(screen.queryByTestId('tcm-strict')).toBeNull();
        rerender(<ToolCallMetricsPanel metrics={{ scoring_mode: 'tool_call', tool_call_pass_rate: 1, tool_call_strict_pass_rate: 2 / 3, total: 3, correct: 3 }} provenance="Run" />);
        expect(screen.getByTestId('tcm-headline')).toHaveTextContent('100.0%');
        expect(screen.getByTestId('tcm-strict')).toHaveTextContent('66.7%');
        expect(screen.getByText(/extra arguments count as wrong/)).toBeInTheDocument();
    });

    it('row view marks extra arguments under strict scoring', () => {
        render(<ToolCallRowView row={{ scoring_mode: 'tool_call', row_tool_call_pass: true, row_tool_selection_correct: true, row_tool_args_match: true, row_tool_args_strict_match: false }} />);
        expect(screen.getByTestId('tcr-chip-strict')).toHaveTextContent('extra arguments (strict)');
    });

    it('shows the eval-time tool flags and a capped row count', () => {
        render(<ToolCallMetricsPanel
            metrics={{ scoring_mode: 'tool_call', tool_call_pass_rate: 0.5, total: 400, correct: 200,
                tool_flags: { allowed_count: 12, tool_count: 30, catalog_fingerprint: 'abc+def', matches_training: false, training_allowed_count: 18 } }}
            provenance="Run #4 · test · result 9" rowsAvailable={1132} />);
        expect(screen.getByTestId('tcm-flags')).toHaveTextContent('Prompted and scored with 12 of 30 catalog tools allowed (catalog + flags abc+def)');
        expect(screen.getByTestId('tcm-flags')).toHaveTextContent('differs from the tool list the training prompts carried (18 tools)');
        expect(screen.getByTestId('tcm-rows')).toHaveTextContent('Scored 400 of 1,132 rows');
    });

    it('flag and row helpers stay quiet when there is nothing to say', () => {
        expect(toolFlagsText(null)).toBeNull();
        expect(toolFlagsText({ allowed_count: 3, tool_count: 3, matches_training: true })).toContain('same tool list as the training prompts');
        expect(rowsScoredText(100, 100)).toBeNull();
        expect(rowsScoredText(100, null)).toBeNull();
    });
});
