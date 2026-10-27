/**
 * Feeds the tool-call components a payload shaped by the G7 contract: the `tools` block follows
 * `tool_call_scoring.score_tool_block` and the rows follow the `row_tool_results` whitelist entry of
 * `predictions_preview`. Reconcile the fixture with a captured backend payload when either side changes.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import fixture from './toolCallBackendShape.fixture.json';
import StructuredPlanMetricsPanel, { type StructuredPlanMetrics } from './StructuredPlanMetricsPanel';
import { ToolCallRowView, type ToolCallRow } from './ToolCallMetricsPanel';

const METRICS = fixture.metrics as unknown as StructuredPlanMetrics;
const ROWS = fixture.predictions_preview as unknown as ToolCallRow[];

describe('tool-call components against the contract payload shape', () => {
    it('the structured-plan panel renders the tools block with provenance, tables and dry run', () => {
        render(<StructuredPlanMetricsPanel metrics={METRICS} provenance="Run #1" />);
        expect(screen.getByTestId('tcm-provenance')).toHaveTextContent('catalog version a1b2c3d4e5f6');
        expect(screen.getByTestId('tcm-provenance')).toHaveTextContent('source mcp_stdio');
        expect(screen.getByTestId('tcm-valid')).toHaveTextContent('75.0%');
        expect(screen.getByTestId('tcm-args')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('tcm-tool-delete_service')).toHaveTextContent('1');
        expect(screen.getByTestId('tcm-unknown-restart_service')).toHaveTextContent('2 times');
        expect(screen.getByTestId('tcm-errors')).toHaveTextContent('/tier');
        expect(screen.getByTestId('tcm-dry-run')).toHaveTextContent('3 read-only calls executed, 2 succeeded, 1 failed, 2 skipped');
        expect(screen.getByTestId('tcm-provenance')).toHaveTextContent('flags 0f1e2d3c4b5a');
    });

    it('a passing row shows the call badges and the dry-run result', () => {
        render(<ToolCallRowView row={ROWS[0]} />);
        expect(screen.getByTestId('tcr-chip-pass')).toHaveTextContent('tool checks pass');
        expect(screen.getByTestId('tcr-call-0')).toHaveTextContent('get_service');
        expect(screen.getByTestId('tcr-call-0-dry')).toHaveTextContent('ok');
    });

    it('a failing row shows argument errors, the unknown tool and the row errors', () => {
        render(<ToolCallRowView row={ROWS[1]} />);
        expect(screen.getByTestId('tcr-chip-valid')).toHaveTextContent('call invalid');
        expect(screen.getByTestId('tcr-call-0-err-0')).toHaveTextContent('/tier');
        expect(screen.getByTestId('tcr-call-1')).toHaveTextContent('restart_service');
        expect(screen.getByTestId('tcr-errors')).toHaveTextContent('unknown tool: restart_service');
        expect(screen.queryByTestId('tcr-chip-order')).toBeNull();
    });

    it('a row scored without a catalog shows the pick, arguments and pass chips', () => {
        render(<ToolCallRowView row={{ scoring_mode: 'tool_call', row_tool_selection_correct: true, row_tool_args_match: false, row_tool_call_pass: false }} />);
        expect(screen.getByTestId('tcr-chip-selection')).toHaveTextContent('right tool picked');
        expect(screen.getByTestId('tcr-chip-args')).toHaveTextContent('arguments differ');
        expect(screen.getByTestId('tcr-chip-call-pass')).toHaveTextContent('call fails');
    });
});
