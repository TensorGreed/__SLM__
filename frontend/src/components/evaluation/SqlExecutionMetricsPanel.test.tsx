import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import SqlExecutionMetricsPanel, { SqlExecutionRowView, hasSqlExecution, isSqlExecRow } from './SqlExecutionMetricsPanel';
import { metricDisplayName, metricUnit } from './liftEvidence';

const BLOCK = {
    status: 'scored',
    exec_match_rate: 0.75,
    n_rows: 5,
    n_scored: 4,
    n_passed: 3,
    n_unscoreable: 1,
    n_blocked: 2,
    n_timeout: 1,
    n_over_cap: 0,
    n_refusal_rows: 1,
    refusal_correct_rate: 1,
    databases: ['main.db', 'alt.db'],
    order: 'auto',
    refusal_marker: 'NO_QUERY',
    errors_top: [{ message: 'main.db: results differ', count: 1 }],
};

describe('SqlExecutionMetricsPanel', () => {
    it('shows the headline, counts and the databases scored against', () => {
        render(<SqlExecutionMetricsPanel metrics={{ sql_execution: BLOCK, sql_execution_fingerprint: 'sql_exec/v1+0123456789abcdef' }} provenance="Run #3 · test · result 9" />);
        expect(screen.getByTestId('sxm-headline')).toHaveTextContent('75.0%');
        expect(screen.getByTestId('sxm-headline')).toHaveTextContent('3 of 4 scored rows');
        expect(screen.getByTestId('sxm-unscoreable')).toHaveTextContent('1');
        expect(screen.getByTestId('sxm-blocked')).toHaveTextContent('2');
        expect(screen.getByTestId('sxm-limits')).toHaveTextContent('1 / 0');
        expect(screen.getByTestId('sxm-refusal')).toHaveTextContent('100.0%');
        expect(screen.getByTestId('sxm-provenance')).toHaveTextContent('Run #3 · test · result 9');
        expect(screen.getByTestId('sxm-provenance')).toHaveTextContent('2 databases main.db, alt.db');
        expect(screen.getByTestId('sxm-provenance')).toHaveTextContent('scoring 89abcdef');
        expect(screen.getByTestId('sxm-errors')).toHaveTextContent('results differ');
    });

    it('explains a settings problem instead of a number', () => {
        render(<SqlExecutionMetricsPanel metrics={{ sql_execution: { status: 'config_error', error: "database path '../x.db' is outside the project data directory" } }} provenance="Run" />);
        expect(screen.getByTestId('sxm-config-error')).toHaveTextContent('outside the project data directory');
        expect(screen.queryByTestId('sxm-headline')).toBeNull();
    });

    it('renders a row verdict with per-database detail', () => {
        render(<SqlExecutionRowView row={{ row_sql_exec_pass: false, row_sql_exec: { pass: false, status: 'mismatch', ordered: true, error: 'alt.db: results differ', per_db: [{ db: 'main.db', pass: true, gold_rows: 2, pred_rows: 2 }, { db: 'alt.db', pass: false, gold_rows: 1, pred_rows: 3 }] } }} />);
        expect(screen.getByTestId('sxr-status')).toHaveTextContent('results differ');
        expect(screen.getByTestId('sxr-status').className).toContain('is-fail');
        expect(screen.getByTestId('sxr-per-db')).toHaveTextContent('alt.db: differs · answer key 1 rows · prediction 3 rows');
        expect(screen.getByTestId('sxr-error')).toHaveTextContent('alt.db: results differ');
    });

    it('labels an unscoreable row neutrally', () => {
        render(<SqlExecutionRowView row={{ row_sql_exec_pass: null, row_sql_exec: { pass: null, status: 'gold_error' } }} />);
        expect(screen.getByTestId('sxr-status')).toHaveTextContent('answer key failed: left out');
        expect(screen.getByTestId('sxr-status').className).not.toContain('is-');
    });

    it('type guards and lift labels', () => {
        expect(hasSqlExecution({ sql_execution: BLOCK })).toBe(true);
        expect(hasSqlExecution({})).toBe(false);
        expect(isSqlExecRow({ row_sql_exec: { pass: true } })).toBe(true);
        expect(isSqlExecRow({})).toBe(false);
        expect(metricDisplayName('exec_match_rate')).toBe('execution match rate');
        expect(metricUnit('exec_match_rate')).toBe('execution match rate');
    });
});
