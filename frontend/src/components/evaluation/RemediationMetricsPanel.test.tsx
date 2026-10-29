import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import RemediationMetricsPanel, {
    RemediationRowView,
    isRemediationMetrics,
    isRemediationRow,
    passageHeader,
} from './RemediationMetricsPanel';
import { metricDisplayName } from './liftEvidence';

const METRICS = {
    scoring_mode: 'remediation',
    remediation_pass_rate: 0.5, total: 8, correct: 4,
    patch_applies_rate: 0.75, code_parses_rate: 0.8, code_checkable_count: 5,
    issue_resolved_rate: 0.6, new_findings_rate: 0.1, detector_evaluated_count: 6,
    citations_valid_rate: 0.9, citation_rate: 0.5,
    json_validity_rate: 0.95, malformed_rate: 0.05,
    patch_reasons: { context_mismatch: 2, no_op: 1 },
    judge_correct: 0.75,
    skipped_non_remediation_rows: 3,
};

describe('RemediationMetricsPanel', () => {
    it('detects the scoring mode', () => {
        expect(isRemediationMetrics({ scoring_mode: 'remediation' })).toBe(true);
        expect(isRemediationMetrics({ scoring_mode: 'finding_set' })).toBe(false);
        expect(isRemediationRow({ row_patch_applies: false })).toBe(true);
        expect(isRemediationRow({ scoring_mode: 'finding_set' })).toBe(false);
    });

    it('shows the rates, counts, reasons and judge score', () => {
        render(<RemediationMetricsPanel metrics={METRICS} provenance="Run #21 · gold_test · result 1001" />);
        expect(screen.getByTestId('rem-pass')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('rem-pass')).toHaveTextContent('4 of 8 rows');
        expect(screen.getByTestId('rem-applies')).toHaveTextContent('75.0%');
        expect(screen.getByTestId('rem-parses')).toHaveTextContent('80.0%');
        expect(screen.getByTestId('rem-parses')).toHaveTextContent('over 5 rows');
        expect(screen.getByTestId('rem-resolved')).toHaveTextContent('60.0%');
        expect(screen.getByTestId('rem-resolved')).toHaveTextContent('6 rows checked by the detector');
        expect(screen.getByTestId('rem-new-findings')).toHaveTextContent('10.0%');
        expect(screen.getByTestId('rem-citations')).toHaveTextContent('90.0%');
        expect(screen.getByTestId('rem-reason-context_mismatch')).toHaveTextContent('context does not match the code');
        expect(screen.getByTestId('rem-judge')).toHaveTextContent('0.75');
        expect(screen.getByTestId('rem-skipped')).toHaveTextContent('3 rows');
        expect(screen.getByText(/Run #21 · gold_test · result 1001/)).toBeInTheDocument();
    });

    it('says what was not evaluated', () => {
        render(
            <RemediationMetricsPanel
                metrics={{ scoring_mode: 'remediation', remediation_pass_rate: 0.25, total: 4, correct: 1, patch_applies_rate: 0.5 }}
                provenance="Run #1"
            />,
        );
        expect(screen.getByTestId('rem-parses')).toHaveTextContent('not evaluated');
        expect(screen.getByTestId('rem-resolved')).toHaveTextContent('not evaluated');
        expect(screen.getByTestId('rem-resolved')).toHaveTextContent('no detector configured');
        expect(screen.getByTestId('rem-citations')).toHaveTextContent('no passages on these rows');
        expect(screen.getByTestId('rem-judge')).toHaveTextContent('not run');
        expect(screen.queryByTestId('rem-new-findings')).not.toBeInTheDocument();
        expect(screen.queryByTestId('rem-reasons')).not.toBeInTheDocument();
    });

    it('surfaces a detector error', () => {
        render(<RemediationMetricsPanel metrics={{ ...METRICS, detector_error: 'model failed to load' }} provenance="Run #1" />);
        expect(screen.getByTestId('rem-detector-error')).toHaveTextContent('model failed to load');
    });

    it('labels the headline metric', () => {
        expect(metricDisplayName('remediation_pass_rate')).toBe('remediation pass rate');
    });
});

const PATCH = '--- a/src/db.py\n+++ b/src/db.py\n@@ -45,1 +45,1 @@\n-q = "select " + name\n+q = "select ?"\n context';

const ROW = {
    scoring_mode: 'remediation',
    path: 'src/db.py', language: 'python', start_line: 40,
    row_finding: { label: 'injection.query', start_line: 45, end_line: 45 },
    parsed_prediction: { patch: PATCH, rationale: 'Use a bound parameter.', citations: ['P1'] },
    row_passage_headers: [
        { id: 'P1', where: 'docs/db.md:3-9 (run_query)' },
        { id: 'P2', where: 'style guide' },
    ],
    row_patch_applies: true, row_code_checkable: true, row_code_parses: true,
    row_issue_resolved: true, row_citations_valid: true, row_check_messages: ['python_ast: ok'],
};

describe('RemediationRowView', () => {
    it('renders the finding, the diff with classed lines, chips and cited passages', () => {
        render(<RemediationRowView row={ROW} />);
        expect(screen.getByTestId('remediation-row-where')).toHaveTextContent('src/db.py · python · starts at line 40');
        expect(screen.getByTestId('remediation-row-finding')).toHaveTextContent('injection.query · line 45');
        const diff = screen.getByTestId('rem-diff');
        expect(diff.querySelectorAll('.is-added')).toHaveLength(1);
        expect(diff.querySelectorAll('.is-removed')).toHaveLength(1);
        expect(screen.getByTestId('rem-chip-applies')).toHaveTextContent('patch applies');
        expect(screen.getByTestId('rem-chip-parses')).toHaveTextContent('patched code parses');
        expect(screen.getByTestId('rem-chip-resolved')).toHaveTextContent('finding resolved');
        expect(screen.getByTestId('rem-chip-citations')).toHaveTextContent('citations valid');
        expect(screen.getByTestId('rem-messages')).toHaveTextContent('python_ast: ok');
        expect(screen.getByTestId('rem-passage-P1')).toHaveTextContent('[P1] docs/db.md:3-9 (run_query) (cited)');
        expect(screen.getByTestId('rem-passage-P2')).toHaveTextContent('[P2] style guide');
        expect(screen.getByTestId('rem-passage-P2')).not.toHaveTextContent('cited');
    });

    it('shows failure states, not-checkable and not-evaluated', () => {
        render(
            <RemediationRowView
                row={{
                    ...ROW,
                    row_patch_applies: false, row_patch_reason: 'context_mismatch',
                    row_code_checkable: false, row_code_parses: null,
                    row_issue_resolved: null, row_citations_valid: false,
                    row_new_findings: ['logging.sensitive'],
                }}
            />,
        );
        expect(screen.getByTestId('rem-chip-applies')).toHaveTextContent('patch does not apply: context does not match the code');
        expect(screen.getByTestId('rem-chip-parses')).toHaveTextContent('not checkable');
        expect(screen.getByTestId('rem-chip-resolved')).toHaveTextContent('resolution not evaluated');
        expect(screen.getByTestId('rem-chip-citations')).toHaveTextContent('citation not in the passages');
        expect(screen.getByTestId('rem-new-findings-row')).toHaveTextContent('logging.sensitive');
    });

    it('flags malformed output', () => {
        render(<RemediationRowView row={{ row_malformed: true, row_patch_applies: false, parsed_prediction: null }} />);
        expect(screen.getByTestId('remediation-row-malformed')).toBeInTheDocument();
        expect(screen.queryByTestId('rem-diff')).not.toBeInTheDocument();
    });

    it('shows no task->code block without the opt-in metrics', () => {
        render(<RemediationMetricsPanel metrics={METRICS} provenance="p" />);
        expect(screen.queryByTestId('ce-metrics')).not.toBeInTheDocument();
    });

    it('shows the task->code block beside the remediation numbers in a mixed split', () => {
        render(
            <RemediationMetricsPanel
                metrics={{
                    ...METRICS,
                    code_example_count: 4, code_example_decided_count: 3, code_example_correct: 2,
                    code_example_pass_rate: 0.6667, code_example_output_parses_rate: 1,
                    code_example_checkable_count: 3, code_example_not_checkable_count: 1,
                    code_example_code_checks_pass_rate: 1, code_example_citations_valid_rate: 0.75,
                    code_example_rows_with_passages: 4, code_example_judged_count: 3, code_example_judge_pass_rate: 0.6667,
                    code_example_judge_unscored_count: 1, code_example_judge_unscored_reasons: { judge_no_verdict: 1 },
                    code_example_judge_label: 'stub:judge',
                    code_example_by_language: { python: { scored: 4, decided: 3, passed: 2, not_checkable: 1, pass_rate: 0.6667 } },
                }}
                provenance="p"
            />,
        );
        expect(screen.getByTestId('rem-pass')).toHaveTextContent('50.0%');
        expect(screen.getByTestId('ce-pass')).toHaveTextContent('66.7%');
        expect(screen.getByTestId('ce-pass')).toHaveTextContent('2 of 3 scored rows (4 total)');
        expect(screen.getByTestId('ce-checks')).toHaveTextContent('3 checkable, 1 not checkable');
        expect(screen.getByTestId('ce-unscored-judge_no_verdict')).toHaveTextContent('1 unscored: judge gave no readable verdict');
        expect(screen.getByTestId('ce-lang-python')).toHaveTextContent('python');
        expect(screen.getByTestId('ce-metrics')).toHaveTextContent('stub:judge');
    });

    it('hides the remediation grid when only task->code rows were scored', () => {
        render(
            <RemediationMetricsPanel
                metrics={{ scoring_mode: 'remediation', total: 1, correct: 1, code_example_count: 1, code_example_decided_count: 1,
                    code_example_correct: 1, code_example_pass_rate: 1 }}
                provenance="p"
            />,
        );
        expect(screen.queryByTestId('rem-pass')).not.toBeInTheDocument();
        expect(screen.getByTestId('ce-pass')).toHaveTextContent('100.0%');
        expect(metricDisplayName('code_example_pass_rate')).toBe('code-example pass rate');
    });

    it('renders a scored task->code row with its own chips', () => {
        render(
            <RemediationRowView
                row={{
                    scoring_mode: 'remediation', row_kind: 'code_example', language: 'python',
                    parsed_prediction: { code: 'x = 1', rationale: 'set x', citations: ['P1'] },
                    row_code_example_output_parses: true, row_code_example_checkable: false, row_code_example_checks_pass: null,
                    row_code_example_citations_valid: true, row_code_example_judge_pass: null, row_code_example_judge_verdict: null,
                    row_code_example_judge_unscored: 'no_judge_available', row_code_example_pass: null, row_cited_ids: ['P1'],
                }}
            />,
        );
        expect(screen.getByTestId('code-example-row')).toBeInTheDocument();
        expect(screen.queryByTestId('remediation-row')).not.toBeInTheDocument();
        expect(screen.getByTestId('ce-chip-pass')).toHaveTextContent('unscored');
        expect(screen.getByTestId('ce-chip-checks')).toHaveTextContent('not checkable');
        expect(screen.getByTestId('ce-chip-judge')).toHaveTextContent('judge unscored: no judge model available');
        expect(screen.getByTestId('ce-code')).toHaveTextContent('x = 1');
    });

    it('builds passage headers', () => {
        expect(passageHeader({ id: 'P1', path: 'a.py', start_line: 2, end_line: 2 })).toBe('a.py:2');
        expect(passageHeader({ id: 'P1', source: 'notes' })).toBe('notes');
    });
});
