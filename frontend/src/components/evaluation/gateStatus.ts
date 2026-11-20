/**
 * gateStatus — how the Verdict reads a pass/fail rule report: a FAIL only when some required rule failed on
 * a measured value; rules whose metric the run's results do not have are "not measured" (the run still does
 * not pass). Reads the backend's `status` / `measured_failed_gate_ids`, with a fallback for older reports.
 */

export interface GateStatusReport {
    passed: boolean;
    failed_gate_ids: string[];
    measured_failed_gate_ids?: string[];
    status?: 'pass' | 'fail' | 'not_measured';
    checks: Array<{ gate_id: string; reason?: string | null }>;
}

/** Required rules that failed on a measured value. */
export function measuredFailedGateIds(report: GateStatusReport): string[] {
    if (Array.isArray(report.measured_failed_gate_ids)) return report.measured_failed_gate_ids;
    return report.failed_gate_ids.filter((id) => {
        const check = report.checks.find((c) => c.gate_id === id);
        return !(check && typeof check.reason === 'string' && check.reason.startsWith('missing_metric_'));
    });
}

export function gateReportStatus(report: GateStatusReport): 'pass' | 'fail' | 'not_measured' {
    if (report.status) return report.status;
    if (report.passed) return 'pass';
    return measuredFailedGateIds(report).length > 0 ? 'fail' : 'not_measured';
}

export function notMeasuredText(n: number): string {
    return n === 1
        ? "This run's results have no value for this metric, so its rule is not checked: not a failure of the model, but the run does not pass until it is measured or the rule set changes."
        : "This run's results have no value for these metrics, so their rules are not checked: not a failure of the model, but the run does not pass until they are measured or the rule set changes.";
}
