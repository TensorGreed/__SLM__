import { describe, expect, it } from 'vitest';
import { gateReportStatus, measuredFailedGateIds } from './gateStatus';

const missing = (id: string, metric: string) => ({ gate_id: id, reason: `missing_metric_${metric}` });

describe('gateStatus', () => {
    it('only missing metrics read "not measured", never a fail', () => {
        const report = {
            passed: false,
            failed_gate_ids: ['min_f1', 'min_llm_judge_pass_rate'],
            checks: [missing('min_f1', 'f1'), missing('min_llm_judge_pass_rate', 'llm_judge_pass_rate')],
        };
        expect(measuredFailedGateIds(report)).toEqual([]);
        expect(gateReportStatus(report)).toBe('not_measured');
    });

    it('a measured failure is a fail even beside missing metrics', () => {
        const report = {
            passed: false,
            failed_gate_ids: ['min_f1', 'min_tool_call_pass_rate'],
            checks: [missing('min_f1', 'f1'), { gate_id: 'min_tool_call_pass_rate', reason: 'below_threshold' }],
        };
        expect(measuredFailedGateIds(report)).toEqual(['min_tool_call_pass_rate']);
        expect(gateReportStatus(report)).toBe('fail');
    });

    it('reads the backend status when present', () => {
        expect(gateReportStatus({ passed: false, failed_gate_ids: ['x'], checks: [], status: 'not_measured', measured_failed_gate_ids: [] })).toBe('not_measured');
        expect(gateReportStatus({ passed: true, failed_gate_ids: [], checks: [] })).toBe('pass');
    });
});
