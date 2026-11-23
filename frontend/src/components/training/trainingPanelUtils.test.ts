import { describe, expect, it } from 'vitest';
import { finalizeRunnerLogs } from './trainingPanelUtils';

describe('finalizeRunnerLogs', () => {
    it('drops stale progress-bar lines, appends a completed line, is idempotent', () => {
        const logs = ['loading model', ' 11%|#1  | 3/28 [00:01<00:09]', 'step 20 loss 1.2'];
        const out = finalizeRunnerLogs(logs);
        expect(out).toEqual(['loading model', 'step 20 loss 1.2', 'Run completed.']);
        expect(finalizeRunnerLogs(out)).toBe(out);
    });
    it('leaves an empty console empty', () => {
        const empty: string[] = [];
        expect(finalizeRunnerLogs(empty)).toBe(empty);
    });
});
