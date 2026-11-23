import { describe, expect, it } from 'vitest';
import { finalizeRunnerLogs, runTotalEpochs } from './trainingPanelUtils';

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

describe('runTotalEpochs (phase 7i)', () => {
    it("uses the trainer's own plan over the form's num_epochs (auto epochs ran 8, the form said 3)", () => {
        expect(runTotalEpochs({ total_epochs: 8, config: { num_epochs: 3 } })).toBe(8);
    });

    it('falls back to the form value before the trainer reports, then to 3', () => {
        expect(runTotalEpochs({ total_epochs: null, config: { num_epochs: 5 } })).toBe(5);
        expect(runTotalEpochs({ total_epochs: 0, config: {} })).toBe(3);
        expect(runTotalEpochs({ config: null })).toBe(3);
    });
});
