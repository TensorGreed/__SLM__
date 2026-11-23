import { describe, expect, it } from 'vitest';
import { finalizeRunnerLogs, parseMetricFromLogLine, runTotalEpochs } from './trainingPanelUtils';

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

    it('rounds a fractional plan up to a whole epoch, like the backend (fix pass, C)', () => {
        expect(runTotalEpochs({ total_epochs: 2.5, config: { num_epochs: 3 } })).toBe(3);
        expect(runTotalEpochs({ total_epochs: null, config: { num_epochs: 1.5 } })).toBe(2);
    });
});

describe('parseMetricFromLogLine: lines that are not trainer steps (phase 7i)', () => {
    it("ignores train.py's final report envelope (its first checkpoint read as epoch 3, step 100)", () => {
        const line = JSON.stringify({ status: 'completed', report: {
            effective_config: { num_epochs: 3, save_steps: 100 },
            epochs: 8.0, total_steps: 216, final_train_loss: 1.5,
            checkpoints: [{ step: 100, epoch: 3, train_loss: 2.17, eval_loss: 2.83 }],
        } });
        expect(parseMetricFromLogLine(line, 1)).toBeNull();
    });

    it('ignores the end-of-run summary row (run-average train_loss is not a step)', () => {
        expect(parseMetricFromLogLine("{'train_runtime': '29.54', 'train_loss': '2.205', 'epoch': '8'}", 1)).toBeNull();
    });

    it('still reads the trainer log rows and SLM_METRIC lines', () => {
        expect(parseMetricFromLogLine("{'loss': '3.834', 'grad_norm': '1.072', 'epoch': '0.7407'}", 1))
            .toEqual({ experiment_id: 1, epoch: 0.7407, train_loss: 3.834 });
        expect(parseMetricFromLogLine('{"loss": 2.1, "epoch": 1.5, "step": 40}', 1))
            .toEqual({ experiment_id: 1, epoch: 1.5, step: 40, train_loss: 2.1 });
        expect(parseMetricFromLogLine('SLM_METRIC {"step": 216, "epoch": 8.0, "eval_loss": 2.8}', 1))
            .toEqual({ experiment_id: 1, step: 216, epoch: 8, eval_loss: 2.8 });
    });
});
