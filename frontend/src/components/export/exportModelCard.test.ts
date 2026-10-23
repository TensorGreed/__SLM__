import { describe, expect, it } from 'vitest';
import { modelCardLine } from './exportModelCard';

describe('modelCardLine', () => {
    it('names the lift check a reader card quotes', () => {
        const line = modelCardLine({
            reader_package: { model_card: 'README.md', evaluated: true, split: 'test', judge: 'ollama:gemma4:12b', verdict: 'better' },
        });
        expect(line).toEqual({
            text: 'Model card: passage reader (test split · judged by ollama:gemma4:12b · better than base + passages).',
            warn: false,
        });
    });

    it('warns when a reader card has no lift check', () => {
        expect(modelCardLine({ reader_package: { evaluated: false } }).warn).toBe(true);
    });

    it('lists a span card evaluations', () => {
        expect(modelCardLine({ span_package: { evaluations: ['gold_test', 'test'] } }).text).toBe(
            'Model card: span tagger (gold_test, test).',
        );
    });

    it('warns that a card-less export publishes a blank Hub page', () => {
        for (const manifest of [{}, null, { span_package: null, reader_package: null }]) {
            const line = modelCardLine(manifest);
            expect(line.warn).toBe(true);
            expect(line.text).toContain('Hub page will be blank');
        }
    });
});
