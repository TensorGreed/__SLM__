import { describe, expect, it } from 'vitest';

import { evidenceNote, heldBackByMcNemar, mcnemarText, shortEvidenceText, type LiftEvidence } from './liftEvidence';

// 4 better / 0 worse on 50 correct / incorrect rows: the interval alone excludes 0, exact McNemar p = 0.125.
const fourOh: LiftEvidence = {
    n: 50, better: 4, worse: 0, same: 46, mean_diff: 0.08, ci_low: 0.002, ci_high: 0.158,
    mcnemar_p: 0.125, verdict: 'within_noise',
};

describe('McNemar guard wording', () => {
    it('formats the p-value', () => {
        expect(mcnemarText(0.125)).toBe('p = 0.125');
        expect(mcnemarText(0.0001)).toBe('p < 0.001');
        expect(mcnemarText(null)).toBeNull();
        expect(mcnemarText(undefined)).toBeNull();
    });

    it('explains a lift the interval would call but McNemar holds back', () => {
        expect(heldBackByMcNemar(fourOh)).toBe(true);
        const note = evidenceNote(fourOh, { trained: true, unit: 'exact match' });
        expect(note.label).toBe('Within noise');
        expect(note.text).toContain('only 4 rows changed (4 better, 0 worse)');
        expect(note.text).toContain('exact McNemar p = 0.125');
        expect(note.text).toContain('at least 6 changed rows');
    });

    it('keeps the plain within-noise text when the interval includes zero', () => {
        const ev = { ...fourOh, ci_low: -0.01, mcnemar_p: 0.5 };
        expect(heldBackByMcNemar(ev)).toBe(false);
        expect(evidenceNote(ev, { trained: false, unit: 'exact match' }).text).toContain('it includes zero');
    });

    it('shows p next to a gain', () => {
        const ev: LiftEvidence = { ...fourOh, better: 6, same: 44, mcnemar_p: 0.03125, verdict: 'better' };
        const note = evidenceNote(ev, { trained: false, unit: 'exact match' });
        expect(note.label).toBe('Gain on these rows');
        expect(note.text).toContain('(exact McNemar p = 0.031)');
        expect(shortEvidenceText(ev)).toBe('beyond row noise · 6 rows better, 0 worse (McNemar p = 0.031)');
    });

    it('continuous scores (no p) read as before', () => {
        const ev: LiftEvidence = { ...fourOh, mcnemar_p: null, verdict: 'better' };
        expect(evidenceNote(ev, { trained: false, unit: 'F1' }).text).not.toContain('McNemar');
        expect(shortEvidenceText({ verdict: 'within_noise', better: 3, worse: 1 })).toBe('within noise · 3 rows better, 1 worse');
    });
});
