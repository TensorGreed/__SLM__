import { describe, expect, it } from 'vitest';

import { evidenceNote, heldBackByMcNemar, mcnemarText, preGuardNote, shortEvidenceText, vsCurrentText, type LiftEvidence } from './liftEvidence';

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

describe('vsCurrentText', () => {
    it('shows the McNemar p and why a one-way change is held back', () => {
        const text = vsCurrentText({ verdict: 'within_noise', n: 30, better: 5, worse: 0, same: 25, mean_diff: 0.167, ci_low: 0.02, ci_high: 0.31, mcnemar_p: 0.0625 });
        expect(text).toBe('within noise · 5 better, 0 worse (McNemar p = 0.063: too few changed rows to count)');
    });

    it('an older stored win without p is labelled as recorded before the check', () => {
        expect(vsCurrentText({ verdict: 'better', n: 46, better: 9, worse: 2, same: 35 }))
            .toBe('beats current · 9 better, 2 worse · recorded before the McNemar check');
        expect(vsCurrentText({ verdict: 'within_noise', n: 46, better: 6, worse: 4, same: 36 })).toBe('within noise · 6 better, 4 worse');
    });

    it('a gain shows its p without the held-back reason', () => {
        expect(vsCurrentText({ verdict: 'better', n: 46, better: 9, worse: 0, same: 37, ci_low: 0.1, ci_high: 0.3, mcnemar_p: 0.0039 }))
            .toBe('beats current · 9 better, 0 worse (McNemar p = 0.004)');
    });
});

describe('preGuardNote', () => {
    it('labels only stored better / worse verdicts that predate mcnemar_p', () => {
        expect(preGuardNote({ verdict: 'better' })).toBe(' · recorded before the McNemar check');
        expect(preGuardNote({ verdict: 'worse', mcnemar_p: null })).toBe('');
        expect(preGuardNote({ verdict: 'better', mcnemar_p: 0.01 })).toBe('');
        expect(preGuardNote({ verdict: 'within_noise' })).toBe('');
        expect(preGuardNote(null)).toBe('');
    });
});
