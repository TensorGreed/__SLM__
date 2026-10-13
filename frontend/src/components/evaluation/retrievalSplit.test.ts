import { describe, expect, it } from 'vitest';

import { retrievalRowText, retrievalSplitText } from './retrievalSplit';

describe('retrievalSplit', () => {
    it('states found-of-judged and splits the wrong answers into missed vs misread', () => {
        expect(retrievalSplitText({ judged: 46, found: 29, missed: 17, found_rate: 0.63, wrong_missed: 9, wrong_misread: 8 }))
            .toBe('Retrieval found the answer in 29 of 46 rows. Of the 17 wrong answers, retrieval missed it in 9 and found it in 8 — the model misread those.');
        expect(retrievalSplitText({ judged: 10, found: 8, missed: 2, found_rate: 0.8, wrong_missed: 1, wrong_misread: 0 }))
            .toBe('Retrieval found the answer in 8 of 10 rows. Of the 1 wrong answer, retrieval missed it in 1 and found it in 0.');
        expect(retrievalSplitText({ judged: 10, found: 10, missed: 0, found_rate: 1, wrong_missed: 0, wrong_misread: 0 }))
            .toBe('Retrieval found the answer in 10 of 10 rows.');
    });

    it('labels a row by what retrieval did and whether the model then misread it', () => {
        expect(retrievalRowText('found', 2, 'wrong')).toBe('retrieval found it (passage 2) — the model misread it');
        expect(retrievalRowText('found', 2, 'correct')).toBe('retrieval found it (passage 2)');
        expect(retrievalRowText('found', null, 'partial')).toBe('retrieval found it');
        expect(retrievalRowText('missed', null, 'wrong')).toBe('retrieval missed it');
        expect(retrievalRowText(null, null, 'wrong')).toBeNull();
    });
});
