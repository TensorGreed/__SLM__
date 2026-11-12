import { describe, expect, it } from 'vitest';

import { unjudgedNoticeText, unjudgedReasonLabel, unjudgedReasonsText } from './unjudgedNotice';

describe('unjudgedNotice', () => {
    it('names counts by reason, where the judge stopped, the judge and the eval', () => {
        const text = unjudgedNoticeText({
            judge: 'command',
            judged: 0,
            unjudged: 80,
            status: 'paused',
            unjudged_reasons: { transient_error: 3, judge_unreachable: 77 },
            breaker: { tripped: true, tripped_at_row: 2, last_error_kind: 'connection error' },
        }, 1391);
        expect(text).toBe(
            '80 of 80 answers have no judge verdict: 77 not sent (the judge was unreachable), 3 connection or provider error. '
            + 'They are left out of the judged score and the pass / fail counts. '
            + 'The judge stopped at row 3 after failed calls (connection error). '
            + '(judge command · eval #1391) Re-run the eval when the judge is reachable again.',
        );
    });

    it('a partial run without an outage does not ask for a re-run', () => {
        const text = unjudgedNoticeText({ judge: 'j', judged: 9, unjudged: 1, status: 'partial', unjudged_reasons: { unparseable: 1 } }, 4);
        expect(text).toContain('1 of 10 answers has no judge verdict: 1 no verdict in the reply.');
        expect(text).not.toContain('Re-run');
        expect(text).not.toContain('stopped');
    });

    it('is silent for complete runs and for results stored before reasons existed', () => {
        expect(unjudgedNoticeText({ judge: 'j', judged: 10, unjudged: 0 }, 1)).toBeNull();
        expect(unjudgedNoticeText({ judge: 'j', judged: 9, unjudged: 1 }, 1)).toBeNull();
        expect(unjudgedNoticeText(null, 1)).toBeNull();
    });

    it('labels reasons in plain words, largest first', () => {
        expect(unjudgedReasonLabel('permanent_error')).toBe('request refused');
        expect(unjudgedReasonLabel('something_new')).toBe('something new');
        expect(unjudgedReasonsText({ unparseable: 1, judge_unreachable: 5, transient_error: 0 })).toBe(
            '5 not sent (the judge was unreachable), 1 no verdict in the reply',
        );
    });
});
