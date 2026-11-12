/**
 * The eval summary's "rows without a judge verdict" notice (judge resilience A1).
 *
 * A held-out eval whose judge was unreachable (or answered without a verdict)
 * leaves those rows unjudged: they are out of the judged score and the pass /
 * fail counts, with a reason per row. The backend snapshot says how many and
 * why (``unjudged_reasons``) and, when the breaker stopped the calls, where
 * (``breaker``). Pure, so it is unit-tested apart from the card.
 */

export interface UnjudgedJudgeBlock {
    judge?: string | null;
    judged?: number | null;
    unjudged?: number | null;
    status?: 'partial' | 'paused' | string | null;
    unjudged_reasons?: Record<string, number> | null;
    breaker?: {
        tripped?: boolean;
        tripped_at_row?: number | null;
        consecutive?: number | null;
        last_error_kind?: string | null;
        mode?: string | null;
        waited_seconds?: number | null;
    } | null;
}

const REASON_WORDS: Record<string, string> = {
    judge_unreachable: 'not sent (the judge was unreachable)',
    transient_error: 'connection or provider error',
    unparseable: 'no verdict in the reply',
    permanent_error: 'request refused',
    invalid_row: 'not a valid row',
};

/** Plain words for an unjudged-row reason. */
export function unjudgedReasonLabel(reason: string): string {
    return REASON_WORDS[reason] ?? reason.replace(/_/g, ' ');
}

/** "77 not sent (the judge was unreachable), 3 connection or provider error" — largest first. */
export function unjudgedReasonsText(reasons: Record<string, number> | null | undefined): string {
    return Object.entries(reasons || {})
        .filter(([, n]) => typeof n === 'number' && n > 0)
        .sort((a, b) => b[1] - a[1])
        .map(([reason, n]) => `${n} ${unjudgedReasonLabel(reason)}`)
        .join(', ');
}

/** The notice text, or null when every row the judge was asked about got a verdict. Results stored
 *  before unjudged rows carried a reason (no ``status``) get no notice: they still count those rows by
 *  exact match, so "left out of the pass / fail counts" would not be true for them. */
export function unjudgedNoticeText(judge: UnjudgedJudgeBlock | null | undefined, evalResultId: number | null | undefined): string | null {
    const unjudged = typeof judge?.unjudged === 'number' ? judge.unjudged : 0;
    if (!judge || unjudged <= 0 || !judge.status) return null;
    const judged = typeof judge.judged === 'number' ? judge.judged : 0;
    const total = judged + unjudged;
    const reasons = unjudgedReasonsText(judge.unjudged_reasons);
    const parts = [
        `${unjudged} of ${total} answer${total === 1 ? '' : 's'} ha${unjudged === 1 ? 's' : 've'} no judge verdict${reasons ? `: ${reasons}` : ''}.`,
        'They are left out of the judged score and the pass / fail counts.',
    ];
    const b = judge.breaker;
    if (judge.status === 'paused' && b) {
        const kind = b.last_error_kind ? ` (${b.last_error_kind})` : '';
        const where = typeof b.tripped_at_row === 'number' ? ` at row ${b.tripped_at_row + 1}` : '';
        const waited = typeof b.waited_seconds === 'number' && b.waited_seconds > 0
            ? ` after waiting ${Math.round(b.waited_seconds / 60)} min`
            : '';
        parts.push(`The judge stopped${where} after failed calls${kind}${waited}.`);
    }
    const provenance = [judge.judge ? `judge ${judge.judge}` : null, evalResultId != null ? `eval #${evalResultId}` : null]
        .filter(Boolean)
        .join(' · ');
    if (provenance) parts.push(`(${provenance})`);
    const outage = Object.keys(judge.unjudged_reasons || {}).some((r) => r === 'judge_unreachable' || r === 'transient_error');
    if (outage) parts.push('Re-run the eval when the judge is reachable again.');
    return parts.join(' ');
}
