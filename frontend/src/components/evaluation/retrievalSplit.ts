/** The retrieval judge's split of a with-passages comparison: did the
 *  retrieved passages hold the answer (found / missed), and of the answers
 *  the judge marked wrong, how many were retrieval's fault vs the model's
 *  (`answer_judge_service.judge_retrieval`). */
export interface RetrievalSplit {
    judged: number;
    unjudged?: number | null;
    found: number;
    missed: number;
    found_rate: number | null;
    wrong_missed: number;
    wrong_misread: number;
    correct_missed?: number | null;
    judge_calls?: number | null;
    judge_cached?: number | null;
}

export type RetrievalVerdict = 'found' | 'missed';

/** "Retrieval found the answer in 29 of 46 rows. Of the 17 wrong answers,
 *  retrieval missed it in 9 and found it in 8 — the model misread those." */
export function retrievalSplitText(split: RetrievalSplit): string {
    const wrong = split.wrong_missed + split.wrong_misread;
    let text = `Retrieval found the answer in ${split.found} of ${split.judged} rows`;
    if (wrong > 0) {
        text += `. Of the ${wrong} wrong answer${wrong === 1 ? '' : 's'}, retrieval missed it in ${split.wrong_missed} and found it in ${split.wrong_misread}`;
        text += split.wrong_misread > 0 ? ' — the model misread those.' : '.';
    } else {
        text += '.';
    }
    return text;
}

/** Per-row label: "retrieval found it (passage 2) — the model misread it"
 *  for a wrong answer, "retrieval found it (passage 2)" otherwise, or
 *  "retrieval missed it". */
export function retrievalRowText(
    verdict: RetrievalVerdict | null | undefined,
    passage: number | null | undefined,
    answerVerdict: string | null | undefined,
): string | null {
    if (!verdict) return null;
    if (verdict === 'missed') return 'retrieval missed it';
    const where = passage ? ` (passage ${passage})` : '';
    return answerVerdict === 'wrong' ? `retrieval found it${where} — the model misread it` : `retrieval found it${where}`;
}
