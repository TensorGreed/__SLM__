/** Text helpers for the project Overview (pure, no React). */

import type { EvalSummary } from '../api/evalSummary';

const VERDICT_TEXT: Record<string, string> = {
    better: 'beats its base model',
    worse: 'loses to its base model',
    same: 'matches its base model',
    evaluated: 'evaluated',
    not_evaluated: 'not evaluated yet',
    no_baseline: 'evaluated, no base-model comparison',
    no_comparison: 'evaluated, no comparison',
    no_trained_run: 'nothing trained yet',
};

function fmt(value: number | null | undefined): string {
    return typeof value === 'number' ? value.toFixed(3) : '—';
}

/** One line for the latest run: "Run #143 · span F1 0.778 on the answer key". */
export function latestRunText(summary: EvalSummary | null): { title: string; detail: string } | null {
    if (!summary) return null;
    if (summary.verdict === 'no_trained_run' && !summary.experiment_id) return null;
    const runId = summary.trained?.experiment_id ?? summary.experiment_id;
    const title = runId ? `Run #${runId}` : 'Latest run';
    if (summary.kind === 'span_eval') {
        const block = summary.answer_key ?? summary.test;
        const where = summary.answer_key ? 'the answer key' : 'the test examples';
        return { title, detail: block ? `span F1 ${fmt(block.f1)} on ${where}` : 'not evaluated yet' };
    }
    if (summary.kind === 'head_eval') {
        return { title, detail: `accuracy ${fmt(summary.head?.accuracy)} on the test examples` };
    }
    const head = summary.headline;
    const verdict = VERDICT_TEXT[summary.verdict] ?? summary.verdict;
    if (head) {
        return { title, detail: `${verdict} · ${head.metric_id} ${fmt(head.baseline_value)} → ${fmt(head.trained_value)}` };
    }
    return { title, detail: verdict };
}
