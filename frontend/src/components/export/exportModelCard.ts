/**
 * What model card a completed export carries, read from its manifest —
 * shown next to "Publish to the Hub" so a card-less publish (a blank Hub
 * page) is never a surprise, and a reader card names the lift check it
 * quotes (split · judge · verdict).
 */

export interface ModelCardLine {
    text: string;
    warn: boolean;
}

const VERDICT_TEXT: Record<string, string> = {
    better: 'better than base + passages',
    worse: 'worse than base + passages',
    within_noise: 'within noise of base + passages',
    too_few_rows: 'too few rows to decide',
};

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

export function modelCardLine(manifest: unknown): ModelCardLine {
    const m = asRecord(manifest) || {};
    const span = asRecord(m.span_package);
    if (span) {
        const evals = Array.isArray(span.evaluations) ? span.evaluations.map(String) : [];
        return {
            text: `Model card: span tagger (${evals.length ? evals.join(', ') : 'no evaluation recorded'}).`,
            warn: evals.length === 0,
        };
    }
    const reader = asRecord(m.reader_package);
    if (reader) {
        if (!reader.evaluated) {
            return { text: 'Model card: passage reader, with no lift check recorded — the card makes no score claim.', warn: true };
        }
        const verdict = VERDICT_TEXT[String(reader.verdict)] || String(reader.verdict || 'no verdict');
        const parts = [`${reader.split || 'test'} split`, reader.judge ? `judged by ${reader.judge}` : null, verdict];
        return { text: `Model card: passage reader (${parts.filter(Boolean).join(' · ')}).`, warn: false };
    }
    return { text: 'No model card in this export: the Hub page will be blank unless you add a README.md.', warn: true };
}
