/**
 * forecastActionRouter — pure (kind, params) → destination URL.
 *
 * The TrainabilityForecastPanel emits ``onActionClicked(kind, params)``
 * for every signal with a ``suggested_action``. T3 added seven new
 * signal kinds whose params (target_rows_per_class, fragment_groups,
 * diversify hints, invalid_row_ids) need to actually flow through to
 * the destination panel and prefill its form. This router defines the
 * URL contract every action lands on, so the destinations can decode
 * it deterministically.
 *
 * Query-param encoding (kept legible — no base64):
 *   /pipeline/synthetic?prefill_mode=<SynthMode>&prefill_count=<int>
 *   /pipeline/goldset?fix_rows=1,7,12
 *   /pipeline/goldset?fragment_groups=positive|Positive;billing|Billing
 *
 * The "diversify" hint on synth_augment maps to a SynthMode:
 *   - "entity_types"      → cluster_targeted (varies entity vocabulary)
 *   - "negative_examples" → hard_negatives  (creates absent-feature rows)
 *   - undefined           → positives_paraphrase (safe default)
 */

import type { SuggestedActionKind } from '../api/trainabilityForecast';
import type { SynthMode } from '../api/synthPlaybook';


export interface ForecastActionRoute {
    path: string;
    search: string;
}


/** Numeric coercion that tolerates the suggested_action params being
 *  typed as ``Record<string, unknown>`` (the backend hands them through
 *  unchecked). Returns ``fallback`` for missing / non-finite values. */
function coerceCount(raw: unknown, fallback: number): number {
    const n = typeof raw === 'number' ? raw : Number(raw);
    if (!Number.isFinite(n) || n <= 0) return fallback;
    return Math.max(1, Math.min(5000, Math.round(n)));
}


function coerceStringArray(raw: unknown): string[] {
    if (!Array.isArray(raw)) return [];
    return raw
        .filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
        .map((s) => s.trim());
}


function coerceNumberArray(raw: unknown): number[] {
    if (!Array.isArray(raw)) return [];
    const out: number[] = [];
    for (const v of raw) {
        const n = typeof v === 'number' ? v : Number(v);
        if (Number.isFinite(n) && n >= 0) {
            out.push(Math.round(n));
        }
    }
    return out;
}


function encodeFragmentGroups(groups: unknown): string {
    if (!Array.isArray(groups)) return '';
    const parts: string[] = [];
    for (const group of groups) {
        const variants = coerceStringArray(group);
        if (variants.length < 2) continue;
        parts.push(variants.map(encodeURIComponent).join('|'));
    }
    return parts.join(';');
}


export function decodeFragmentGroups(encoded: string | null | undefined): string[][] {
    if (!encoded) return [];
    return encoded
        .split(';')
        .map((group) =>
            group
                .split('|')
                .map((v) => {
                    try {
                        return decodeURIComponent(v);
                    } catch {
                        return v;
                    }
                })
                .map((v) => v.trim())
                .filter(Boolean),
        )
        .filter((group) => group.length >= 2);
}


export function decodeRowIds(encoded: string | null | undefined): number[] {
    if (!encoded) return [];
    return encoded
        .split(',')
        .map((token) => {
            const trimmed = token.trim();
            // Number('') and Number(' ') both coerce to 0, which would
            // false-positive empty tokens as row 0. Reject explicitly.
            if (!trimmed) return -1;
            const n = Number(trimmed);
            return Number.isFinite(n) && n >= 0 ? Math.round(n) : -1;
        })
        .filter((n) => n >= 0);
}


function diversifyToMode(diversify: unknown): SynthMode | null {
    if (typeof diversify !== 'string') return null;
    const token = diversify.trim().toLowerCase();
    if (token === 'entity_types') return 'cluster_targeted';
    if (token === 'negative_examples') return 'hard_negatives';
    return null;
}


/** Map a (kind, params) pair to the destination route + query string.
 *
 *  Returns ``null`` when ``kind`` is not a recognised
 *  ``SuggestedActionKind`` — callers should treat this as "don't
 *  navigate" (which is safer than landing on a wrong tab).
 *
 *  The router is intentionally permissive about ``params``: it accepts
 *  ``Record<string, unknown>`` because the backend hands the dict
 *  through unchecked, and individual signals carry different param
 *  shapes. Missing / malformed fields fall back to sensible defaults
 *  rather than throwing.
 */
export function routeForecastAction(
    projectId: number,
    kind: SuggestedActionKind,
    params: Record<string, unknown>,
): ForecastActionRoute | null {
    const search = new URLSearchParams();

    if (kind === 'fix_gold_rows') {
        const rowIds = coerceNumberArray(params.invalid_row_ids);
        if (rowIds.length > 0) {
            search.set('fix_rows', rowIds.join(','));
        }
        const fragments = encodeFragmentGroups(params.fragment_groups);
        if (fragments) {
            search.set('fragment_groups', fragments);
        }
        return {
            path: `/project/${projectId}/pipeline/goldset`,
            search: search.toString(),
        };
    }

    // All synth_* actions land on the synthetic tab. The destination
    // is the same; what differs is the SynthMode the picker preselects.
    let mode: SynthMode;
    if (kind === 'synth_balance') {
        mode = 'class_balance_fill';
        const perClass = coerceCount(params.target_rows_per_class, 0);
        const under = coerceStringArray(params.underrepresented_classes);
        // target_rows = max(per_class * len(under_classes), 30) when
        // per_class > 0; fall back to 30 (the picker's default-ish).
        const count = perClass > 0 && under.length > 0
            ? Math.max(30, perClass * under.length)
            : 30;
        search.set('prefill_mode', mode);
        search.set('prefill_count', String(count));
    } else if (kind === 'synth_diversify') {
        mode = 'positives_paraphrase';
        search.set('prefill_mode', mode);
        search.set('prefill_count', String(coerceCount(params.target_rows, 50)));
    } else if (kind === 'synth_augment') {
        // The diversify hint (T3 span-extraction signals) overrides the
        // safe paraphrase default when present. Falls back when the
        // hint is unrecognised so future hints don't break old clients.
        mode = diversifyToMode(params.diversify) ?? 'positives_paraphrase';
        search.set('prefill_mode', mode);
        search.set('prefill_count', String(coerceCount(params.target_rows, 50)));
    } else {
        return null;
    }

    return {
        path: `/project/${projectId}/pipeline/synthetic`,
        search: search.toString(),
    };
}
