/**
 * Span-tagger threshold tuning (validation split) and the project's
 * per-type recall floors.
 *   GET/PUT /api/projects/{id}/span-recall-floors
 *   POST    /api/projects/{id}/span-thresholds/tune
 */

import api from './client';

export interface RecallFloorSuggestion {
    floor: number;
    /** Compliance regimes the type is tagged with (e.g. pci_dss). */
    regimes: string[];
}

export interface RecallFloorsPayload {
    floors: Record<string, number>;
    default: number | null;
    /** Types a floor may name (data, definitions, compliance bundles). */
    types: string[];
    /** Offered by the compliance bundles — never applied by the server. */
    suggestions: Record<string, RecallFloorSuggestion>;
}

export type FloorStatus = 'met' | 'not_reachable' | 'no_support' | null;

export interface ThresholdTypeReport {
    threshold: number;
    f1_before: number;
    f1_after: number;
    precision_before: number | null;
    precision_after: number | null;
    recall_before: number | null;
    recall_after: number | null;
    support: number;
    predicted: number;
    floor: number | null;
    floor_status: FloorStatus;
}

export interface RecallBiasSuggestion {
    /** Smallest bias meeting every unreachable floor; null = none in the grid. */
    recall_bias: number | null;
    grid: number[];
    per_type: Record<string, {
        floor: number;
        bias: number | null;
        recall_before: number | null;
        precision_before: number | null;
        recall_at_bias: number | null;
        precision_at_bias: number | null;
        precision_cost: number | null;
    }>;
    micro_precision_cost?: number;
}

export interface TuneThresholdsResult {
    experiment_id: number;
    split: string;
    rows: number;
    micro_f1_before: number;
    micro_f1_after: number;
    raised: Record<string, number>;
    report: Record<string, ThresholdTypeReport>;
    floors?: Record<string, number>;
    floors_unknown?: string[];
    floors_met?: string[];
    floors_not_reachable?: string[];
    floors_no_support?: string[];
    recall_bias?: number;
    recall_bias_suggestion?: RecallBiasSuggestion | null;
}

export async function fetchRecallFloors(projectId: number): Promise<RecallFloorsPayload> {
    const res = await api.get<RecallFloorsPayload>(`/projects/${projectId}/span-recall-floors`);
    return res.data;
}

export async function saveRecallFloors(
    projectId: number,
    body: { floors: Record<string, number>; default: number | null },
): Promise<RecallFloorsPayload> {
    const res = await api.put<RecallFloorsPayload>(`/projects/${projectId}/span-recall-floors`, body);
    return res.data;
}

export async function tuneSpanThresholds(
    projectId: number,
    body: { experiment_id: number; floors?: Record<string, number>; apply_recall_bias?: boolean },
): Promise<TuneThresholdsResult> {
    const res = await api.post<TuneThresholdsResult>(`/projects/${projectId}/span-thresholds/tune`, body);
    return res.data;
}

const fmt = (v: number | null | undefined) => (v == null ? '–' : v.toFixed(2));

/** "0.50 → 1.00" (or "–" sides when a side has no spans). */
export function beforeAfter(before: number | null | undefined, after: number | null | undefined): string {
    return `${fmt(before)} → ${fmt(after)}`;
}

export function floorStatusText(status: FloorStatus): string {
    if (status === 'met') return 'met';
    if (status === 'not_reachable') return 'not reachable at threshold 0';
    if (status === 'no_support') return 'no reference spans';
    return '—';
}

/** One line per unreachable floor: the recall bias that would meet it and
 *  what it costs in precision. */
export function recallBiasLines(s: RecallBiasSuggestion | null | undefined): string[] {
    if (!s) return [];
    const lines = Object.entries(s.per_type).map(([type, t]) => (
        t.bias == null
            ? `${type}: floor ${fmt(t.floor)} not reachable even at recall bias ${s.grid[s.grid.length - 1] ?? '–'}.`
            : `${type}: recall bias ${t.bias} meets the ${fmt(t.floor)} floor (recall ${beforeAfter(t.recall_before, t.recall_at_bias)}, precision ${beforeAfter(t.precision_before, t.precision_at_bias)}).`
    ));
    if (s.recall_bias != null) {
        const c = s.micro_precision_cost;
        const cost = c == null ? '' : c > 0
            ? `; overall precision −${c.toFixed(2)} (every type gets more spans)`
            : '; no overall precision cost on validation';
        lines.push(`Suggested recall bias: ${s.recall_bias}${cost}.`);
    }
    return lines;
}
