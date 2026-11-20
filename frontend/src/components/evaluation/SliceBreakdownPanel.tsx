/**
 * SliceBreakdownPanel — per-slice pass counts of one eval result (`metrics.slice_breakdown`), with
 * Wilson 95% intervals. Slices come from the evaluated rows' own tags (`slice` / `slices` / `tags`,
 * top level or under `metadata`) and from the project's slice definitions; each row is counted on the
 * result's final verdict, and the opt-in strict pass is shown when the rows carry it. Collapsed by
 * default: it adds one closed line to the result.
 */

import './SliceBreakdownPanel.css';

export interface SliceRate {
    scored?: number;
    pass?: number;
    rate?: number | null;
    ci?: [number, number] | null;
}

export interface SliceEntry {
    id: string;
    source?: string;
    field?: string | null;
    label?: string;
    rows?: number;
    scored?: number;
    pass?: number;
    pass_rate?: number | null;
    pass_ci?: [number, number] | null;
    strict?: SliceRate | null;
}

export interface SliceBreakdown {
    version?: number;
    interval?: string;
    overall?: Omit<SliceEntry, 'id'> | null;
    slices?: SliceEntry[];
    tag_slices_omitted?: number;
}

export function sliceBreakdownOf(metrics: Record<string, unknown> | null | undefined): SliceBreakdown | null {
    const block = metrics?.slice_breakdown as SliceBreakdown | undefined;
    return block && Array.isArray(block.slices) && block.slices.length > 0 ? block : null;
}

function pct(v: number): string {
    return `${(v * 100).toFixed(1)}%`;
}

/** "174 / 200 · 87.0% (95% CI 81.6%–91.0%)", or "no scored rows". */
export function rateText(pass: number | undefined, scored: number | undefined, ci: [number, number] | null | undefined): string {
    if (!scored) return 'no scored rows';
    const p = pass ?? 0;
    const interval = ci ? ` (95% CI ${pct(ci[0])}–${pct(ci[1])})` : '';
    return `${p} / ${scored} · ${pct(p / scored)}${interval}`;
}

function sourceText(s: SliceEntry): string {
    return s.source === 'slice_definition' ? 'project slice' : `row ${s.field ?? 'tag'}`;
}

interface Props {
    breakdown: SliceBreakdown;
    /** Where the numbers come from, e.g. "Run #21 · eval set fresh v1 · result 1001". */
    provenance: string;
}

export default function SliceBreakdownPanel({ breakdown, provenance }: Props) {
    const slices = breakdown.slices ?? [];
    const overall = breakdown.overall ?? null;
    const hasStrict = slices.some((s) => s.strict && typeof s.strict.scored === 'number');
    const strictCell = (r: SliceRate | null | undefined) => (r ? rateText(r.pass, r.scored, r.ci) : 'not computed');
    return (
        <details className="card slice-breakdown" data-testid="slice-breakdown">
            <summary className="slice-breakdown__summary">By slice ({slices.length})</summary>
            <p className="u-hint" data-testid="slice-breakdown-provenance">
                {provenance}. Each row counts on the result's own pass / fail verdict; intervals are Wilson 95%. Slices
                come from the rows' own slice / tags fields and the project's slice definitions.
            </p>
            <table className="slice-breakdown__table">
                <thead>
                    <tr>
                        <th>Slice</th>
                        <th>Rows</th>
                        <th>Pass</th>
                        {hasStrict && <th>Strict pass</th>}
                    </tr>
                </thead>
                <tbody>
                    {slices.map((s) => (
                        <tr key={s.id} data-testid={`slice-row-${s.id}`}>
                            <td>
                                {s.label ?? s.id}
                                <span className="slice-breakdown__sub">{sourceText(s)}</span>
                            </td>
                            <td className="is-num">{s.rows ?? 0}</td>
                            <td className="is-num" data-testid={`slice-pass-${s.id}`}>{rateText(s.pass, s.scored, s.pass_ci)}</td>
                            {hasStrict && <td className="is-num" data-testid={`slice-strict-${s.id}`}>{strictCell(s.strict)}</td>}
                        </tr>
                    ))}
                    {overall && (
                        <tr data-testid="slice-row-overall">
                            <td>All rows</td>
                            <td className="is-num">{overall.rows ?? 0}</td>
                            <td className="is-num">{rateText(overall.pass, overall.scored, overall.pass_ci)}</td>
                            {hasStrict && <td className="is-num">{strictCell(overall.strict)}</td>}
                        </tr>
                    )}
                </tbody>
            </table>
            {breakdown.tag_slices_omitted ? (
                <p className="u-hint" data-testid="slice-breakdown-omitted">
                    {breakdown.tag_slices_omitted} smaller tag value{breakdown.tag_slices_omitted === 1 ? '' : 's'} not shown (the 50 largest are).
                </p>
            ) : null}
        </details>
    );
}
