import { useCallback, useEffect, useState } from 'react';

import {
    listProvenanceSources,
    updateProvenanceSources,
    USAGE_LABEL,
    type ProvenanceList,
    type ProvenanceSource,
    type ProvenanceUpdate,
    type ProvenanceUsage,
} from '../../api/provenance';
import { toast } from '../../stores/toastStore';
import { extractApiErrorMessage } from '../../utils/apiError';
import './ProvenancePanel.css';

interface ProvenancePanelProps {
    projectId: number;
}

const FLAG_LABEL: Record<string, string> = {
    unknown: 'Unknown licence',
    non_commercial: 'Non-commercial',
};

/** `locator @ ref[:12]` so the user can tell which revision a row came from. */
export function sourceLabel(s: ProvenanceSource): string {
    const ref = s.source_ref ? ` @ ${s.source_ref.slice(0, 12)}` : '';
    return `${s.locator}${ref}`;
}

/** Licence & provenance ledger: one row per data source, bulk-editable. */
export default function ProvenancePanel({ projectId }: ProvenancePanelProps) {
    const [data, setData] = useState<ProvenanceList | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [picked, setPicked] = useState<Set<number>>(new Set());
    const [licence, setLicence] = useState('');
    const [usage, setUsage] = useState<ProvenanceUsage | ''>('');
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            setData(await listProvenanceSources(projectId));
            setError(null);
        } catch (err) {
            setError(extractApiErrorMessage(err, 'Could not load the data sources.').message);
        }
    }, [projectId]);

    useEffect(() => { void load(); }, [load]);

    const sources = data?.sources ?? [];
    const allPicked = sources.length > 0 && sources.every((s) => picked.has(s.id));
    const pickedSources = sources.filter((s) => picked.has(s.id));

    const toggle = (id: number) => {
        const next = new Set(picked);
        if (next.has(id)) next.delete(id); else next.add(id);
        setPicked(next);
    };
    const toggleAll = () => setPicked(allPicked ? new Set() : new Set(sources.map((s) => s.id)));

    const apply = async (patch: Omit<ProvenanceUpdate, 'source_ids'>, what: string) => {
        if (picked.size === 0) return;
        setBusy(true);
        try {
            await updateProvenanceSources(projectId, { source_ids: Array.from(picked), ...patch });
            toast.info(`${what} for ${picked.size} source(s).`, 3000);
            await load();
        } catch (err) {
            toast.error(extractApiErrorMessage(err, `Could not update: ${what}.`).message, 5000);
        } finally {
            setBusy(false);
        }
    };

    // Usage is fixed for predefined-split sources, so a bulk usage change must not include them.
    const lockedPicked = pickedSources.some((s) => s.usage_locked);

    if (error && !data) {
        return <p className="form-hint-warning" role="alert" data-testid="prov-error">{error}</p>;
    }
    if (!data) return <p className="u-hint" data-testid="prov-loading">Loading data sources...</p>;

    return (
        <div className="prov" data-testid="provenance-panel">
            <p className="u-hint prov__help">
                Where each piece of data came from and what you may do with it. Eval-only sources are held out of
                training and go to the test split; retrieval-only documents are indexed for passages but never trained on.
                {' '}{data.totals.sources} source(s), {data.totals.unknown_licence} with an unknown licence,
                {' '}{data.totals.non_commercial} non-commercial.
            </p>
            {sources.length === 0 ? (
                <p className="u-hint" data-testid="prov-empty">No data sources yet. Upload or import documents first.</p>
            ) : (
                <>
                    <div className="prov__bar" data-testid="prov-bar">
                        <label>
                            Licence
                            <input type="text" value={licence} placeholder="e.g. MIT, CC-BY-4.0"
                                onChange={(e) => setLicence(e.target.value)} data-testid="prov-licence-input" />
                        </label>
                        <button type="button" className="btn btn-sm btn-secondary"
                            disabled={busy || picked.size === 0 || !licence.trim()}
                            onClick={() => void apply({ licence: licence.trim() }, 'Licence set')}
                            data-testid="prov-set-licence">Set licence</button>
                        <button type="button" className="btn btn-sm btn-ghost"
                            disabled={busy || picked.size === 0}
                            onClick={() => void apply({ licence: null }, 'Licence cleared to unknown')}
                            data-testid="prov-clear-licence">Mark unknown</button>
                        <button type="button" className="btn btn-sm btn-secondary"
                            disabled={busy || picked.size === 0}
                            onClick={() => void apply({ licence_verified: true }, 'Licence marked verified')}
                            data-testid="prov-verify">Mark licence verified</button>
                        <label>
                            Usage
                            <select value={usage} onChange={(e) => setUsage(e.target.value as ProvenanceUsage | '')}
                                disabled={lockedPicked}
                                title={lockedPicked ? 'A predefined split keeps its own role; deselect it to change usage.' : undefined}
                                data-testid="prov-usage-select">
                                <option value="">Choose...</option>
                                {(Object.keys(USAGE_LABEL) as ProvenanceUsage[]).map((u) => (
                                    <option key={u} value={u}>{USAGE_LABEL[u]}</option>
                                ))}
                            </select>
                        </label>
                        <button type="button" className="btn btn-sm btn-secondary"
                            disabled={busy || picked.size === 0 || !usage || lockedPicked}
                            onClick={() => usage && void apply({ usage }, 'Usage set')}
                            data-testid="prov-set-usage">Set usage</button>
                    </div>
                    <div className="prov__scroll">
                        <table className="prov__table">
                            <thead>
                                <tr>
                                    <th><input type="checkbox" checked={allPicked} onChange={toggleAll}
                                        aria-label="Select all sources" data-testid="prov-select-all" /></th>
                                    <th>Kind</th>
                                    <th>Source</th>
                                    <th>Licence</th>
                                    <th>Verified</th>
                                    <th>Usage</th>
                                    <th>Docs</th>
                                    <th>Rows</th>
                                    <th>In splits (train / val / test)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {sources.map((s) => (
                                    <tr key={s.id} data-testid={`prov-row-${s.id}`}>
                                        <td><input type="checkbox" checked={picked.has(s.id)} onChange={() => toggle(s.id)}
                                            aria-label={`Select ${s.title}`} data-testid={`prov-pick-${s.id}`} /></td>
                                        <td>{s.source_kind.replace(/_/g, ' ')}</td>
                                        <td>
                                            <strong>{s.title}</strong>
                                            <small className="prov__locator form-hint" data-testid={`prov-locator-${s.id}`}>{sourceLabel(s)}</small>
                                        </td>
                                        <td>
                                            <div className="prov__badge-row">
                                                <span data-testid={`prov-licence-${s.id}`}>{s.licence || 'unknown'}</span>
                                                {s.licence_flags.map((f) => (
                                                    <span key={f} className="badge badge-warning" data-testid={`prov-flag-${s.id}-${f}`}>
                                                        {FLAG_LABEL[f] ?? f}
                                                    </span>
                                                ))}
                                            </div>
                                        </td>
                                        <td data-testid={`prov-verified-${s.id}`}>
                                            {s.licence_verified
                                                ? `Verified${s.licence_verified_by ? ` by ${s.licence_verified_by}` : ''}${s.licence_verified_at ? ` on ${new Date(s.licence_verified_at).toLocaleDateString()}` : ''}`
                                                : 'Not verified'}
                                        </td>
                                        <td title={s.usage_locked ? 'Governed by the predefined split role; not editable.' : undefined}
                                            data-testid={`prov-usage-${s.id}`}>
                                            {USAGE_LABEL[s.usage] ?? s.usage}
                                        </td>
                                        <td>{s.document_count}</td>
                                        <td>{s.row_count}</td>
                                        <td data-testid={`prov-splits-${s.id}`}>
                                            {s.in_prepared
                                                ? `${s.in_prepared.train} / ${s.in_prepared.val} / ${s.in_prepared.test}${s.in_prepared.excluded ? ` (${s.in_prepared.excluded} excluded)` : ''}`
                                                : 'Not prepared'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
        </div>
    );
}
