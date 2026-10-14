import { useEffect, useState } from 'react';

import {
    applyComplianceBundles,
    fetchComplianceBundles,
    saveTypeRegimes,
    type ComplianceBundlesPayload,
} from '../../api/compliance';
import { toast } from '../../stores/toastStore';
import { SPAN_DEFINITIONS_CHANGED_EVENT } from './SpanTypeDefinitionsEditor';

/** The entity-type list after a bundle change: the user's list keeps its
 *  order, types a removed bundle took out of the map drop, new ones append. */
export function mergeEntityTypes(current: string[], before: string[], after: string[]): string[] {
    const removed = new Set(before.filter((t) => !after.includes(t)));
    const kept = current.filter((t) => !removed.has(t));
    return [...kept, ...after.filter((t) => !kept.includes(t))];
}

/** "type: tag, tag" lines ↔ map. */
export function typeRegimesToText(map: Record<string, string[]>): string {
    return Object.entries(map).map(([t, tags]) => `${t}: ${tags.join(', ')}`).join('\n');
}

export function textToTypeRegimes(text: string): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    for (const line of text.split('\n')) {
        const idx = line.indexOf(':');
        if (idx <= 0) continue;
        const type = line.slice(0, idx).trim();
        const tags = line.slice(idx + 1).split(',').map((t) => t.trim()).filter(Boolean);
        if (type && tags.length) out[type] = tags;
    }
    return out;
}

/** Compliance bundles: pick PCI-DSS / GDPR / HIPAA / CCPA (or tag types with
 *  your own regime) — fills the entity-type list, adds starting labelling
 *  conventions for types you haven't defined, and lets the eval report
 *  recall per regime. */
export default function ComplianceBundlePicker({
    projectId,
    entityTypes,
    onEntityTypesChange,
}: {
    projectId: number;
    entityTypes: string[];
    onEntityTypesChange: (types: string[]) => void;
}) {
    const [open, setOpen] = useState(false);
    const [payload, setPayload] = useState<ComplianceBundlesPayload | null>(null);
    const [selected, setSelected] = useState<string[]>([]);
    const [regimeText, setRegimeText] = useState('');
    const [busy, setBusy] = useState(false);

    useEffect(() => {
        if (!open || payload) return;
        fetchComplianceBundles(projectId)
            .then((data) => {
                setPayload(data);
                setSelected(data.bundles);
                setRegimeText(typeRegimesToText(data.type_regimes));
            })
            .catch(() => undefined);
    }, [open, payload, projectId]);

    const toggle = (id: string) =>
        setSelected((prev) => (prev.includes(id) ? prev.filter((b) => b !== id) : [...prev, id]));

    const applied = payload?.bundles ?? [];
    const toApply = selected.filter((id) => !applied.includes(id));
    const toRemove = applied.filter((id) => !selected.includes(id));

    const apply = async () => {
        if (!payload) return;
        setBusy(true);
        try {
            const res = await applyComplianceBundles(projectId, toApply, toRemove);
            onEntityTypesChange(mergeEntityTypes(entityTypes, payload.entity_types, res.entity_types));
            setPayload({ ...payload, ...res, builtin: payload.builtin.map((b) => ({ ...b, applied: res.bundles.includes(b.id) })) });
            setSelected(res.bundles);
            setRegimeText(typeRegimesToText(res.type_regimes));
            window.dispatchEvent(new Event(SPAN_DEFINITIONS_CHANGED_EVENT));
            const kept = res.definitions_kept.length;
            toast.success(
                `${res.entity_types.length} entity types under ${res.bundles.length} bundle${res.bundles.length === 1 ? '' : 's'}` +
                    (res.definitions_added.length ? ` · added ${res.definitions_added.length} labelling conventions` : '') +
                    (kept ? ` · kept your wording for ${kept}` : ''),
                4000,
            );
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not apply the bundles.', 5000);
        } finally {
            setBusy(false);
        }
    };

    const saveRegimes = async () => {
        if (!payload) return;
        setBusy(true);
        try {
            const res = await saveTypeRegimes(projectId, textToTypeRegimes(regimeText));
            onEntityTypesChange(mergeEntityTypes(entityTypes, payload.entity_types, res.entity_types));
            setPayload({ ...payload, ...res });
            setRegimeText(typeRegimesToText(res.type_regimes));
            toast.success(`Saved regime tags for ${res.entity_types.length} types — the span eval reports recall per regime.`, 4000);
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not save the regime tags.', 5000);
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="compliance-bundles" data-testid="compliance-bundles">
            <button type="button" className="btn btn-secondary" onClick={() => setOpen((v) => !v)} data-testid="compliance-bundles-toggle">
                {open ? 'Hide compliance bundles' : 'Compliance bundles'}
            </button>
            {open && payload && (
                <div className="form-group">
                    <div className="form-hint">
                        A bundle is a starting set of entity types for one regime. Applying it adds its types to the list
                        below and starting labelling conventions for the types you haven&apos;t defined (your own wording is
                        kept). The span evaluation then reports recall per regime.
                    </div>
                    {payload.builtin.map((bundle) => (
                        <label key={bundle.id} className="form-hint" data-testid={`compliance-bundle-${bundle.id}`}>
                            <input
                                type="checkbox"
                                checked={selected.includes(bundle.id)}
                                onChange={() => toggle(bundle.id)}
                                data-testid={`compliance-bundle-check-${bundle.id}`}
                            />{' '}
                            <strong>{bundle.name}</strong>
                            {bundle.applied ? ' (applied)' : ''} — {bundle.types.map((t) => t.type).join(', ')}
                        </label>
                    ))}
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void apply()}
                        disabled={busy || (toApply.length === 0 && toRemove.length === 0)}
                        data-testid="compliance-bundles-apply"
                    >
                        {busy ? 'Applying…' : 'Apply bundles'}
                    </button>
                    <div className="form-hint">
                        Regime tags per type — <code>type: tag, tag</code>. Add your own tag (e.g. <code>internal_tier_1</code>)
                        to group types your own way.
                    </div>
                    <textarea
                        className="input"
                        rows={6}
                        value={regimeText}
                        onChange={(e) => setRegimeText(e.target.value)}
                        placeholder="credit_card: pci_dss, ccpa"
                        data-testid="compliance-type-regimes-text"
                    />
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void saveRegimes()}
                        disabled={busy}
                        data-testid="compliance-type-regimes-save"
                    >
                        Save regime tags
                    </button>
                </div>
            )}
        </div>
    );
}
