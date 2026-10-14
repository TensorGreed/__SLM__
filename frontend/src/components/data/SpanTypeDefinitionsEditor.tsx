import { useEffect, useState } from 'react';

import api from '../../api/client';
import { toast } from '../../stores/toastStore';

interface TypeDefinitionsPayload {
    definitions: Record<string, string>;
    types: string[];
    suggestions: Record<string, string>;
    /** The project's texts come from scans (OCR). */
    scan_tolerant?: boolean;
}

/** Fired on window when the project's definitions change outside the editor. */
export const SPAN_DEFINITIONS_CHANGED_EVENT = 'brewslm:span-definitions-changed';

/** "type: definition" lines ↔ map. */
export function definitionsToText(defs: Record<string, string>): string {
    return Object.entries(defs).map(([t, d]) => `${t}: ${d}`).join('\n');
}

export function textToDefinitions(text: string): Record<string, string> {
    const out: Record<string, string> = {};
    for (const line of text.split('\n')) {
        const idx = line.indexOf(':');
        if (idx <= 0) continue;
        const type = line.slice(0, idx).trim();
        const def = line.slice(idx + 1).trim();
        if (type && def) out[type] = def;
    }
    return out;
}

/** The project's labelling conventions — one line per entity type, sent to
 *  every model that labels spans (the training-label generator and the
 *  answer-key relabeller), so they draw type boundaries the same way. */
export default function SpanTypeDefinitionsEditor({ projectId }: { projectId: number }) {
    const [open, setOpen] = useState(false);
    const [payload, setPayload] = useState<TypeDefinitionsPayload | null>(null);
    const [text, setText] = useState('');
    const [saving, setSaving] = useState(false);
    const [scanTolerant, setScanTolerant] = useState(false);

    // Saved immediately: both labellers then get the scan-tolerant clause,
    // and evaluation / redaction tidy OCR debris off predicted spans.
    const toggleScanTolerant = async (next: boolean) => {
        setScanTolerant(next);
        try {
            await api.put(`/projects/${projectId}/synthetic/type-definitions`, { scan_tolerant: next });
            toast.success(next ? 'Scanned-text mode on: labelling and evaluation now tolerate OCR debris.' : 'Scanned-text mode off.', 4000);
        } catch (err) {
            setScanTolerant(!next);
            toast.error(err instanceof Error ? err.message : 'Could not save the setting.', 5000);
        }
    };

    useEffect(() => {
        if (!open || payload) return;
        api.get<TypeDefinitionsPayload>(`/projects/${projectId}/synthetic/type-definitions`)
            .then((res) => {
                setPayload(res.data);
                setScanTolerant(Boolean(res.data.scan_tolerant));
                setText(definitionsToText(res.data.definitions));
            })
            .catch(() => undefined);
    }, [open, payload, projectId]);

    // A compliance bundle applied elsewhere adds definitions — reload.
    useEffect(() => {
        const reload = () => setPayload(null);
        window.addEventListener(SPAN_DEFINITIONS_CHANGED_EVENT, reload);
        return () => window.removeEventListener(SPAN_DEFINITIONS_CHANGED_EVENT, reload);
    }, []);

    const suggestionCount = payload ? Object.keys(payload.suggestions).length : 0;

    const addSuggestions = () => {
        if (!payload) return;
        const merged = { ...payload.suggestions, ...textToDefinitions(text) };
        setText(definitionsToText(merged));
    };

    const save = async () => {
        setSaving(true);
        try {
            const res = await api.put<{ definitions: Record<string, string> }>(`/projects/${projectId}/synthetic/type-definitions`, {
                definitions: textToDefinitions(text),
            });
            setPayload((prev) => (prev ? { ...prev, definitions: res.data.definitions } : prev));
            setText(definitionsToText(res.data.definitions));
            toast.success(`Saved ${Object.keys(res.data.definitions).length} labelling conventions — the next span rows and answer key use them.`, 4000);
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not save the conventions.', 5000);
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="span-type-definitions" data-testid="span-type-definitions">
            <button type="button" className="btn btn-secondary" onClick={() => setOpen((v) => !v)} data-testid="span-type-definitions-toggle">
                {open ? 'Hide labelling conventions' : 'Labelling conventions'}
            </button>
            {open && (
                <div className="form-group">
                    <div className="form-hint">
                        One line per type — <code>type: what counts and what doesn&apos;t</code>. Both the model that writes the
                        training labels and the model that writes the answer key get these, so a date, an account number or a
                        name is labelled the same way by both.
                    </div>
                    <label className="form-hint" data-testid="span-scan-tolerant-label">
                        <input
                            type="checkbox"
                            checked={scanTolerant}
                            onChange={(e) => void toggleScanTolerant(e.target.checked)}
                            data-testid="span-scan-tolerant"
                        />
                        {' '}Texts come from scanned documents (OCR) — a value broken by stray spaces or marks is one span,
                        labelled without the marks; predicted spans are tidied the same way.
                    </label>
                    <textarea
                        className="input"
                        rows={8}
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="date_of_birth: a person's birth date only; any other date is not a date_of_birth"
                        data-testid="span-type-definitions-text"
                    />
                    <div className="seed-documents-card__row">
                        {suggestionCount > 0 && (
                            <button type="button" className="btn btn-secondary" onClick={addSuggestions} data-testid="span-type-definitions-suggest">
                                Add suggested wording for {suggestionCount} type{suggestionCount === 1 ? '' : 's'}
                            </button>
                        )}
                        <button type="button" className="btn btn-primary" onClick={() => void save()} disabled={saving} data-testid="span-type-definitions-save">
                            {saving ? 'Saving…' : 'Save conventions'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
