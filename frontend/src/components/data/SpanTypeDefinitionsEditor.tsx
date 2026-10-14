import { useEffect, useState } from 'react';

import api from '../../api/client';
import { toast } from '../../stores/toastStore';

interface TypeDefinitionsPayload {
    definitions: Record<string, string>;
    types: string[];
    suggestions: Record<string, string>;
}

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

    useEffect(() => {
        if (!open || payload) return;
        api.get<TypeDefinitionsPayload>(`/projects/${projectId}/synthetic/type-definitions`)
            .then((res) => {
                setPayload(res.data);
                setText(definitionsToText(res.data.definitions));
            })
            .catch(() => undefined);
    }, [open, payload, projectId]);

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
