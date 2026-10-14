import { useEffect, useState } from 'react';

import api from '../../api/client';
import { toast } from '../../stores/toastStore';

export interface TypePattern {
    type: string;
    pattern: string;
    description?: string;
    flags?: string;
}

interface TypePatternsPayload {
    patterns: TypePattern[];
    errors: { index: number; type: string; pattern: string; error: string }[];
    builtin_types: string[];
    limits: { max_patterns: number; max_pattern_chars: number; flags: string };
}

interface PatternMatch {
    text_index: number;
    source: string | null;
    start: number;
    end: number;
    match: string;
    before: string;
    after: string;
}

export interface PatternTestResult {
    valid: boolean;
    error: string | null;
    type: string;
    texts_scanned: number;
    texts_matched: number;
    count: number;
    matches: PatternMatch[];
}

/** Same normalisation the backend applies to type names. */
export function normaliseTypeName(raw: string): string {
    return raw.trim().toLowerCase().replace(/\s+/g, '_');
}

function errorText(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: unknown; message?: unknown } } })?.response?.data;
    if (detail && typeof detail.detail === 'string') return detail.detail;
    if (detail && typeof detail.message === 'string') return detail.message;
    return err instanceof Error ? err.message : fallback;
}

/** "Add a type": a custom infotype in one place — its name, a one-line
 *  labelling convention (saved with the type definitions), and optionally
 *  the regex its identifiers follow (saved as a project pattern, so every
 *  regex pass — generation, regex-augment, relabelling — catches it with
 *  exact offsets). "Test pattern" shows what it matches in the project's
 *  saved rows and passages before anything is saved. */
export default function SpanAddTypeForm({
    projectId,
    onTypeAdded,
}: {
    projectId: number;
    onTypeAdded?: (type: string) => void;
}) {
    const [open, setOpen] = useState(false);
    const [name, setName] = useState('');
    const [definition, setDefinition] = useState('');
    const [pattern, setPattern] = useState('');
    const [ignoreCase, setIgnoreCase] = useState(false);
    const [testResult, setTestResult] = useState<PatternTestResult | null>(null);
    const [testing, setTesting] = useState(false);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState<TypePatternsPayload | null>(null);

    useEffect(() => {
        if (!open || saved) return;
        api.get<TypePatternsPayload>(`/projects/${projectId}/synthetic/type-patterns`)
            .then((res) => setSaved(res.data))
            .catch(() => undefined);
    }, [open, saved, projectId]);

    const type = normaliseTypeName(name);
    const flags = ignoreCase ? 'i' : '';

    const testPattern = async () => {
        setTesting(true);
        try {
            const res = await api.post<PatternTestResult>(`/projects/${projectId}/synthetic/type-patterns/test`, {
                pattern, type, flags,
            });
            setTestResult(res.data);
        } catch (err) {
            toast.error(errorText(err, 'Could not test the pattern.'), 5000);
        } finally {
            setTesting(false);
        }
    };

    const putPatterns = async (patterns: TypePattern[]) => {
        const res = await api.put<{ patterns: TypePattern[] }>(`/projects/${projectId}/synthetic/type-patterns`, { patterns });
        setSaved((prev) => (prev ? { ...prev, patterns: res.data.patterns, errors: [] } : prev));
        return res.data.patterns;
    };

    const addType = async () => {
        if (!type) return;
        setSaving(true);
        try {
            if (pattern.trim()) {
                const current = saved?.patterns
                    ?? (await api.get<TypePatternsPayload>(`/projects/${projectId}/synthetic/type-patterns`)).data.patterns;
                await putPatterns([...current, { type, pattern: pattern.trim(), description: definition.trim(), flags }]);
            }
            if (definition.trim()) {
                const res = await api.get<{ definitions: Record<string, string> }>(`/projects/${projectId}/synthetic/type-definitions`);
                await api.put(`/projects/${projectId}/synthetic/type-definitions`, {
                    definitions: { ...res.data.definitions, [type]: definition.trim() },
                });
            }
            onTypeAdded?.(type);
            toast.success(
                `Added ${type}${pattern.trim() ? ' with its pattern' : ''} — it's in the entity-type list for the next span rows.`,
                4000,
            );
            setName('');
            setDefinition('');
            setPattern('');
            setIgnoreCase(false);
            setTestResult(null);
        } catch (err) {
            toast.error(errorText(err, 'Could not add the type.'), 6000);
        } finally {
            setSaving(false);
        }
    };

    const removePattern = async (index: number) => {
        if (!saved) return;
        try {
            await putPatterns(saved.patterns.filter((_, i) => i !== index));
        } catch (err) {
            toast.error(errorText(err, 'Could not remove the pattern.'), 5000);
        }
    };

    return (
        <div className="span-add-type" data-testid="span-add-type">
            <button type="button" className="btn btn-secondary" onClick={() => setOpen((v) => !v)} data-testid="span-add-type-toggle">
                {open ? 'Hide add a type' : 'Add a type'}
            </button>
            {open && (
                <div className="form-group">
                    <div className="form-hint">
                        Your own identifier types (an employee id, a ticket reference, an account format). The definition goes
                        to every labelling model; a pattern lets the regex pass catch the type with exact offsets, beside the
                        built-in ones ({(saved?.builtin_types ?? []).join(', ') || 'email, phone, …'}).
                    </div>
                    <input
                        className="input"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Type name, e.g. employee_id"
                        data-testid="span-add-type-name"
                    />
                    <input
                        className="input"
                        value={definition}
                        onChange={(e) => setDefinition(e.target.value)}
                        placeholder="One-line definition, e.g. a staff id like EMP-12345; not a customer number"
                        data-testid="span-add-type-definition"
                    />
                    <input
                        className="input"
                        value={pattern}
                        onChange={(e) => { setPattern(e.target.value); setTestResult(null); }}
                        placeholder="Pattern (optional regex), e.g. \bEMP-\d{5}\b"
                        maxLength={saved?.limits.max_pattern_chars ?? 300}
                        data-testid="span-add-type-pattern"
                    />
                    <div className="seed-documents-card__row">
                        <label className="form-hint">
                            <input type="checkbox" checked={ignoreCase} onChange={(e) => { setIgnoreCase(e.target.checked); setTestResult(null); }} data-testid="span-add-type-ignore-case" />
                            {' '}Ignore case
                        </label>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => void testPattern()}
                            disabled={!pattern.trim() || testing}
                            data-testid="span-add-type-test"
                        >
                            {testing ? 'Testing…' : 'Test pattern'}
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => void addType()}
                            disabled={!type || saving || (testResult !== null && !testResult.valid)}
                            data-testid="span-add-type-save"
                        >
                            {saving ? 'Adding…' : 'Add type'}
                        </button>
                    </div>
                    {testResult && (
                        <div className="form-hint" data-testid="span-add-type-test-result">
                            {testResult.valid ? (
                                <>
                                    {testResult.count} match{testResult.count === 1 ? '' : 'es'} in {testResult.texts_matched} of{' '}
                                    {testResult.texts_scanned} texts (saved rows + cleaned passages).
                                    {testResult.matches.length > 0 && (
                                        <ul data-testid="span-add-type-matches">
                                            {testResult.matches.map((m, i) => (
                                                <li key={`${m.text_index}-${m.start}-${i}`}>
                                                    <code>…{m.before}<mark>{m.match}</mark>{m.after}…</code>{' '}
                                                    ({m.source ?? 'text'} #{m.text_index}, chars {m.start}–{m.end})
                                                </li>
                                            ))}
                                        </ul>
                                    )}
                                </>
                            ) : (
                                <span data-testid="span-add-type-test-error">Pattern rejected: {testResult.error}</span>
                            )}
                        </div>
                    )}
                    {saved && saved.patterns.length > 0 && (
                        <div className="form-hint" data-testid="span-add-type-saved">
                            Saved patterns:
                            <ul>
                                {saved.patterns.map((p, i) => (
                                    <li key={`${p.type}-${p.pattern}-${i}`}>
                                        <strong>{p.type}</strong> <code>{p.pattern}</code>{p.flags ? ` (${p.flags})` : ''}{' '}
                                        <button type="button" className="btn btn-secondary" onClick={() => void removePattern(i)} data-testid={`span-add-type-remove-${i}`}>
                                            Remove
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
