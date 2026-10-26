/**
 * Paste-YAML editor for the project's own checks: validate, save, delete.
 * Errors are grouped by code.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    deleteUserProbePacks,
    fetchUserProbePacks,
    restoreUserProbePacks,
    saveUserProbePack,
    validateUserProbePack,
} from '../../api/probePack';
import type { UserPackSummary, UserProbeHistoryEntry, UserProbePackIssue } from '../../api/probePack';

interface Props {
    projectId: number;
    packs: UserPackSummary[];
    onChanged: () => void;
}

function groupByCode(issues: UserProbePackIssue[]): [string, UserProbePackIssue[]][] {
    const m = new Map<string, UserProbePackIssue[]>();
    for (const i of issues) m.set(i.code, [...(m.get(i.code) ?? []), i]);
    return [...m.entries()];
}

function errorsOf(err: unknown): { message: string; errors: UserProbePackIssue[] } {
    const data = (err as { response?: { data?: Record<string, unknown> } })?.response?.data;
    const meta = (data?.metadata ?? {}) as { errors?: UserProbePackIssue[] };
    return {
        message: typeof data?.detail === 'string' ? data.detail : 'Request failed',
        errors: Array.isArray(meta.errors) ? meta.errors : [],
    };
}

/** The pack id a pasted YAML/JSON document declares (top-level `id`), if any. */
export function declaredPackId(text: string): string | null {
    const m = text.match(/^\s*"?id"?\s*:\s*"?([A-Za-z0-9][A-Za-z0-9._-]*)"?\s*,?\s*$/m);
    return m ? m[1] : null;
}

export default function UserProbeEditor({ projectId, packs, onChanged }: Props) {
    const [history, setHistory] = useState<UserProbeHistoryEntry[]>([]);
    const [text, setText] = useState('');
    const [errors, setErrors] = useState<UserProbePackIssue[]>([]);
    const [message, setMessage] = useState<string | null>(null);
    const [okNote, setOkNote] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const loadHistory = useCallback(async () => {
        try {
            const res = await fetchUserProbePacks(projectId);
            setHistory(Array.isArray(res?.history) ? res.history : []);
        } catch {
            setHistory([]);
        }
    }, [projectId]);

    useEffect(() => {
        void loadHistory();
    }, [loadHistory, packs]);

    const run = async (fn: () => Promise<void>) => {
        setBusy(true);
        setMessage(null);
        setOkNote(null);
        try {
            await fn();
        } catch (err) {
            const { message: m, errors: e } = errorsOf(err);
            if (e.length > 0) setErrors(e);
            else setMessage(m);
        } finally {
            setBusy(false);
        }
    };

    const validate = () =>
        run(async () => {
            const res = await validateUserProbePack(projectId, text);
            const errs = res.errors ?? [];
            setErrors(errs);
            if (res.valid && errs.length === 0) setOkNote('Valid.');
        });

    const save = () =>
        run(async () => {
            await saveUserProbePack(projectId, text);
            setErrors([]);
            setOkNote('Saved.');
            onChanged();
        });

    const restore = (version: string) =>
        run(async () => {
            await restoreUserProbePacks(projectId, version);
            setErrors([]);
            setOkNote(`Restored version ${version.slice(0, 8)}.`);
            onChanged();
        });

    const remove = () =>
        run(async () => {
            await deleteUserProbePacks(projectId);
            setErrors([]);
            setOkNote('Removed your checks.');
            onChanged();
        });

    return (
        <div className="probe-pack__user-editor" data-testid="user-probe-editor">
            <span className="probe-pack__weights-editor-label">Your checks</span>
            {packs.length > 0 && (
                <ul className="probe-pack__user-packs" data-testid="user-probe-packs">
                    {packs.map((p) => (
                        <li key={p.id}>
                            {p.name || p.id}
                            {p.version ? ` v${p.version}` : ''}: {p.probe_count} checks
                            {p.critical_count > 0 ? `, ${p.critical_count} critical` : ''}
                            {p.source?.kind === 'domain_pack' && p.source.pack_id
                                ? ` (from pack ${p.source.pack_id}${p.source.version ? `@${p.source.version}` : ''})`
                                : ''}
                        </li>
                    ))}
                </ul>
            )}
            {history.some((h) => h.restorable && !h.current) && (
                <details className="probe-pack__user-history" data-testid="user-probe-history">
                    <summary>Earlier versions</summary>
                    <ul>
                        {history
                            .filter((h) => h.restorable && !h.current)
                            .map((h) => (
                                <li key={h.version}>
                                    {h.updated_at ? `${h.updated_at.slice(0, 16).replace('T', ' ')}: ` : ''}
                                    {h.pack_count} packs, {h.probe_count} checks{' '}
                                    <button type="button" className="btn btn-sm" disabled={busy}
                                        onClick={() => void restore(h.version)}
                                        data-testid={`user-probe-restore-${h.version}`}>Restore</button>
                                </li>
                            ))}
                    </ul>
                </details>
            )}
            <textarea
                className="probe-pack__user-text"
                rows={10}
                value={text}
                placeholder="Paste a check pack as YAML or JSON"
                aria-label="Your checks (YAML)"
                data-testid="user-probe-text"
                onChange={(e) => setText(e.target.value)}
            />
            {text.trim() && (
                <p className="probe-pack__score-note" data-testid="user-probe-replace-note">
                    {declaredPackId(text)
                        ? packs.some((p) => p.id === declaredPackId(text))
                            ? `Saving replaces your pack "${declaredPackId(text)}"; your other packs are kept.`
                            : `Saving adds the pack "${declaredPackId(text)}"; your other packs are kept.`
                        : 'Saving replaces the pack with the same id (if any); your other packs are kept.'}
                </p>
            )}
            {errors.length > 0 && (
                <div className="probe-pack__user-errors" data-testid="user-probe-errors">
                    {groupByCode(errors).map(([code, list]) => (
                        <div key={code} data-testid={`user-probe-errors-${code}`}>
                            <strong>{code}</strong> ({list.length})
                            <ul>
                                {list.map((e, n) => (
                                    <li key={n}>
                                        {e.path ? <code>{e.path}</code> : null} {e.message}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            )}
            {message && <p className="probe-pack__user-error" role="alert">{message}</p>}
            {okNote && <p className="probe-pack__score-note" data-testid="user-probe-ok">{okNote}</p>}
            <div className="probe-pack__user-actions">
                <button type="button" className="btn btn-sm" disabled={busy || !text.trim()}
                    onClick={() => void validate()} data-testid="user-probe-validate">Validate</button>
                <button type="button" className="btn btn-sm" disabled={busy || !text.trim()}
                    onClick={() => void save()} data-testid="user-probe-save">Save</button>
                <button type="button" className="btn btn-sm" disabled={busy || packs.length === 0}
                    onClick={() => void remove()} data-testid="user-probe-delete">Delete all</button>
            </div>
        </div>
    );
}
