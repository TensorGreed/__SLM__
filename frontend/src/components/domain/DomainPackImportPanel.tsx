/**
 * Import a domain pack zip: preview (validation issues grouped by code, content
 * summary, diff against the installed version) and commit.
 */

import { useRef, useState } from 'react';

import {
    extractPackErrors,
    groupIssuesByCode,
    importPack,
    previewPackImport,
} from '../../api/domainPacks';
import type { IdDiff, ImportPreview, PackDiff, PackIssue, PackSummary } from '../../api/domainPacks';
import './DomainPackImportPanel.css';

interface Props {
    /** Called after a successful (changed) import so lists can refresh. */
    onImported?: () => void;
}

const SUMMARY_LABEL: [keyof PackSummary, string][] = [
    ['taxonomy_entries', 'labels'],
    ['probe_packs', 'check packs'],
    ['probes', 'checks'],
    ['critical_probes', 'critical checks'],
    ['rubrics', 'rubrics'],
    ['schemas', 'schemas'],
    ['documents', 'documents'],
    ['gold_dev_rows', 'dev rows'],
    ['gold_test_rows', 'test rows'],
];

const ID_SECTIONS: [keyof PackDiff['sections'], string][] = [
    ['taxonomy', 'Labels'],
    ['probes', 'Checks'],
    ['rubrics', 'Rubrics'],
    ['schemas', 'Schemas'],
    ['documents', 'Documents'],
];

function IssueGroups({ title, issues, testId }: { title: string; issues: PackIssue[]; testId: string }) {
    if (issues.length === 0) return null;
    return (
        <div className="dp-import__issues" data-testid={testId}>
            <h4 className="dp-import__issues-title">
                {title} ({issues.length})
            </h4>
            {groupIssuesByCode(issues).map(([code, list]) => (
                <div key={code} className="dp-import__issue-group" data-testid={`${testId}-${code}`}>
                    <span className="dp-import__code">{code}</span>
                    <span className="dp-import__count">{list.length}</span>
                    <ul className="dp-import__issue-list">
                        {list.map((i, n) => (
                            <li key={`${i.path ?? ''}-${n}`}>
                                {i.path ? <code>{i.path}</code> : null} {i.message}
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}

function isEmptyIdDiff(d: IdDiff | undefined): boolean {
    return !d || (d.added.length + d.removed.length + d.changed.length === 0);
}

function DiffView({ diff }: { diff: PackDiff }) {
    const s = diff.sections;
    const rows: { key: string; label: string; text: string }[] = [];
    for (const [key, label] of ID_SECTIONS) {
        const d = s[key] as IdDiff | undefined;
        if (!isEmptyIdDiff(d)) {
            rows.push({
                key,
                label,
                text: `${d!.added.length} added, ${d!.removed.length} removed, ${d!.changed.length} changed`,
            });
        }
    }
    if (s.gold) {
        for (const split of ['dev', 'test'] as const) {
            const g = s.gold[split];
            if (g && (g.changed || g.from !== g.to)) {
                rows.push({
                    key: `gold-${split}`,
                    label: `Gold ${split}`,
                    text: g.from !== g.to ? `${g.from} to ${g.to} rows` : `${g.to} rows, content changed`,
                });
            }
        }
    }
    if (s.project_template && s.project_template.changed_fields.length > 0) {
        rows.push({
            key: 'project_template',
            label: 'Project template',
            text: `changed: ${s.project_template.changed_fields.join(', ')}`,
        });
    }
    if (s.overlay?.changed) rows.push({ key: 'overlay', label: 'Overlay', text: 'changed' });
    if (s.tool_catalog?.changed) rows.push({ key: 'tool_catalog', label: 'Tool catalog', text: 'changed' });
    return (
        <div className="dp-import__diff" data-testid="dp-import-diff">
            <h4 className="dp-import__issues-title">
                Changes: {diff.version.from ?? 'none'} to {diff.version.to ?? 'none'}
            </h4>
            {rows.length === 0 ? (
                <p className="dp-import__muted">No content sections changed.</p>
            ) : (
                <ul className="dp-import__diff-list">
                    {rows.map((r) => (
                        <li key={r.key} data-testid={`dp-diff-${r.key}`}>
                            <strong>{r.label}</strong>: {r.text}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

export default function DomainPackImportPanel({ onImported }: Props) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<ImportPreview | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [done, setDone] = useState<string | null>(null);
    const [doneWarnings, setDoneWarnings] = useState<PackIssue[]>([]);

    const reset = () => {
        setPreview(null);
        setError(null);
        setDone(null);
        setDoneWarnings([]);
    };

    const pick = async (f: File | null) => {
        setFile(f);
        reset();
        if (!f) return;
        setBusy(true);
        try {
            setPreview(await previewPackImport(f));
        } catch (err) {
            const { message, errors } = extractPackErrors(err);
            if (errors.length > 0) {
                setPreview({
                    valid: false, errors, warnings: [], summary: null, manifest: null,
                    content_hash: null, existing: null, proposed_version: null,
                    unchanged: false, diff: null,
                });
            } else {
                setError(`Could not preview the pack: ${message}`);
            }
        } finally {
            setBusy(false);
        }
    };

    const commit = async () => {
        if (!file) return;
        setBusy(true);
        setError(null);
        try {
            const res = await importPack(file);
            setDone(
                res.unchanged
                    ? `No changes: ${res.pack.pack_id}@${res.version} is already installed.`
                    : `Imported ${res.pack.pack_id}@${res.version}.`,
            );
            setDoneWarnings(res.warnings ?? []);
            if (!res.unchanged) onImported?.();
        } catch (err) {
            const { message, errors } = extractPackErrors(err);
            if (errors.length > 0) {
                setPreview((p) => (p ? { ...p, valid: false, errors } : p));
            } else {
                setError(`Import failed: ${message}`);
            }
        } finally {
            setBusy(false);
        }
    };

    const manifestId = preview?.manifest ? String(preview.manifest.pack_id ?? '') : '';
    const version = preview?.proposed_version ?? (preview?.manifest ? String(preview.manifest.version ?? '') : '');
    const canImport = !!preview && preview.valid && !busy && !done;

    return (
        <section className="dp-import card" data-testid="dp-import-panel">
            <header>
                <h3 className="dp-import__title">Import a domain pack</h3>
                <p className="dp-import__muted">
                    Upload a pack zip. You see what it contains and what changes before anything is saved.
                </p>
            </header>
            <div className="dp-import__pick">
                <input
                    ref={inputRef}
                    type="file"
                    accept=".zip,application/zip"
                    aria-label="Domain pack zip"
                    data-testid="dp-import-file"
                    onChange={(e) => void pick(e.target.files?.[0] ?? null)}
                />
            </div>
            {busy && <p className="dp-import__muted">Working...</p>}
            {error && <p className="dp-import__error" role="alert">{error}</p>}
            {preview && (
                <div data-testid="dp-import-preview">
                    {manifestId && (
                        <p className="dp-import__ident" data-testid="dp-import-ident">
                            {manifestId}@{version}
                            {preview.existing && (
                                <span className="dp-import__muted">
                                    {' '}(installed: {preview.existing.version})
                                </span>
                            )}
                        </p>
                    )}
                    <IssueGroups title="Errors" issues={preview.errors} testId="dp-import-errors" />
                    <IssueGroups title="Warnings" issues={preview.warnings} testId="dp-import-warnings" />
                    {preview.summary && (
                        <ul className="dp-import__summary" data-testid="dp-import-summary">
                            {SUMMARY_LABEL.map(([key, label]) => (
                                <li key={key}>
                                    <strong>{String(preview.summary![key])}</strong> {label}
                                </li>
                            ))}
                            {preview.summary.tool_catalog && <li>tool catalog</li>}
                            {preview.summary.project_template && <li>project template</li>}
                        </ul>
                    )}
                    {preview.valid && preview.unchanged && (
                        <p className="dp-import__muted" data-testid="dp-import-unchanged">
                            Unchanged: identical to the installed version, nothing to import.
                        </p>
                    )}
                    {preview.valid && !preview.unchanged && preview.diff && <DiffView diff={preview.diff} />}
                </div>
            )}
            {done && <p className="dp-import__ok" data-testid="dp-import-done">{done}</p>}
            {done && doneWarnings.length > 0 && (
                <IssueGroups title="Warnings" issues={doneWarnings} testId="dp-import-done-warnings" />
            )}
            <div className="dp-import__actions">
                <button
                    type="button"
                    className="btn btn-primary"
                    disabled={!canImport || (preview?.unchanged ?? false)}
                    onClick={() => void commit()}
                    data-testid="dp-import-commit"
                >
                    Import
                </button>
            </div>
        </section>
    );
}
