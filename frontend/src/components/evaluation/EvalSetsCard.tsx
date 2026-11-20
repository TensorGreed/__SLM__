/**
 * EvalSetsCard — named, versioned eval sets and their prompt-format variants (any task shape).
 *
 * Lives under Results › Verdict › "Test sets" (an Advanced section), never on the main path. A set is made
 * from a file (the answer-key import: unusable rows are grouped by reason, selected and dropped, never
 * all-or-nothing; rows flagged as leaking training data may instead be kept on purpose, which the version
 * records and every result on it shows) or copied from a built-in set or another eval set. Every change is a new immutable
 * version; a variant renders the same rows with the system message stripped, replaced or added. Each
 * set + version + variant that has results can open the base-vs-fine-tuned view on exactly that set.
 */

import { useRef, useState } from 'react';

import {
    addEvalSetVariant,
    createEvalSetFromSource,
    crossVariantBaseLabels,
    deleteEvalSet,
    deleteEvalSetRows,
    deleteEvalSetVariant,
    evalDatasetLabel,
    evalSetErrorMessage,
    fetchEvalSetRows,
    keptLeakText,
    lastEvalText,
    LEAK_REASON,
    lockEvalSet,
    uploadEvalSet,
    type EvalSet,
    type EvalSetRowsPage,
    type SplitSource,
    type VariantMode,
} from '../../api/evalSets';
import type { GoldRejects } from '../../api/goldFiles';
import { toast } from '../../stores/toastStore';
import { schemaUsedForRow, taskReadsRowContext } from '../../types/rowContext';
import RowContextChip from '../shared/RowContextChip';
import SftLiftPanel from './SftLiftPanel';
import './EvalSetsCard.css';

const SPLIT_LABEL: Record<SplitSource, string> = {
    test: 'Test examples',
    validation: 'Validation',
    gold_test: 'Final answer key',
    gold_dev: 'Practice answer key',
};

const MODE_LABEL: Record<VariantMode, string> = {
    strip: 'Strip the system message',
    replace: 'Replace it with this text',
    add: 'Add this text where a row has none',
};

interface Props {
    projectId: number;
    sets: EvalSet[];
    onChanged: () => void | Promise<void>;
}

interface PendingUpload {
    file: File;
    name: string;
    slug?: string;
    rejected: GoldRejects;
    usable: number;
    dropped: number[];
    kept: number[];
}

function rowPreview(row: Record<string, unknown>): string {
    const text = JSON.stringify(row);
    return text.length > 160 ? `${text.slice(0, 160)}…` : text;
}

export default function EvalSetsCard({ projectId, sets, onChanged }: Props) {
    const fileRef = useRef<HTMLInputElement>(null);
    const [newName, setNewName] = useState('');
    const [newFile, setNewFile] = useState<File | null>(null);
    const [copySplit, setCopySplit] = useState<SplitSource>('test');
    const [copyName, setCopyName] = useState('');
    // Built-in sets take no variants: "copy + variant" makes an eval set with one in a single call.
    const [copyWithVariant, setCopyWithVariant] = useState(false);
    const [copyVariantName, setCopyVariantName] = useState('');
    const [copyVariantMode, setCopyVariantMode] = useState<VariantMode>('strip');
    const [copyVariantText, setCopyVariantText] = useState('');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [pending, setPending] = useState<PendingUpload | null>(null);
    const [selectedLines, setSelectedLines] = useState<Set<number>>(new Set());
    const [rowsView, setRowsView] = useState<EvalSetRowsPage | null>(null);
    const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
    const [variantFor, setVariantFor] = useState<string | null>(null);
    const [variantName, setVariantName] = useState('');
    const [variantMode, setVariantMode] = useState<VariantMode>('strip');
    const [variantText, setVariantText] = useState('');
    const [liftLabel, setLiftLabel] = useState<string | null>(null);
    const [liftBaseLabel, setLiftBaseLabel] = useState<string | null>(null);

    const act = async (fn: () => Promise<void>) => {
        setBusy(true);
        setError(null);
        try {
            await fn();
        } catch (err) {
            setError(evalSetErrorMessage(err));
        } finally {
            setBusy(false);
        }
    };

    const doUpload = (
        file: File, name: string, slug: string | undefined, dropLines: number[], keepLeakLines: number[] = [],
    ) => act(async () => {
        const out = await uploadEvalSet(projectId, file, { name, slug, dropLines, keepLeakLines });
        if ('set' in out) {
            const s = out.set;
            toast.success(
                `${slug ? 'New version of' : 'Created'} eval set "${s.name}" (${s.slug} v${s.current_version}, ${s.rows} rows`
                + `${dropLines.length ? `, ${dropLines.length} dropped` : ''}`
                + `${keepLeakLines.length ? `, ${keepLeakLines.length} kept despite a leak flag` : ''}).`,
                5000,
            );
            for (const w of Object.values(out.warnings ?? {})) {
                toast.warning(`${w.count} row${w.count === 1 ? '' : 's'} (line${w.count === 1 ? '' : 's'} ${w.lines.slice(0, 5).join(', ')}): ${w.label}.`, 9000);
            }
            setPending(null);
            setSelectedLines(new Set());
            setNewFile(null);
            setNewName('');
            if (fileRef.current) fileRef.current.value = '';
            await onChanged();
        } else {
            setPending({ file, name, slug, rejected: out.rejected, usable: out.usable, dropped: dropLines, kept: keepLeakLines });
            setSelectedLines(new Set());
        }
    });

    const copyFromSplit = () => act(async () => {
        const name = copyName.trim() || `${SPLIT_LABEL[copySplit]} copy`;
        const variant = copyWithVariant
            ? { name: copyVariantName.trim(), system: copyVariantMode, ...(copyVariantMode === 'strip' ? {} : { text: copyVariantText }) }
            : undefined;
        const s = await createEvalSetFromSource(projectId, name, { kind: 'split', split: copySplit }, variant);
        toast.success(
            `Created eval set "${s.name}" from ${SPLIT_LABEL[copySplit]} (${s.rows} rows`
            + `${variant ? `, variant "${variant.name}"` : ''}).`,
            5000,
        );
        setCopyName('');
        setCopyWithVariant(false);
        setCopyVariantName('');
        setCopyVariantText('');
        await onChanged();
    });

    const openRows = (s: EvalSet, version: number) => act(async () => {
        setRowsView(await fetchEvalSetRows(projectId, s.slug, version));
        setSelectedRows(new Set());
    });

    const deleteSelectedRows = () => act(async () => {
        if (!rowsView) return;
        const positions = [...selectedRows];
        const out = await deleteEvalSetRows(projectId, rowsView.slug, positions, rowsView.version);
        toast.success(
            `Deleted ${out.deleted} row${out.deleted === 1 ? '' : 's'}: ${out.eval_set.slug} is now v${out.eval_set.current_version}.`,
            5000,
        );
        setRowsView(null);
        await onChanged();
    });

    const saveVariant = (slug: string) => act(async () => {
        await addEvalSetVariant(projectId, slug, variantName.trim(), variantMode, variantText);
        toast.success(`Variant "${variantName.trim()}" added to ${slug}.`, 4000);
        setVariantFor(null);
        setVariantName('');
        setVariantText('');
        setVariantMode('strip');
        await onChanged();
    });

    const toggleLines = (lines: number[], on: boolean) => setSelectedLines((prev) => {
        const next = new Set(prev);
        for (const l of lines) { if (on) next.add(l); else next.delete(l); }
        return next;
    });

    const groups = pending ? Object.entries(pending.rejected.by_reason) : [];
    const allLines = groups.flatMap(([, g]) => g.lines);
    // Only rows flagged as leaking training data may be kept (an eval-set choice: training-direction
    // leaks are never keepable). Keeping is per row or per group, never all-or-nothing.
    const leakLines = new Set(pending?.rejected.by_reason[LEAK_REASON]?.lines ?? []);
    const canKeep = selectedLines.size > 0 && [...selectedLines].every((l) => leakLines.has(l));

    return (
        <div className="eval-sets" data-testid="eval-sets-card">
            <p className="u-hint u-mt0">
                Named test sets beside the built-in ones, for any task. Every change makes a new version (old
                versions stay for the results that used them). A prompt-format variant scores the same rows with
                the system message stripped, replaced or added. Pick a set in the Dataset list above.
            </p>

            {sets.length === 0 ? (
                <p className="u-hint" data-testid="eval-sets-empty">No eval sets yet.</p>
            ) : (
                <ul className="eval-sets__list">
                    {sets.map((s) => (
                        <li key={s.slug} className="eval-sets__item" data-testid={`eval-set-${s.slug}`}>
                            <div className="eval-sets__head">
                                <strong>{s.name}</strong>
                                <span className="u-meta">
                                    {' '}· {s.slug} · v{s.current_version} · {s.rows} row{s.rows === 1 ? '' : 's'}
                                    {s.versions.length > 1 ? ` · ${s.versions.length} versions` : ''}
                                    {s.is_locked ? ' · locked' : ''}
                                    {(() => {
                                        const cur = s.versions.find((v) => v.version === s.current_version);
                                        const text = keptLeakText(cur?.kept_despite_leak);
                                        return text ? ` · ${text}` : '';
                                    })()}
                                </span>
                            </div>
                            <div className="eval-sets__variants" data-testid={`eval-set-variants-${s.slug}`}>
                                Prompt formats: as in the rows
                                {s.variants.map((v) => (
                                    <span key={v.name} className="eval-sets__variant">
                                        {' '}· <strong>{v.name}</strong> ({v.label})
                                        <button
                                            type="button"
                                            className="btn btn-link"
                                            disabled={busy}
                                            onClick={() => { void act(async () => { await deleteEvalSetVariant(projectId, s.slug, v.name); await onChanged(); }); }}
                                            aria-label="Remove this variant"
                                            data-testid={`eval-set-variant-delete-${s.slug}-${v.name}`}
                                        >
                                            remove
                                        </button>
                                    </span>
                                ))}
                            </div>
                            <div className="eval-sets__row">
                                <button type="button" className="btn btn-secondary" disabled={busy}
                                    onClick={() => { void openRows(s, s.current_version); }}
                                    data-testid={`eval-set-rows-${s.slug}`}>
                                    View rows
                                </button>
                                <button type="button" className="btn btn-secondary" disabled={busy}
                                    onClick={() => setVariantFor(variantFor === s.slug ? null : s.slug)}
                                    data-testid={`eval-set-add-variant-${s.slug}`}>
                                    Add a variant
                                </button>
                                {!s.is_locked && (
                                    <label className="btn btn-secondary eval-sets__file">
                                        New version from file
                                        <input
                                            type="file"
                                            accept=".jsonl,.ndjson,.json,.csv,.tsv"
                                            onChange={(e) => {
                                                const f = e.target.files?.[0];
                                                if (f) void doUpload(f, s.name, s.slug, []);
                                                e.target.value = '';
                                            }}
                                            data-testid={`eval-set-version-file-${s.slug}`}
                                        />
                                    </label>
                                )}
                                {!s.is_locked && (
                                    <button type="button" className="btn btn-secondary" disabled={busy}
                                        onClick={() => { void act(async () => { await lockEvalSet(projectId, s.slug); await onChanged(); }); }}
                                        data-testid={`eval-set-lock-${s.slug}`}>
                                        Lock
                                    </button>
                                )}
                                {!s.is_locked && (
                                    <button type="button" className="btn btn-link" disabled={busy}
                                        onClick={() => {
                                            if (!window.confirm(`Delete eval set "${s.name}" and all its versions? Past results keep their label.`)) return;
                                            void act(async () => { await deleteEvalSet(projectId, s.slug); await onChanged(); });
                                        }}
                                        data-testid={`eval-set-delete-${s.slug}`}>
                                        Delete set
                                    </button>
                                )}
                            </div>

                            {variantFor === s.slug && (
                                <div className="eval-sets__form" data-testid={`eval-set-variant-form-${s.slug}`}>
                                    <label htmlFor={`variant-name-${s.slug}`}>Variant name</label>
                                    <input id={`variant-name-${s.slug}`} className="input u-w-auto" value={variantName}
                                        onChange={(e) => setVariantName(e.target.value)} placeholder="native"
                                        data-testid="eval-set-variant-name" />
                                    <select className="input u-w-auto" value={variantMode} aria-label="System message"
                                        onChange={(e) => setVariantMode(e.target.value as VariantMode)}
                                        data-testid="eval-set-variant-mode">
                                        {(Object.keys(MODE_LABEL) as VariantMode[]).map((m) => (
                                            <option key={m} value={m}>{MODE_LABEL[m]}</option>
                                        ))}
                                    </select>
                                    {variantMode !== 'strip' && (
                                        <textarea className="input eval-sets__text" value={variantText} aria-label="System message text"
                                            onChange={(e) => setVariantText(e.target.value)} data-testid="eval-set-variant-text" />
                                    )}
                                    <button type="button" className="btn btn-primary"
                                        disabled={busy || !variantName.trim() || (variantMode !== 'strip' && !variantText.trim())}
                                        onClick={() => { void saveVariant(s.slug); }}
                                        data-testid="eval-set-variant-save">
                                        Save variant
                                    </button>
                                </div>
                            )}

                            {Object.keys(s.last_evals ?? {}).length > 0 && (
                                <div className="eval-sets__evals" data-testid={`eval-set-evals-${s.slug}`}>
                                    {Object.entries(s.last_evals ?? {}).map(([label, ev]) => (
                                        <div key={label} className="eval-sets__row">
                                            <span className="u-meta">
                                                {evalDatasetLabel(label)} · {lastEvalText(ev)}
                                            </span>
                                            <button type="button" className="btn btn-link"
                                                onClick={() => { setLiftLabel(liftLabel === label ? null : label); setLiftBaseLabel(null); }}
                                                data-testid={`eval-set-lift-${label}`}>
                                                {liftLabel === label ? 'Hide base vs fine-tuned' : 'Base vs fine-tuned on this set'}
                                            </button>
                                        </div>
                                    ))}
                                    {liftLabel && liftLabel.includes(`:${s.slug}@`) && (() => {
                                        // The base model on another variant of the same version (same rows): offered
                                        // as a labelled cross-variant pair.
                                        const others = crossVariantBaseLabels(s.last_evals ?? {}, liftLabel);
                                        return (
                                            <>
                                                {others.length > 0 && (
                                                    <label className="u-meta" data-testid="eval-set-lift-base-variant">
                                                        Base model on{' '}
                                                        <select className="input u-w-auto" value={liftBaseLabel ?? liftLabel}
                                                            aria-label="Base model's prompt format"
                                                            onChange={(e) => setLiftBaseLabel(e.target.value === liftLabel ? null : e.target.value)}>
                                                            <option value={liftLabel}>the same prompt format</option>
                                                            {others.map((o) => (
                                                                <option key={o} value={o}>{evalDatasetLabel(o)} (prompt differs too)</option>
                                                            ))}
                                                        </select>
                                                    </label>
                                                )}
                                                <SftLiftPanel projectId={projectId} datasetName={liftLabel}
                                                    baseDatasetName={liftBaseLabel} />
                                            </>
                                        );
                                    })()}
                                </div>
                            )}
                        </li>
                    ))}
                </ul>
            )}

            {rowsView && (
                <div className="eval-sets__rows" data-testid="eval-set-rows-view">
                    <p>
                        <strong>{rowsView.slug} v{rowsView.version}</strong> · {rowsView.total} row{rowsView.total === 1 ? '' : 's'}
                        {rowsView.total > rowsView.rows.length ? ` (first ${rowsView.rows.length} shown)` : ''}.
                        Deleting rows makes a new version.
                    </p>
                    <ul>
                        {rowsView.rows.map((r) => (
                            <li key={r.position}>
                                <label>
                                    <input type="checkbox" checked={selectedRows.has(r.position)}
                                        onChange={(e) => setSelectedRows((prev) => {
                                            const next = new Set(prev);
                                            if (e.target.checked) next.add(r.position); else next.delete(r.position);
                                            return next;
                                        })}
                                        data-testid={`eval-set-row-${r.position}`} />
                                    {' '}row {r.position + 1}: <code>{rowPreview(r.row)}</code>
                                </label>
                                <RowContextChip
                                    context={r.row.row_context as string | Record<string, unknown> | null | undefined}
                                    toolsUsed={taskReadsRowContext(rowsView, 'tools')}
                                    schemaUsed={schemaUsedForRow(rowsView, r.row)}
                                    testId={`eval-set-row-context-${r.position}`}
                                />
                            </li>
                        ))}
                    </ul>
                    <div className="eval-sets__row">
                        <button type="button" className="btn btn-primary" disabled={busy || selectedRows.size === 0}
                            onClick={() => { void deleteSelectedRows(); }} data-testid="eval-set-rows-delete">
                            Delete {selectedRows.size} selected row{selectedRows.size === 1 ? '' : 's'} (new version)
                        </button>
                        <button type="button" className="btn btn-link" onClick={() => setRowsView(null)}>Close</button>
                    </div>
                </div>
            )}

            <div className="eval-sets__form" data-testid="eval-set-create">
                <label htmlFor="eval-set-name">New eval set</label>
                <input id="eval-set-name" className="input u-w-auto" value={newName} placeholder="Name, e.g. fresh"
                    onChange={(e) => setNewName(e.target.value)} data-testid="eval-set-name" />
                <input ref={fileRef} type="file" accept=".jsonl,.ndjson,.json,.csv,.tsv" aria-label="Eval set file"
                    onChange={(e) => { setNewFile(e.target.files?.[0] ?? null); setPending(null); }}
                    data-testid="eval-set-file" />
                <button type="button" className="btn btn-secondary" disabled={busy || !newFile || !newName.trim() || pending !== null}
                    onClick={() => { if (newFile) void doUpload(newFile, newName.trim(), undefined, []); }}
                    data-testid="eval-set-upload">
                    Create from file
                </button>
            </div>
            <div className="eval-sets__form" data-testid="eval-set-copy">
                <label htmlFor="eval-set-copy-split">Or copy a built-in set</label>
                <select id="eval-set-copy-split" className="input u-w-auto" value={copySplit}
                    onChange={(e) => setCopySplit(e.target.value as SplitSource)} data-testid="eval-set-copy-split">
                    {(Object.keys(SPLIT_LABEL) as SplitSource[]).map((k) => (
                        <option key={k} value={k}>{SPLIT_LABEL[k]}</option>
                    ))}
                </select>
                <input className="input u-w-auto" value={copyName} placeholder="Name (optional)" aria-label="Copy name"
                    onChange={(e) => setCopyName(e.target.value)} data-testid="eval-set-copy-name" />
                <label className="eval-sets__check">
                    <input type="checkbox" checked={copyWithVariant}
                        onChange={(e) => setCopyWithVariant(e.target.checked)} data-testid="eval-set-copy-with-variant" />
                    {' '}and add a prompt-format variant
                </label>
                {copyWithVariant && (
                    <>
                        <input className="input u-w-auto" value={copyVariantName} placeholder="Variant name, e.g. native"
                            aria-label="Variant name" onChange={(e) => setCopyVariantName(e.target.value)}
                            data-testid="eval-set-copy-variant-name" />
                        <select className="input u-w-auto" value={copyVariantMode} aria-label="System message"
                            onChange={(e) => setCopyVariantMode(e.target.value as VariantMode)}
                            data-testid="eval-set-copy-variant-mode">
                            {(Object.keys(MODE_LABEL) as VariantMode[]).map((m) => (
                                <option key={m} value={m}>{MODE_LABEL[m]}</option>
                            ))}
                        </select>
                        {copyVariantMode !== 'strip' && (
                            <textarea className="input eval-sets__text" value={copyVariantText} aria-label="System message text"
                                onChange={(e) => setCopyVariantText(e.target.value)} data-testid="eval-set-copy-variant-text" />
                        )}
                    </>
                )}
                <button type="button" className="btn btn-secondary"
                    disabled={busy || (copyWithVariant && (!copyVariantName.trim() || (copyVariantMode !== 'strip' && !copyVariantText.trim())))}
                    onClick={() => { void copyFromSplit(); }} data-testid="eval-set-copy-button">
                    Copy into a new eval set
                </button>
                <p className="u-hint u-mt0">
                    Built-in sets take no prompt-format variants (they change on every re-split, and their results
                    are the project&apos;s headline numbers): copy one into an eval set to score it under a variant.
                </p>
            </div>

            {error && <p className="eval-sets__error" role="alert" data-testid="eval-sets-error">{error}</p>}

            {pending && (
                <div className="eval-sets__rejects" data-testid="eval-set-rejects">
                    <p>
                        <strong>{pending.rejected.count} row{pending.rejected.count === 1 ? '' : 's'} in {pending.file.name} cannot be used</strong>
                        {' '}({pending.usable} usable{pending.dropped.length ? `, ${pending.dropped.length} already dropped` : ''}
                        {pending.kept.length ? `, ${pending.kept.length} kept despite a leak flag` : ''}).
                        Nothing is written until every unusable row is dropped. Select the rows to drop, grouped by reason.
                        {leakLines.size > 0 && ' Rows that leak training data can instead be kept on purpose: they are scored as seen data, and the version and its results say so.'}
                    </p>
                    {groups.map(([reason, group]) => (
                        <fieldset key={reason} className="eval-sets__group" data-testid={`eval-set-reject-group-${reason}`}>
                            <legend>
                                <label>
                                    <input type="checkbox" checked={group.lines.every((l) => selectedLines.has(l))}
                                        onChange={(e) => toggleLines(group.lines, e.target.checked)}
                                        data-testid={`eval-set-reject-group-select-${reason}`} />
                                    {' '}{group.label} ({group.count})
                                </label>
                            </legend>
                            <ul>
                                {group.rows.map((row) => (
                                    <li key={row.line}>
                                        <label>
                                            <input type="checkbox" checked={selectedLines.has(row.line)}
                                                onChange={(e) => toggleLines([row.line], e.target.checked)}
                                                data-testid={`eval-set-reject-row-${row.line}`} />
                                            {' '}line {row.line}: <code>{row.preview}</code>
                                        </label>
                                    </li>
                                ))}
                            </ul>
                        </fieldset>
                    ))}
                    <div className="eval-sets__row">
                        <button type="button" className="btn btn-secondary" onClick={() => toggleLines(allLines, true)}>
                            Select all
                        </button>
                        <button type="button" className="btn btn-primary" disabled={busy || selectedLines.size === 0}
                            onClick={() => { void doUpload(pending.file, pending.name, pending.slug, [...pending.dropped, ...selectedLines], pending.kept); }}
                            data-testid="eval-set-reject-drop">
                            Drop {selectedLines.size} selected row{selectedLines.size === 1 ? '' : 's'}
                            {selectedLines.size === allLines.length ? ' and import' : ''}
                        </button>
                        {leakLines.size > 0 && (
                            <button type="button" className="btn btn-secondary" disabled={busy || !canKeep}
                                onClick={() => { void doUpload(pending.file, pending.name, pending.slug, pending.dropped, [...pending.kept, ...selectedLines]); }}
                                title="Only rows flagged as leaking training data can be kept"
                                data-testid="eval-set-reject-keep">
                                Keep {selectedLines.size} selected leak-flagged row{selectedLines.size === 1 ? '' : 's'} anyway
                            </button>
                        )}
                        <button type="button" className="btn btn-link" onClick={() => setPending(null)}>Cancel import</button>
                    </div>
                </div>
            )}
        </div>
    );
}
