/**
 * GoldFileManager — import, replace and clear an answer-key set from the UI, for any task shape.
 *
 * The set is labelled with its dataset, type and row count (picked-data provenance). A file with
 * rows that cannot be used is not imported; the unusable rows are listed grouped by reason, the
 * user selects the ones to drop (per row, per group, or all), and the import proceeds once every
 * unusable row is dropped. Never all-or-nothing: each drop is a step, and what is left is shown.
 * Deleting chosen rows lives in the entries list (GoldSetPanel).
 */

import { useRef, useState } from 'react';

import {
    clearGoldSet,
    goldErrorMessage,
    importGoldFile,
    type GoldDatasetLabel,
    type GoldImportMode,
    type GoldRejects,
    type GoldType,
} from '../../api/goldFiles';
import { toast } from '../../stores/toastStore';
import './GoldFileManager.css';

const TYPE_LABEL: Record<GoldType, string> = {
    gold_dev: 'Practice answer key',
    gold_test: 'Final answer key',
};

interface Props {
    projectId: number;
    datasetType: GoldType;
    label: GoldDatasetLabel | null;
    onChanged: () => void | Promise<void>;
}

export default function GoldFileManager({ projectId, datasetType, label, onChanged }: Props) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [file, setFile] = useState<File | null>(null);
    const [mode, setMode] = useState<GoldImportMode>('append');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [rejected, setRejected] = useState<GoldRejects | null>(null);
    const [usable, setUsable] = useState(0);
    const [dropped, setDropped] = useState<number[]>([]);
    const [selected, setSelected] = useState<Set<number>>(new Set());

    const locked = label?.is_locked === true;
    const rowCount = label?.record_count ?? 0;
    const setName = `${TYPE_LABEL[datasetType]} (${datasetType})`;

    const reset = () => {
        setFile(null);
        setRejected(null);
        setDropped([]);
        setSelected(new Set());
        setUsable(0);
        if (inputRef.current) inputRef.current.value = '';
    };

    const run = async (dropLines: number[]) => {
        if (!file) return;
        if (mode === 'replace' && rowCount > 0 && dropLines.length === dropped.length) {
            const ok = window.confirm(
                `Replace all ${rowCount} rows of the ${setName} with ${file.name}? The current rows are kept as an earlier version.`,
            );
            if (!ok) return;
        }
        setBusy(true);
        setError(null);
        try {
            const out = await importGoldFile(projectId, file, { datasetType, mode, dropLines });
            if ('result' in out) {
                const r = out.result;
                toast.success(
                    `${mode === 'replace' ? 'Replaced with' : 'Imported'} ${r.imported} row${r.imported === 1 ? '' : 's'} from ${r.source_filename}`
                    + `${r.dropped_rows ? ` (${r.dropped_rows} dropped)` : ''}. ${setName} now has ${r.total}.`,
                    5000,
                );
                reset();
                await onChanged();
            } else {
                setRejected(out.rejected);
                setUsable(out.usable);
                setDropped(dropLines);
                setSelected(new Set());
            }
        } catch (err) {
            setError(goldErrorMessage(err));
        } finally {
            setBusy(false);
        }
    };

    const toggle = (lines: number[], on: boolean) => {
        setSelected((prev) => {
            const next = new Set(prev);
            for (const line of lines) {
                if (on) next.add(line); else next.delete(line);
            }
            return next;
        });
    };

    const clear = async () => {
        if (!window.confirm(`Delete all ${rowCount} rows of the ${setName}? They are kept as an earlier version.`)) return;
        setBusy(true);
        setError(null);
        try {
            await clearGoldSet(projectId, datasetType);
            toast.success(`${setName} cleared.`, 4000);
            await onChanged();
        } catch (err) {
            setError(goldErrorMessage(err));
        } finally {
            setBusy(false);
        }
    };

    const groups = rejected ? Object.entries(rejected.by_reason) : [];
    const allLines = groups.flatMap(([, g]) => g.lines);

    return (
        <div className="gold-files" data-testid="gold-file-manager">
            <p className="gold-files__label" data-testid="gold-set-label">
                <strong>{setName}</strong>
                {label ? <> · dataset &quot;{label.name}&quot; · {rowCount} row{rowCount === 1 ? '' : 's'}</> : null}
                {locked ? ' · locked' : ''}
            </p>

            {locked ? (
                <p className="u-hint" data-testid="gold-files-locked">
                    This answer key is locked, so rows cannot be imported, replaced or deleted.
                </p>
            ) : (
                <div className="gold-files__row">
                    <label htmlFor="gold-file-input">Answer-key file (JSONL, JSON, CSV or TSV)</label>
                    <input
                        id="gold-file-input"
                        ref={inputRef}
                        type="file"
                        accept=".jsonl,.ndjson,.json,.csv,.tsv"
                        onChange={(e) => { setFile(e.target.files?.[0] ?? null); setRejected(null); setDropped([]); setError(null); }}
                        data-testid="gold-file-input"
                    />
                    <select
                        className="input u-w-auto"
                        value={mode}
                        onChange={(e) => setMode(e.target.value as GoldImportMode)}
                        aria-label="Import mode"
                        data-testid="gold-file-mode"
                    >
                        <option value="append">Add to this set</option>
                        <option value="replace">Replace this set</option>
                    </select>
                    <button
                        type="button"
                        className="btn btn-secondary"
                        disabled={!file || busy || rejected !== null}
                        onClick={() => { void run([]); }}
                        data-testid="gold-file-import"
                    >
                        {mode === 'replace' ? 'Replace from file' : 'Import file'}
                    </button>
                    {rowCount > 0 && (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            disabled={busy}
                            onClick={() => { void clear(); }}
                            data-testid="gold-clear-set"
                        >
                            Clear set
                        </button>
                    )}
                </div>
            )}

            {error && <p className="gold-files__error" role="alert" data-testid="gold-file-error">{error}</p>}

            {rejected && (
                <div className="gold-files__rejects" data-testid="gold-file-rejects">
                    <p>
                        <strong>{rejected.count} row{rejected.count === 1 ? '' : 's'} in {file?.name} cannot be used</strong>
                        {' '}({usable} usable{dropped.length > 0 ? `, ${dropped.length} already dropped` : ''}).
                        Nothing is imported until every unusable row is dropped. Select the rows to drop, grouped by reason.
                    </p>
                    {groups.map(([reason, group]) => {
                        const all = group.lines.every((l) => selected.has(l));
                        return (
                            <fieldset key={reason} className="gold-files__group" data-testid={`gold-reject-group-${reason}`}>
                                <legend>
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={all}
                                            onChange={(e) => toggle(group.lines, e.target.checked)}
                                            data-testid={`gold-reject-group-select-${reason}`}
                                        />
                                        {' '}{group.label} ({group.count})
                                    </label>
                                </legend>
                                <ul>
                                    {group.rows.map((row) => (
                                        <li key={row.line}>
                                            <label>
                                                <input
                                                    type="checkbox"
                                                    checked={selected.has(row.line)}
                                                    onChange={(e) => toggle([row.line], e.target.checked)}
                                                    data-testid={`gold-reject-row-${row.line}`}
                                                />
                                                {' '}line {row.line}: <code>{row.preview}</code>
                                            </label>
                                        </li>
                                    ))}
                                    {group.count > group.rows.length && (
                                        <li className="u-hint">
                                            …and {group.count - group.rows.length} more (select the group to drop them all).
                                        </li>
                                    )}
                                </ul>
                            </fieldset>
                        );
                    })}
                    <div className="gold-files__row">
                        <button type="button" className="btn btn-secondary" onClick={() => toggle(allLines, true)} data-testid="gold-reject-select-all">
                            Select all
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            disabled={busy || selected.size === 0}
                            onClick={() => { void run([...dropped, ...selected]); }}
                            data-testid="gold-reject-drop"
                        >
                            Drop {selected.size} selected row{selected.size === 1 ? '' : 's'}
                            {selected.size === allLines.length ? ' and import' : ''}
                        </button>
                        <button type="button" className="btn btn-link" onClick={reset} data-testid="gold-reject-cancel">
                            Cancel import
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
