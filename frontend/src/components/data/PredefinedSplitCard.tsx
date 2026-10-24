/**
 * "Use my own train / val / test files" — import files whose split
 * membership is already decided. Nothing is re-divided or shuffled; the
 * result card shows what landed where, the duplicate counts and the leak
 * report, and a failed import shows the grouped reasons with row numbers.
 */
import { useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { parseErrorEnvelope } from '../../api/errors';
import {
    importPredefinedSplit,
    type OnLeak,
    type PredefinedSplitManifest,
} from '../../api/predefinedSplit';
import './PredefinedSplitCard.css';

export interface ChatTemplateOption {
    value: string;
    label: string;
}

interface Props {
    projectId: number;
    /** The split form's chat-template control (same options, same value). */
    chatTemplates?: ChatTemplateOption[];
    chatTemplate?: string;
    onChatTemplateChange?: (value: string) => void;
    /** False while the profile's default template applies (the split form sends nothing then). */
    sendChatTemplate?: boolean;
    onImported?: (manifest: PredefinedSplitManifest) => void;
}

interface Problem {
    message: string;
    fix: string;
    groups: { label: string; count: number; rows: number[] }[];
}

const ROLE_LABEL: Record<string, string> = { train: 'Train', val: 'Validation', test: 'Test' };

function rowsText(rows: number[]): string {
    return rows.length ? `lines ${rows.join(', ')}` : 'no sample lines';
}

/** Turn a 409 (leakage) / 422 (rejected rows) envelope into grouped lines. */
function toProblem(err: unknown): Problem {
    const env = parseErrorEnvelope(err);
    const meta = (env.metadata ?? {}) as Record<string, unknown>;
    const groups: Problem['groups'] = [];
    const rejected = meta.rejected as
        | Record<string, { by_reason: Record<string, { count: number; sample_rows: number[] }> }>
        | undefined;
    if (rejected) {
        for (const [role, info] of Object.entries(rejected)) {
            for (const [reason, slot] of Object.entries(info.by_reason)) {
                groups.push({
                    label: `${ROLE_LABEL[role] ?? role} file, ${reason.replace(/_/g, ' ')}`,
                    count: slot.count,
                    rows: slot.sample_rows,
                });
            }
        }
    }
    const pairs = meta.pairs as
        | { pair: string; counts: Record<string, number>; sample_rows: Record<string, number[]> }[]
        | undefined;
    if (pairs) {
        for (const p of pairs) {
            for (const [role, count] of Object.entries(p.counts)) {
                if (!count) continue;
                const other = p.pair
                    .split('_')
                    .filter((r) => r !== role)
                    .map((r) => ROLE_LABEL[r] ?? r)
                    .join('');
                groups.push({
                    label: `${ROLE_LABEL[role] ?? role} file rows that also appear in ${other}`,
                    count,
                    rows: p.sample_rows[role] ?? [],
                });
            }
        }
    }
    return { message: env.message, fix: env.actionableFix, groups };
}

export default function PredefinedSplitCard({
    projectId,
    chatTemplates,
    chatTemplate,
    onChatTemplateChange,
    sendChatTemplate = true,
    onImported,
}: Props) {
    const [train, setTrain] = useState<File | null>(null);
    const [val, setVal] = useState<File | null>(null);
    const [test, setTest] = useState<File | null>(null);
    const [onLeak, setOnLeak] = useState<OnLeak>('fail');
    const [fieldMapping, setFieldMapping] = useState('');
    const [busy, setBusy] = useState(false);
    const [problem, setProblem] = useState<Problem | null>(null);
    const [result, setResult] = useState<PredefinedSplitManifest | null>(null);

    const submit = async () => {
        if (!train) return;
        setBusy(true);
        setProblem(null);
        setResult(null);
        try {
            const manifest = await importPredefinedSplit(projectId, { train, val, test }, {
                onLeak,
                fieldMapping,
                chatTemplate: sendChatTemplate ? chatTemplate : null,
            });
            setResult(manifest);
            onImported?.(manifest);
        } catch (err) {
            setProblem(toProblem(err));
        } finally {
            setBusy(false);
        }
    };

    const picker = (id: string, label: string, hint: string, onPick: (f: File | null) => void) => (
        <label className="pdsplit__file" htmlFor={id}>
            <span>{label}</span>
            <input
                id={id}
                type="file"
                data-testid={id}
                accept=".jsonl,.json,.csv,.tsv,.xlsx,.parquet"
                onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            />
            <small>{hint}</small>
        </label>
    );

    return (
        <div className="pdsplit" data-testid="predefined-split-card">
            <p className="pdsplit__help">
                Already have train / validation / test files (for example a benchmark with a fixed test set)?
                Each row stays in the split you put it in: nothing is re-divided, shuffled or removed, and the
                files go through the same mapping as a normal split. Problems are reported by file line: line 1 of a
                CSV is the header, so its first row is line 2; blank lines count; for JSON arrays, Excel and Parquet
                files the number is the record number. Your test file stays out of any later re-split unless you
                import new files.
            </p>
            <div className="pdsplit__files">
                {picker('pdsplit-train', 'Train file (required)', 'JSONL, JSON, CSV, TSV, XLSX or Parquet', setTrain)}
                {picker('pdsplit-val', 'Validation file (optional)', 'Leave out to train without validation', setVal)}
                {picker('pdsplit-test', 'Test file (optional)', 'The answer key the model is scored on', setTest)}
            </div>
            <label className="pdsplit__field" htmlFor="pdsplit-onleak">
                <span>If a train row also appears in validation or test</span>
                <select
                    id="pdsplit-onleak"
                    data-testid="pdsplit-onleak"
                    value={onLeak}
                    onChange={(e) => setOnLeak(e.target.value as OnLeak)}
                >
                    <option value="fail">Stop and show me</option>
                    <option value="drop_from_train">Drop the overlapping rows from train</option>
                </select>
            </label>
            {chatTemplates && chatTemplate !== undefined && onChatTemplateChange && (
                <label className="pdsplit__field" htmlFor="pdsplit-template">
                    <span>Chat template (same choice as the split form below)</span>
                    <select
                        id="pdsplit-template"
                        data-testid="pdsplit-template"
                        value={chatTemplate}
                        onChange={(e) => onChatTemplateChange(e.target.value)}
                    >
                        {chatTemplates.map((t) => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                        ))}
                    </select>
                </label>
            )}
            <label className="pdsplit__field" htmlFor="pdsplit-mapping">
                <span>Column mapping (optional JSON object, for example {'{"prompt": "question"}'})</span>
                <input
                    id="pdsplit-mapping"
                    type="text"
                    value={fieldMapping}
                    onChange={(e) => setFieldMapping(e.target.value)}
                />
            </label>
            <div className="pdsplit__actions">
                <button className="btn-primary" data-testid="pdsplit-submit" disabled={!train || busy} onClick={() => void submit()}>
                    {busy ? 'Importing...' : 'Import my files'}
                </button>
            </div>

            {problem && (
                <div className="pdsplit__problem" role="alert" data-testid="pdsplit-problem">
                    <strong><AlertTriangle className="u-icon" aria-hidden="true" /> {problem.message}</strong>
                    {problem.fix && <p>{problem.fix}</p>}
                    {problem.groups.length > 0 && (
                        <ul>
                            {problem.groups.map((g) => (
                                <li key={g.label}>
                                    {g.label}: {g.count} row(s) ({rowsText(g.rows)}
                                    {g.count > g.rows.length ? ', first 20 shown' : ''})
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            )}

            {result && (
                <div className="pdsplit__result" data-testid="pdsplit-result">
                    <h4>Imported with your own split (dataset version {result.prepared_version ?? 'n/a'})</h4>
                    <table className="pdsplit__table">
                        <thead>
                            <tr><th>Split</th><th>Rows</th><th>Repeated inputs</th><th>Source file</th></tr>
                        </thead>
                        <tbody>
                            {['train', 'val', 'test'].map((role) => (
                                <tr key={role}>
                                    <td>{ROLE_LABEL[role]}</td>
                                    <td>{result.splits[role] ?? 0}</td>
                                    <td>{result.duplicate_report?.[role] ?? 0}</td>
                                    <td>{result.source_files?.[role]?.filename ?? 'not provided'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p className="pdsplit__help">
                        Repeated inputs are rows whose text already appears earlier in the same split. They were kept.
                    </p>
                    {result.leak_report.dropped_from_train > 0 ? (
                        <p data-testid="pdsplit-leak">
                            Dropped {result.leak_report.dropped_from_train} train row(s) that also appear in validation or test
                            ({rowsText(result.leak_report.dropped_sample_rows ?? [])} of the train file). Validation and test were not changed.
                        </p>
                    ) : (
                        <p data-testid="pdsplit-leak">No train, validation or test rows overlap.</p>
                    )}
                </div>
            )}
        </div>
    );
}
