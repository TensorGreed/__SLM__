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
    type AdapterWiring,
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
    /** Rows of the project's own (imported) test file while it is active, else null/undefined. */
    ownTestRows?: number | null;
    /** The project's current adapter wiring (the split form's): sent with any column mapping so the
     *  mapping refines it instead of resetting the adapter. */
    adapter?: AdapterWiring | null;
}

interface Problem {
    message: string;
    fix: string;
    groups: { label: string; count: number; rows: number[] }[];
    code: string;
    /** Set on a 422 where no column is recognised as the input: the file's columns. */
    unrecognisedColumns: string[] | null;
}

/** Provenance / id-like names (the server's NON_CONTENT_KEYS plus index-like names). */
const PROVENANCE_NAMES = new Set([
    'id', 'uuid', 'source', 'split', 'timestamp', 'created_at', 'updated_at', 'metadata', 'idx', 'index', 'guid',
]);
/** Target-named columns: what the model should produce (the server's target names). */
const TARGET_NAMES = new Set([
    'answer', 'output', 'completion', 'response', 'target', 'label', 'labels', 'class', 'category', 'output_label',
    'structured_output', 'json', 'extracted', 'entities', 'target_text', 'chosen', 'rejected', 'tool_calls',
    'expected', 'ner_tags', 'pos_tags', 'chunk_tags', 'tags', 'bio_tags', 'spans', 'label_ids', 'caption',
    'transcript', 'summary',
]);

function isProvenance(column: string): boolean {
    const name = column.trim().toLowerCase();
    return PROVENANCE_NAMES.has(name) || name.startsWith('_') || name.endsWith('_id') || name.endsWith('_ids');
}

/** A column that may be preselected as the model's input: not provenance, id-like or target-named. */
export function isInputCandidate(column: string): boolean {
    return !isProvenance(column) && !TARGET_NAMES.has(column.trim().toLowerCase());
}

/** The mapper's starting selections: the first plausible input column, and a different answer column
 *  (a target-named one first, else the next non-provenance column). Never the same column twice. */
export function defaultMapping(columns: string[]): { input: string; answer: string } {
    const input = columns.find(isInputCandidate) ?? '';
    const others = columns.filter((c) => c !== input && !isProvenance(c));
    const answer = others.find((c) => TARGET_NAMES.has(c.trim().toLowerCase())) ?? others[0] ?? '';
    return { input, answer };
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
    const columns = meta.reason === 'input_column_unrecognised' && Array.isArray(meta.columns)
        ? (meta.columns as unknown[]).map(String)
        : null;
    return { message: env.message, fix: env.actionableFix, groups, code: env.errorCode, unrecognisedColumns: columns };
}

export default function PredefinedSplitCard({
    projectId,
    chatTemplates,
    chatTemplate,
    onChatTemplateChange,
    sendChatTemplate = true,
    onImported,
    ownTestRows = null,
    adapter = null,
}: Props) {
    const [train, setTrain] = useState<File | null>(null);
    const [val, setVal] = useState<File | null>(null);
    const [test, setTest] = useState<File | null>(null);
    const [onLeak, setOnLeak] = useState<OnLeak>('fail');
    const [fieldMapping, setFieldMapping] = useState('');
    const [dropOwnTest, setDropOwnTest] = useState(false);
    const [mapInput, setMapInput] = useState('');
    const [mapAnswer, setMapAnswer] = useState('');
    const [busy, setBusy] = useState(false);
    const [problem, setProblem] = useState<Problem | null>(null);
    const [result, setResult] = useState<PredefinedSplitManifest | null>(null);

    const submit = async (mappingOverride?: string) => {
        if (!train) return;
        setBusy(true);
        setProblem(null);
        setResult(null);
        try {
            const manifest = await importPredefinedSplit(projectId, { train, val, test }, {
                onLeak,
                fieldMapping: mappingOverride ?? fieldMapping,
                chatTemplate: sendChatTemplate ? chatTemplate : null,
                dropExistingTest: !test && dropOwnTest,
                adapter,
            });
            setResult(manifest);
            setDropOwnTest(false);
            onImported?.(manifest);
        } catch (err) {
            const next = toProblem(err);
            setProblem(next);
            if (next.unrecognisedColumns) {
                const start = defaultMapping(next.unrecognisedColumns);
                setMapInput(start.input);
                setMapAnswer(start.answer);
            }
        } finally {
            setBusy(false);
        }
    };

    const applyMapping = () => {
        if (!mapInput) return;
        const mapping: Record<string, string> = { question: mapInput };
        if (mapAnswer && mapAnswer !== mapInput) mapping.answer = mapAnswer;
        const text = JSON.stringify(mapping);
        setFieldMapping(text);
        void submit(text);
    };

    const showDropOwnTest = !test && (ownTestRows !== null || problem?.code === 'PREDEFINED_TEST_WOULD_BE_DROPPED');

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
                Your files go in as they are: nothing is re-divided or shuffled, and they go through the same
                mapping as a normal split. If a train row also appears in validation or test, the import stops and
                lists it, unless you choose to drop those overlapping rows from train (that removes them from train
                only; validation and test are never edited). Problems are reported by file line: line 1 of a
                CSV is the header, so its first row is line 2; blank lines count; for JSON arrays, Excel and Parquet
                files the number is the record number. Your test file stays out of any later re-split unless you
                import new files, and a later re-split re-draws validation from train + val.
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
            {showDropOwnTest && (
                <label className="pdsplit__field" htmlFor="pdsplit-drop-test">
                    <span>
                        <input
                            id="pdsplit-drop-test"
                            type="checkbox"
                            data-testid="pdsplit-drop-test"
                            checked={dropOwnTest}
                            onChange={(e) => setDropOwnTest(e.target.checked)}
                        />{' '}
                        Drop my current test file{ownTestRows ? ` (${ownTestRows} rows)` : ''}
                    </span>
                    <small>
                        Importing without a test file would retire the test file you imported earlier. Tick this to
                        confirm, or choose a new test file above.
                    </small>
                </label>
            )}
            <div className="pdsplit__actions">
                <button className="btn-primary" data-testid="pdsplit-submit" disabled={!train || busy} onClick={() => void submit()}>
                    {busy ? 'Importing...' : 'Import my files'}
                </button>
            </div>

            {problem && (
                <div className="pdsplit__problem" role="alert" data-testid="pdsplit-problem">
                    <strong><AlertTriangle className="u-icon" aria-hidden="true" /> {problem.message}</strong>
                    {problem.fix && <p>{problem.fix}</p>}
                    {problem.unrecognisedColumns && (
                        <div className="pdsplit__mapper" data-testid="pdsplit-mapper">
                            <p>The file's columns: {problem.unrecognisedColumns.join(', ')}</p>
                            <label className="pdsplit__field" htmlFor="pdsplit-map-input">
                                <span>Question / text column (what the model is given)</span>
                                <select id="pdsplit-map-input" data-testid="pdsplit-map-input" value={mapInput} onChange={(e) => {
                                    setMapInput(e.target.value);
                                    if (e.target.value === mapAnswer) setMapAnswer('');
                                }}>
                                    <option value="" disabled>(choose a column)</option>
                                    {problem.unrecognisedColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </label>
                            <label className="pdsplit__field" htmlFor="pdsplit-map-answer">
                                <span>Answer column (what it should produce)</span>
                                <select id="pdsplit-map-answer" data-testid="pdsplit-map-answer" value={mapAnswer} onChange={(e) => setMapAnswer(e.target.value)}>
                                    <option value="">(none)</option>
                                    {problem.unrecognisedColumns.filter((c) => c !== mapInput).map((c) => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </label>
                            <button className="btn-secondary" type="button" data-testid="pdsplit-map-apply" disabled={!mapInput || busy} onClick={applyMapping}>
                                Use this mapping and import again
                            </button>
                        </div>
                    )}
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
                    {result.overlap_input_columns && result.overlap_input_columns.length > 0 && (
                        <p data-testid="pdsplit-overlap-columns">
                            Overlap checked on: {result.overlap_input_columns.join(' + ')}
                        </p>
                    )}
                    <p className="pdsplit__help">
                        Repeated inputs are rows whose text already appears earlier in the same split. They were kept.
                    </p>
                    {result.dropped_existing_test_document_ids && result.dropped_existing_test_document_ids.length > 0 && (
                        <p data-testid="pdsplit-dropped-test">
                            Your earlier test file was dropped; this import has no test file.
                        </p>
                    )}
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
