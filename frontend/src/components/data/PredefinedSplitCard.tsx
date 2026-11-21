/**
 * "Use my own train / val / test files" — import files whose split
 * membership is already decided. Nothing is re-divided or shuffled; the
 * result card shows what landed where, the duplicate counts and the leak
 * report, and a failed import shows the grouped reasons with row numbers.
 *
 * UI simplification pass, phase 3: one multi-file pick assigns roles by file
 * name (``roleFromFileName``), editable per file; the per-role pickers, the
 * leak choice, the chat template and the column mapping sit under the card's
 * Advanced section (``own-files-options``). The request is unchanged.
 */
import { useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { parseErrorEnvelope } from '../../api/errors';
import AdvancedSection from '../shared/AdvancedSection';
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
    /** The page's ``?open=`` value (opens the card's Advanced section when it is ``own-files-options``). */
    openSection?: string | null;
}

export type FileRole = 'train' | 'val' | 'test';

/** One word of a file name per role: train / training, val / valid / validation / dev, test / testing,
 *  optionally followed by set / split / data and digits (``trainset``, ``dev2``, ``test_v2``). */
const ROLE_WORD: Record<FileRole, RegExp> = {
    train: /^train(ing)?(set|split|data)?\d*$/,
    val: /^(val|valid|validation|dev)(set|split|data)?\d*$/,
    test: /^test(ing)?(set|split|data)?\d*$/,
};

/** The role a file name suggests: one of its words (split on anything but letters and digits) is a
 *  role word (``train.jsonl``, ``squad_dev.jsonl``, ``my-test-v2.csv``; not ``devops`` or ``contest``).
 *  Null when the name suggests none or more than one. The user can change it per file. */
export function roleFromFileName(name: string): FileRole | null {
    const base = name.toLowerCase().replace(/^.*[\\/]/, '').replace(/\.[^.]*$/, '');
    const words = base.split(/[^a-z0-9]+/).filter(Boolean);
    const found = (Object.keys(ROLE_WORD) as FileRole[]).filter((role) => words.some((w) => ROLE_WORD[role].test(w)));
    return found.length === 1 ? found[0] : null;
}

export interface PickedFile {
    file: File;
    /** '' = not used. */
    role: FileRole | '';
}

/** Roles for a multi-file pick: each name's suggested role; the first file to suggest a role gets it. */
export function assignRoles(files: File[]): PickedFile[] {
    const taken = new Set<FileRole>();
    return files.map((file) => {
        const role = roleFromFileName(file.name);
        if (role && !taken.has(role)) {
            taken.add(role);
            return { file, role };
        }
        return { file, role: '' };
    });
}

interface Problem {
    message: string;
    fix: string;
    groups: { label: string; count: number; rows: number[] }[];
    code: string;
    /** Set on a 422 where no column is recognised as the input: the file's columns. */
    unrecognisedColumns: string[] | null;
    /** The columns the overlap check keyed on (sent with 409 / 422 errors), when the server says. */
    overlapColumns: string[];
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
    'transcript', 'summary', 'highlights', 'abstract',
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
    const overlapColumns = Array.isArray(meta.overlap_input_columns) ? (meta.overlap_input_columns as unknown[]).map(String) : [];
    return { message: env.message, fix: env.actionableFix, groups, code: env.errorCode, unrecognisedColumns: columns, overlapColumns };
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
    openSection = null,
}: Props) {
    const [picked, setPicked] = useState<PickedFile[]>([]);
    const fileFor = (role: FileRole): File | null => picked.find((p) => p.role === role)?.file ?? null;
    const train = fileFor('train');
    const val = fileFor('val');
    const test = fileFor('test');
    const duplicateRoles = (['train', 'val', 'test'] as FileRole[]).filter((role) => picked.filter((p) => p.role === role).length > 1);
    const [onLeak, setOnLeak] = useState<OnLeak>('fail');
    const [fieldMapping, setFieldMapping] = useState('');
    const [dropOwnTest, setDropOwnTest] = useState(false);
    const [mapInput, setMapInput] = useState('');
    const [mapAnswer, setMapAnswer] = useState('');
    const [busy, setBusy] = useState(false);
    const [problem, setProblem] = useState<Problem | null>(null);
    const [result, setResult] = useState<PredefinedSplitManifest | null>(null);

    const submit = async (mappingOverride?: string, onLeakOverride?: OnLeak) => {
        if (!train || duplicateRoles.length > 0) return;
        setBusy(true);
        setProblem(null);
        setResult(null);
        try {
            const manifest = await importPredefinedSplit(projectId, { train, val, test }, {
                onLeak: onLeakOverride ?? onLeak,
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
        // Classification and structured extraction read the input as ``text``; everything else as ``question``.
        const inputKey = adapter?.adapterId === 'classification-label' || adapter?.adapterId === 'structured-extraction'
            ? 'text' : 'question';
        const mapping: Record<string, string> = { [inputKey]: mapInput };
        if (mapAnswer && mapAnswer !== mapInput) mapping.answer = mapAnswer;
        const text = JSON.stringify(mapping);
        setFieldMapping(text);
        void submit(text);
    };

    const showDropOwnTest = !test && (ownTestRows !== null || problem?.code === 'PREDEFINED_TEST_WOULD_BE_DROPPED');

    /** A per-role picker (Advanced): replaces whatever file holds that role. */
    const pickRole = (role: FileRole, file: File | null) => {
        setPicked((prev) => {
            const rest = prev.filter((p) => p.role !== role);
            return file ? [...rest, { file, role }] : rest;
        });
    };

    const setRoleAt = (index: number, role: FileRole | '') => {
        setPicked((prev) => prev.map((p, i) => (i === index ? { ...p, role } : p)));
    };

    const picker = (id: string, label: string, hint: string, role: FileRole) => (
        <label className="pdsplit__file" htmlFor={id}>
            <span>{label}</span>
            <input
                id={id}
                type="file"
                data-testid={id}
                accept=".jsonl,.json,.csv,.tsv,.xlsx,.parquet"
                onChange={(e) => pickRole(role, e.target.files?.[0] ?? null)}
            />
            <small>{hint}</small>
        </label>
    );

    const hasTemplateControl = Boolean(chatTemplates && chatTemplate !== undefined && onChatTemplateChange);
    /** Per-role pickers (3), the leak choice, the chat template (when offered), the column mapping. */
    const optionCount = 3 + 1 + (hasTemplateControl ? 1 : 0) + 1;

    return (
        <div className="pdsplit" data-testid="predefined-split-card">
            <p className="pdsplit__help">
                Already have train / validation / test files (for example a benchmark with a fixed test set)? Pick
                them together: each file's role comes from its name (train, val / valid / dev, test) and you can
                change it. Your files go in as they are: nothing is re-divided or shuffled.
            </p>
            <label className="pdsplit__file pdsplit__file--multi" htmlFor="pdsplit-files">
                <span>Your files (train required; validation and test optional)</span>
                <input
                    id="pdsplit-files"
                    type="file"
                    multiple
                    data-testid="pdsplit-files"
                    accept=".jsonl,.json,.csv,.tsv,.xlsx,.parquet"
                    onChange={(e) => setPicked(assignRoles(Array.from(e.target.files ?? [])))}
                />
                <small>JSONL, JSON, CSV, TSV, XLSX or Parquet</small>
            </label>
            {picked.length > 0 && (
                <table className="pdsplit__table" data-testid="pdsplit-roles">
                    <thead>
                        <tr><th>File</th><th>Used as</th></tr>
                    </thead>
                    <tbody>
                        {picked.map((p, index) => (
                            <tr key={`${p.file.name}-${index}`}>
                                <td>{p.file.name}</td>
                                <td>
                                    <select
                                        aria-label={`Role of ${p.file.name}`}
                                        data-testid={`pdsplit-role-${index}`}
                                        value={p.role}
                                        onChange={(e) => setRoleAt(index, e.target.value as FileRole | '')}
                                    >
                                        <option value="train">Train</option>
                                        <option value="val">Validation</option>
                                        <option value="test">Test</option>
                                        <option value="">Not used</option>
                                    </select>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            {picked.length > 0 && !train && (
                <p className="pdsplit__note" data-testid="pdsplit-need-train">Set one file to Train to import.</p>
            )}
            {duplicateRoles.length > 0 && (
                <p className="pdsplit__note" role="alert" data-testid="pdsplit-duplicate-roles">
                    More than one file is set to {duplicateRoles.map((r) => ROLE_LABEL[r]).join(' and ')}; keep one.
                </p>
            )}
            <AdvancedSection
                pageKey="data"
                id="own-files-options"
                count={optionCount}
                hint="one picker per file, overlapping rows, chat template, column mapping"
                openSection={openSection}
                testId="pdsplit-options"
            >
                <p className="pdsplit__help">
                    Your files go through the same mapping as a normal split. If a train row also appears in
                    validation or test, the import stops and lists it, unless you choose to drop those overlapping rows
                    from train (that removes them from train only; validation and test are never edited). Problems are
                    reported by file line: line 1 of a CSV is the header, so its first row is line 2; blank lines count;
                    for JSON arrays, Excel and Parquet files the number is the record number. Your test file stays out
                    of any later re-split unless you import new files, and a later re-split re-draws validation from
                    train + val.
                </p>
                <div className="pdsplit__files">
                    {picker('pdsplit-train', 'Train file (required)', 'JSONL, JSON, CSV, TSV, XLSX or Parquet', 'train')}
                    {picker('pdsplit-val', 'Validation file (optional)', 'Leave out to train without validation', 'val')}
                    {picker('pdsplit-test', 'Test file (optional)', 'The answer key the model is scored on', 'test')}
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
                {hasTemplateControl && (
                    <label className="pdsplit__field" htmlFor="pdsplit-template">
                        <span>Chat template (same choice as the split form)</span>
                        <select
                            id="pdsplit-template"
                            data-testid="pdsplit-template"
                            value={chatTemplate}
                            onChange={(e) => onChatTemplateChange?.(e.target.value)}
                        >
                            {(chatTemplates ?? []).map((t) => (
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
            </AdvancedSection>
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
                <button className="btn-primary" data-testid="pdsplit-submit" disabled={!train || busy || duplicateRoles.length > 0} onClick={() => void submit()}>
                    {busy ? 'Importing...' : 'Import my files'}
                </button>
            </div>

            {problem && (
                <div className="pdsplit__problem" role="alert" data-testid="pdsplit-problem">
                    <strong><AlertTriangle className="u-icon" aria-hidden="true" /> {problem.message}</strong>
                    {problem.fix && <p>{problem.fix}</p>}
                    {problem.code === 'PREDEFINED_SPLIT_LEAKAGE' && onLeak === 'fail' && (
                        <button
                            className="btn-secondary"
                            type="button"
                            data-testid="pdsplit-retry-drop"
                            disabled={busy}
                            onClick={() => {
                                setOnLeak('drop_from_train');
                                void submit(undefined, 'drop_from_train');
                            }}
                        >
                            Drop the overlapping rows from train and import again
                        </button>
                    )}
                    {problem.overlapColumns.length > 0 && (
                        <p data-testid="pdsplit-problem-overlap-columns">
                            Overlap checked on: {problem.overlapColumns.join(' + ')}
                        </p>
                    )}
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
                        <p data-testid="pdsplit-leak">No train, validation or test rows are exact repeats of each other (near-duplicates are flagged separately on the Train page).</p>
                    )}
                </div>
            )}
        </div>
    );
}
