/**
 * StructuredPlanConfigCard — how a structured-plan project's JSON plans are
 * judged: the JSON-Schema bundle (several files that reference each other and
 * a root), the JSON-pointer fields that must match the answer key, an optional
 * rubric scored by a judge model, and passage retrieval for rows without
 * their own passages.
 *
 * Shown on the Train page for `structured-plan` projects. Saved settings apply
 * at dataset prep, held-out eval, guided decoding and the scenario-grid
 * playbook. Files are read in the browser; the server validates the bundle.
 */

import { useCallback, useEffect, useState } from 'react';

import {
    getStructuredPlanConfig,
    putStructuredPlanConfig,
    shortSha,
    structuredPlanErrors,
    validateStructuredPlanJson,
    type StructuredPlanConfigResponse,
    type StructuredPlanValidateResponse,
} from '../../api/structuredPlan';
import { toast } from '../../stores/toastStore';
import './StructuredPlanConfigCard.css';

interface Props {
    projectId: number;
}

function readText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ''));
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
}

export default function StructuredPlanConfigCard({ projectId }: Props) {
    const [state, setState] = useState<StructuredPlanConfigResponse | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [files, setFiles] = useState<Record<string, unknown>>({});
    const [filesChanged, setFilesChanged] = useState(false);
    const [root, setRoot] = useState('');
    const [rubricYaml, setRubricYaml] = useState<string | null>(null);
    const [rubricName, setRubricName] = useState<string | null>(null);
    const [pointersText, setPointersText] = useState('');
    const [judge, setJudge] = useState(true);
    const [retrieve, setRetrieve] = useState(false);
    // Only a user change is sent: untouched, the server keeps its automatic default (on iff documents are indexed).
    const [retrieveChanged, setRetrieveChanged] = useState(false);
    const [topK, setTopK] = useState(3);
    const [saving, setSaving] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [saveErrors, setSaveErrors] = useState<string[]>([]);
    const [pasted, setPasted] = useState('');
    const [checkResult, setCheckResult] = useState<StructuredPlanValidateResponse | null>(null);
    const [checkError, setCheckError] = useState<string | null>(null);

    const apply = useCallback((res: StructuredPlanConfigResponse) => {
        setState(res);
        setFiles((res.config.schema_files as Record<string, unknown> | undefined) ?? {});
        setFilesChanged(false);
        setRoot(res.config.root_schema ?? res.schema?.root ?? '');
        setRubricYaml(res.config.rubric_yaml ?? null);
        setRubricName(null);
        setPointersText((res.config.pointers ?? res.effective.pointers ?? []).join('\n'));
        setJudge(res.config.judge_enabled ?? res.effective.judge_enabled ?? true);
        setRetrieve(res.config.retrieve_passages ?? res.effective.retrieve_passages ?? false);
        setRetrieveChanged(false);
        setTopK(res.config.top_k ?? res.effective.top_k ?? 3);
    }, []);

    useEffect(() => {
        let cancelled = false;
        getStructuredPlanConfig(projectId)
            .then((res) => { if (!cancelled) apply(res); })
            .catch(() => { if (!cancelled) setLoadError('Could not load the plan settings.'); });
        return () => { cancelled = true; };
    }, [projectId, apply]);

    const onSchemaFiles = async (list: FileList | null) => {
        if (!list || list.length === 0) return;
        setUploadError(null);
        const next: Record<string, unknown> = { ...files };
        for (const file of Array.from(list)) {
            try {
                next[file.name] = JSON.parse(await readText(file));
            } catch {
                setUploadError(`${file.name} is not valid JSON.`);
                return;
            }
        }
        setFiles(next);
        setFilesChanged(true);
        const names = Object.keys(next);
        if (!root || !names.includes(root)) {
            setRoot(names.find((n) => n.startsWith('plan')) ?? names[0] ?? '');
        }
    };

    const removeFile = (name: string) => {
        const next = { ...files };
        delete next[name];
        setFiles(next);
        setFilesChanged(true);
        if (root === name) setRoot(Object.keys(next)[0] ?? '');
    };

    const onRubricFile = async (list: FileList | null) => {
        const file = list?.[0];
        if (!file) return;
        try {
            setRubricYaml(await readText(file));
            setRubricName(file.name);
            setUploadError(null);
        } catch {
            setUploadError(`${file.name} could not be read.`);
        }
    };

    const save = async () => {
        setSaving(true);
        setSaveErrors([]);
        const pointers = pointersText.split('\n').map((l) => l.trim()).filter(Boolean);
        const names = Object.keys(files);
        try {
            const res = await putStructuredPlanConfig(projectId, {
                ...(filesChanged || state?.config.schema_files == null
                    ? { schema_files: names.length > 0 ? files : null, root_schema: names.length > 0 ? root : null }
                    : { root_schema: root }),
                rubric_yaml: rubricYaml && rubricYaml.trim() ? rubricYaml : null,
                pointers,
                judge_enabled: judge,
                ...(retrieveChanged ? { retrieve_passages: retrieve } : {}),
                top_k: topK,
            });
            apply(res);
            toast.success('Plan settings saved.', 3000);
        } catch (err) {
            setSaveErrors(structuredPlanErrors(err));
        } finally {
            setSaving(false);
        }
    };

    const check = async () => {
        setCheckResult(null);
        setCheckError(null);
        let value: unknown;
        try {
            value = JSON.parse(pasted);
        } catch {
            setCheckError('That is not valid JSON.');
            return;
        }
        try {
            setCheckResult(await validateStructuredPlanJson(projectId, value));
        } catch (err) {
            setCheckError(structuredPlanErrors(err)[0]);
        }
    };

    if (loadError) return <p className="u-hint" data-testid="sp-config-error">{loadError}</p>;
    if (!state) return <p className="u-hint">Loading plan settings…</p>;

    const fileNames = Object.keys(files);
    const schema = state.schema;
    const rubric = state.rubric;
    return (
        <div className="sp-config" data-testid="sp-config">
            <p className="u-hint u-mt0">
                These settings apply when plans are prepared, scored and generated. A plan passes when it is valid JSON,
                validates against the schema, every pointer field matches the answer key and, when a rubric runs, no
                critical criterion fails.
            </p>

            <fieldset className="sp-config__group">
                <legend>Schema bundle</legend>
                <div className="sp-config__row">
                    <label htmlFor="sp-schema-files">Schema files (JSON)</label>
                    <input id="sp-schema-files" type="file" accept=".json,application/json" multiple data-testid="sp-schema-upload"
                        onChange={(e) => { void onSchemaFiles(e.target.files); e.target.value = ''; }} />
                </div>
                {uploadError && <p className="sp-config__error" data-testid="sp-upload-error">{uploadError}</p>}
                {fileNames.length > 0 && (
                    <>
                        <ul className="sp-config__files" data-testid="sp-files">
                            {fileNames.map((n) => (
                                <li key={n} data-testid={`sp-file-${n}`}>
                                    <code>{n}</code>
                                    {n === root && <span className="sp-config__ok">root</span>}
                                    <button type="button" className="btn btn-sm btn-secondary" aria-label={`Remove ${n}`} onClick={() => removeFile(n)}>Remove</button>
                                </li>
                            ))}
                        </ul>
                        <div className="sp-config__row">
                            <label htmlFor="sp-root">Root schema</label>
                            <select id="sp-root" value={root} onChange={(e) => setRoot(e.target.value)} data-testid="sp-root-select">
                                {fileNames.map((n) => <option key={n} value={n}>{n}</option>)}
                            </select>
                        </div>
                    </>
                )}
                {fileNames.length === 0 && (
                    <p className="u-hint u-mt0" data-testid="sp-no-schema">
                        No schema saved: plans are only checked for being a JSON object, and the pointer checks still run.
                    </p>
                )}
                {schema && (
                    <p className="u-hint" data-testid="sp-schema-provenance">
                        Saved schema: <code>{schema.root}</code> · {schema.files.length} file{schema.files.length === 1 ? '' : 's'} · sha {shortSha(schema.sha256)}
                        {' '}
                        {schema.valid
                            ? <span className="sp-config__ok">valid</span>
                            : <span className="sp-config__error">not valid</span>}
                    </p>
                )}
                {schema && schema.errors.length > 0 && (
                    <ul className="sp-config__errors" data-testid="sp-schema-errors">
                        {schema.errors.map((e, i) => <li key={i}>{e}</li>)}
                    </ul>
                )}
                {state.guided_json_present && (
                    <p className="u-hint" data-testid="sp-guided">Guided decoding hint is set from this schema.</p>
                )}
            </fieldset>

            <fieldset className="sp-config__group">
                <legend>Rubric (optional)</legend>
                <div className="sp-config__row">
                    <label htmlFor="sp-rubric-file">Rubric file (YAML or JSON)</label>
                    <input id="sp-rubric-file" type="file" accept=".yaml,.yml,.json" data-testid="sp-rubric-upload"
                        onChange={(e) => { void onRubricFile(e.target.files); e.target.value = ''; }} />
                    {rubricYaml && (
                        <button type="button" className="btn btn-sm btn-secondary" onClick={() => { setRubricYaml(null); setRubricName(null); }} data-testid="sp-rubric-clear">
                            Clear rubric
                        </button>
                    )}
                </div>
                {rubricName && <p className="u-hint u-mt0" data-testid="sp-rubric-pending">{rubricName} will be validated when you save.</p>}
                {rubric && !rubricName && rubricYaml && (
                    <>
                        <p className="u-hint" data-testid="sp-rubric-provenance">
                            Rubric: <code>{rubric.id}</code>{rubric.title ? ` · ${rubric.title}` : ''} · sha {shortSha(rubric.sha256)}
                        </p>
                        <table className="sp-config__table" data-testid="sp-rubric-table">
                            <thead><tr><th>Criterion</th><th>Weight</th><th>Critical</th></tr></thead>
                            <tbody>
                                {rubric.criteria.map((c) => (
                                    <tr key={c.id} data-testid={`sp-criterion-${c.id}`}>
                                        <td><code>{c.id}</code></td>
                                        <td>{c.weight}</td>
                                        <td>{c.critical ? 'critical' : ''}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </>
                )}
                <label className="sp-config__row">
                    <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} data-testid="sp-judge-toggle" />
                    Score plans against the rubric with the judge model when one is configured
                </label>
            </fieldset>

            <fieldset className="sp-config__group">
                <legend>Pointer checks</legend>
                <label htmlFor="sp-pointers" className="u-hint u-mt0">One JSON pointer per line, for example /environment or /steps/*/action</label>
                <textarea id="sp-pointers" className="sp-config__textarea" value={pointersText}
                    onChange={(e) => setPointersText(e.target.value)} data-testid="sp-pointers" />
            </fieldset>

            <fieldset className="sp-config__group">
                <legend>Passages</legend>
                <label className="sp-config__row">
                    <input type="checkbox" checked={retrieve} onChange={(e) => { setRetrieve(e.target.checked); setRetrieveChanged(true); }} data-testid="sp-retrieve-toggle" />
                    Retrieve reference passages for rows that have none
                </label>
                <div className="sp-config__row">
                    <label htmlFor="sp-top-k">Passages per row</label>
                    <input id="sp-top-k" type="number" min={1} max={10} value={topK}
                        onChange={(e) => setTopK(Math.max(1, Math.min(10, Number(e.target.value) || 1)))} data-testid="sp-top-k" />
                </div>
            </fieldset>

            {saveErrors.length > 0 && (
                <ul className="sp-config__errors" data-testid="sp-save-errors">
                    {saveErrors.map((e, i) => <li key={i}>{e}</li>)}
                </ul>
            )}
            <div className="sp-config__row">
                <button type="button" className="btn btn-primary" onClick={save} disabled={saving} data-testid="sp-save">
                    {saving ? 'Saving…' : 'Save settings'}
                </button>
            </div>

            <fieldset className="sp-config__group">
                <legend>Check a plan against the saved schema</legend>
                <label htmlFor="sp-paste" className="u-hint u-mt0">Paste a plan as JSON</label>
                <textarea id="sp-paste" className="sp-config__textarea" value={pasted}
                    onChange={(e) => setPasted(e.target.value)} data-testid="sp-paste" />
                <div className="sp-config__row">
                    <button type="button" className="btn btn-secondary" onClick={check} disabled={!pasted.trim()} data-testid="sp-check">Check</button>
                </div>
                {checkError && <p className="sp-config__error" data-testid="sp-check-error">{checkError}</p>}
                {checkResult && checkResult.valid && <p className="sp-config__ok" data-testid="sp-check-valid">Valid against the schema.</p>}
                {checkResult && !checkResult.valid && (
                    <ul className="sp-config__errors" data-testid="sp-check-errors">
                        {checkResult.errors.map((e, i) => <li key={i}><code>{e.pointer || '/'}</code> {e.message}</li>)}
                    </ul>
                )}
            </fieldset>
        </div>
    );
}
