/**
 * Generic dataset-import wizard (Phase F of DATASET_IMPORT_PLAN.md).
 *
 * 3-step modal that lets a user go from "here's a locator" to "rows
 * landed in my project's synthetic dataset" without touching the CLI:
 *
 *   1. Source     — pick source type, enter locator, server-side
 *                   credential reminder for hf / kaggle.
 *   2. Map        — calls /introspect, shows column signatures and
 *                   ranked hypotheses, lets the user pick a mapper and
 *                   edit the proposed field map. Enforces the same
 *                   confidence threshold the CLI does.
 *   3. Preview &  — calls /preview, shows accepted samples + rejected
 *      Confirm     rows grouped by reason for bulk-drop, then /run
 *                   commits.
 *
 * The same architectural rules from the CLI apply: never silently
 * auto-pick a mapping; group rejected rows by reason and let the user
 * bulk-drop categories before persisting (per the [[rejected-rows-
 * selectable]] feedback memory).
 */

import { useEffect, useMemo, useState } from 'react';
import {
    introspectLocator,
    listMappers,
    listSources,
    previewImport,
    runImport,
    saveConfig,
    type ImportResultDict,
    type IntrospectResponse,
    type SavedConfig,
    type ShapeHypothesisDict,
} from '../../api/datasetImport';

interface DatasetImportWizardProps {
    projectId: number;
    onClose: () => void;
    onSuccess?: (result: ImportResultDict) => void;
    onConfigSaved?: (config: SavedConfig) => void;
}

type WizardStep = 'source' | 'map' | 'preview';

interface SourceHelp {
    label: string;
    placeholder: string;
    helpText: string;
    authNote?: string;
}

const SOURCE_HELP: Record<string, SourceHelp> = {
    jsonl: {
        label: 'JSONL file',
        placeholder: '/absolute/path/to/data.jsonl',
        helpText: 'One JSON object per line.',
    },
    csv: {
        label: 'CSV file',
        placeholder: '/absolute/path/to/data.csv',
        helpText: 'First row is the header; every cell becomes a string.',
    },
    hf: {
        label: 'HuggingFace Hub',
        placeholder: 'Anthropic/hh-rlhf:train',
        helpText:
            'Format: dataset_id[:split[:revision]] — e.g. ai4privacy/pii-masking-200k:train',
        authNote:
            'Gated datasets require HF_TOKEN or HUGGING_FACE_HUB_TOKEN on the server before importing.',
    },
    kaggle: {
        label: 'Kaggle',
        placeholder: 'competition:pii-detection-…  or  dataset:owner/slug',
        helpText:
            'Use competition:<slug> or dataset:<owner/slug>. Append ?file=<path> to disambiguate multi-file archives.',
        authNote:
            'Requires KAGGLE_USERNAME + KAGGLE_KEY on the server (or ~/.kaggle/kaggle.json). Set under Project → Secrets.',
    },
};

function buildLocator(sourceId: string, locatorBody: string): string {
    const trimmed = locatorBody.trim();
    if (!trimmed) {
        return '';
    }
    return `${sourceId}:${trimmed}`;
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const data = (err as { response?: { data?: { detail?: unknown } } }).response?.data
            ?.detail;
        if (typeof data === 'string' && data.trim()) {
            return data;
        }
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) {
            return message;
        }
    }
    return 'Unknown error';
}

function fieldMapToJsonString(value: Record<string, unknown>): string {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return '{}';
    }
}

function parseJsonFieldMap(raw: string): {
    value: Record<string, unknown>;
    error: string;
} {
    const text = raw.trim();
    if (!text) {
        return { value: {}, error: '' };
    }
    try {
        const parsed = JSON.parse(text);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return { value: {}, error: 'field_map must be a JSON object' };
        }
        return { value: parsed as Record<string, unknown>, error: '' };
    } catch (err) {
        if (err instanceof Error) {
            return { value: {}, error: err.message };
        }
        return { value: {}, error: 'invalid JSON' };
    }
}

export default function DatasetImportWizard({
    projectId,
    onClose,
    onSuccess,
    onConfigSaved,
}: DatasetImportWizardProps) {
    // Step state.
    const [step, setStep] = useState<WizardStep>('source');

    // Catalog state.
    const [sources, setSources] = useState<string[]>(['jsonl', 'csv', 'hf', 'kaggle']);
    const [mappers, setMappers] = useState<string[]>([]);

    // Source step.
    const [sourceId, setSourceId] = useState<string>('jsonl');
    const [locatorBody, setLocatorBody] = useState<string>('');

    // Map step.
    const [introspecting, setIntrospecting] = useState(false);
    const [introspectError, setIntrospectError] = useState<string>('');
    const [introspection, setIntrospection] = useState<IntrospectResponse | null>(null);
    const [selectedMapperId, setSelectedMapperId] = useState<string>('');
    const [fieldMapJson, setFieldMapJson] = useState<string>('{}');
    const [forceLowConfidence, setForceLowConfidence] = useState(false);

    // Preview step.
    const [previewing, setPreviewing] = useState(false);
    const [previewResult, setPreviewResult] = useState<ImportResultDict | null>(null);
    const [previewError, setPreviewError] = useState<string>('');
    const [dropReasons, setDropReasons] = useState<Set<string>>(new Set());
    const [running, setRunning] = useState(false);
    const [runError, setRunError] = useState<string>('');
    const [runResult, setRunResult] = useState<ImportResultDict | null>(null);

    // Save-mapping state (Phase G).
    const [savingConfig, setSavingConfig] = useState(false);
    const [saveError, setSaveError] = useState<string>('');
    const [savedConfigName, setSavedConfigName] = useState<string | null>(null);

    // Load source / mapper catalog once (best-effort — fall back to
    // built-in lists if the request fails).
    useEffect(() => {
        let cancelled = false;
        Promise.allSettled([listSources(), listMappers()]).then(([srcRes, mapRes]) => {
            if (cancelled) return;
            if (srcRes.status === 'fulfilled' && srcRes.value.length > 0) {
                setSources(srcRes.value);
                if (!srcRes.value.includes(sourceId)) {
                    setSourceId(srcRes.value[0]);
                }
            }
            if (mapRes.status === 'fulfilled') {
                setMappers(mapRes.value);
            }
        });
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const locator = useMemo(() => buildLocator(sourceId, locatorBody), [sourceId, locatorBody]);
    const sourceHelp = SOURCE_HELP[sourceId] ?? {
        label: sourceId,
        placeholder: '',
        helpText: '',
    };

    const handleIntrospect = async () => {
        if (!locator) {
            setIntrospectError('Enter a locator first.');
            return;
        }
        setIntrospecting(true);
        setIntrospectError('');
        try {
            const res = await introspectLocator(locator);
            setIntrospection(res);
            // Default-select the top hypothesis if any; users can pick a
            // different one in the UI.
            const topMapper = res.proposal?.mapper_id || res.hypotheses[0]?.mapper_id || '';
            setSelectedMapperId(topMapper);
            const topFieldMap = res.proposal?.field_map || res.hypotheses[0]?.field_map || {};
            setFieldMapJson(fieldMapToJsonString(topFieldMap));
            setForceLowConfidence(false);
            setStep('map');
        } catch (err) {
            setIntrospectError(extractErrorMessage(err));
        } finally {
            setIntrospecting(false);
        }
    };

    const selectedHypothesis: ShapeHypothesisDict | undefined = useMemo(() => {
        if (!introspection) return undefined;
        return introspection.hypotheses.find((h) => h.mapper_id === selectedMapperId);
    }, [introspection, selectedMapperId]);

    const selectionConfidence = selectedHypothesis?.confidence ?? introspection?.proposal?.confidence ?? 0;
    const confidenceThreshold = introspection?.confidence_threshold ?? 0.8;
    const isBelowThreshold = selectionConfidence > 0 && selectionConfidence < confidenceThreshold;
    const fieldMapParsed = useMemo(() => parseJsonFieldMap(fieldMapJson), [fieldMapJson]);
    const canRunPreview =
        Boolean(selectedMapperId) &&
        !fieldMapParsed.error &&
        (!isBelowThreshold || forceLowConfidence);

    const handlePreview = async () => {
        if (!canRunPreview) return;
        setPreviewing(true);
        setPreviewError('');
        setPreviewResult(null);
        try {
            const res = await previewImport(projectId, {
                locator,
                mapper_id: selectedMapperId,
                field_map: fieldMapParsed.value,
                sample_cap: 5,
                drop_reasons: Array.from(dropReasons),
            });
            setPreviewResult(res);
            setStep('preview');
        } catch (err) {
            setPreviewError(extractErrorMessage(err));
        } finally {
            setPreviewing(false);
        }
    };

    const toggleDropReason = (reason: string) => {
        setDropReasons((prev) => {
            const next = new Set(prev);
            if (next.has(reason)) {
                next.delete(reason);
            } else {
                next.add(reason);
            }
            return next;
        });
    };

    const handleRefreshPreview = async () => {
        // Re-run the preview with the current drop_reasons selection so
        // the sample updates after the user picks bulk-drop categories.
        setPreviewing(true);
        setPreviewError('');
        try {
            const res = await previewImport(projectId, {
                locator,
                mapper_id: selectedMapperId,
                field_map: fieldMapParsed.value,
                sample_cap: 5,
                drop_reasons: Array.from(dropReasons),
            });
            setPreviewResult(res);
        } catch (err) {
            setPreviewError(extractErrorMessage(err));
        } finally {
            setPreviewing(false);
        }
    };

    const handleRun = async () => {
        setRunning(true);
        setRunError('');
        try {
            const res = await runImport(projectId, {
                locator,
                mapper_id: selectedMapperId,
                field_map: fieldMapParsed.value,
                drop_reasons: Array.from(dropReasons),
            });
            setRunResult(res);
            onSuccess?.(res);
        } catch (err) {
            setRunError(extractErrorMessage(err));
        } finally {
            setRunning(false);
        }
    };

    const handleSaveConfig = async (name: string, description: string) => {
        setSavingConfig(true);
        setSaveError('');
        try {
            const cfg = await saveConfig(projectId, {
                name: name.trim(),
                description: description.trim() || null,
                locator,
                mapper_id: selectedMapperId,
                field_map: fieldMapParsed.value,
                drop_reasons: Array.from(dropReasons),
            });
            setSavedConfigName(cfg.name);
            onConfigSaved?.(cfg);
        } catch (err) {
            setSaveError(extractErrorMessage(err));
        } finally {
            setSavingConfig(false);
        }
    };

    const stepLabel: Record<WizardStep, string> = {
        source: '1. Source',
        map: '2. Map',
        preview: '3. Preview & Confirm',
    };

    return (
        <div className="modal-backdrop" data-testid="dataset-import-wizard">
            <div className="modal" style={{ maxWidth: 920, width: '95vw' }}>
                <div className="modal-header">
                    <h2 style={{ margin: 0 }}>Import dataset</h2>
                    <button className="btn btn-ghost" onClick={onClose} aria-label="Close">
                        ✕
                    </button>
                </div>

                {/* Step indicator */}
                <div
                    style={{
                        display: 'flex',
                        gap: 'var(--space-md)',
                        padding: 'var(--space-md) var(--space-lg)',
                        borderBottom: '1px solid var(--border-color)',
                        fontSize: '0.9rem',
                    }}
                >
                    {(Object.keys(stepLabel) as WizardStep[]).map((s) => (
                        <div
                            key={s}
                            style={{
                                color: s === step ? 'var(--primary-color)' : 'var(--text-secondary)',
                                fontWeight: s === step ? 600 : 400,
                            }}
                        >
                            {stepLabel[s]}
                        </div>
                    ))}
                </div>

                <div className="modal-body" style={{ maxHeight: '70vh', overflowY: 'auto' }}>
                    {step === 'source' && (
                        <SourceStep
                            sources={sources}
                            sourceId={sourceId}
                            onSourceChange={setSourceId}
                            locatorBody={locatorBody}
                            onLocatorBodyChange={setLocatorBody}
                            sourceHelp={sourceHelp}
                            introspecting={introspecting}
                            introspectError={introspectError}
                            onNext={handleIntrospect}
                        />
                    )}

                    {step === 'map' && introspection && (
                        <MapStep
                            introspection={introspection}
                            mappers={mappers}
                            selectedMapperId={selectedMapperId}
                            onSelectedMapperChange={(m) => {
                                setSelectedMapperId(m);
                                // Snap field_map to the new hypothesis when
                                // switching ranked options.
                                const hyp = introspection.hypotheses.find((h) => h.mapper_id === m);
                                if (hyp) {
                                    setFieldMapJson(fieldMapToJsonString(hyp.field_map));
                                }
                            }}
                            fieldMapJson={fieldMapJson}
                            onFieldMapChange={setFieldMapJson}
                            fieldMapError={fieldMapParsed.error}
                            isBelowThreshold={isBelowThreshold}
                            forceLowConfidence={forceLowConfidence}
                            onForceChange={setForceLowConfidence}
                            confidenceThreshold={confidenceThreshold}
                            previewing={previewing}
                            previewError={previewError}
                            canRunPreview={canRunPreview}
                            onBack={() => setStep('source')}
                            onPreview={handlePreview}
                        />
                    )}

                    {step === 'preview' && previewResult && (
                        <PreviewStep
                            previewResult={previewResult}
                            previewing={previewing}
                            previewError={previewError}
                            dropReasons={dropReasons}
                            onToggleDropReason={toggleDropReason}
                            onRefreshPreview={handleRefreshPreview}
                            running={running}
                            runError={runError}
                            runResult={runResult}
                            onBack={() => setStep('map')}
                            onRun={handleRun}
                            onClose={onClose}
                            onSaveConfig={handleSaveConfig}
                            savingConfig={savingConfig}
                            saveError={saveError}
                            savedConfigName={savedConfigName}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}

// ── Step 1: Source ───────────────────────────────────────────────────

interface SourceStepProps {
    sources: string[];
    sourceId: string;
    onSourceChange: (id: string) => void;
    locatorBody: string;
    onLocatorBodyChange: (value: string) => void;
    sourceHelp: SourceHelp;
    introspecting: boolean;
    introspectError: string;
    onNext: () => void;
}

function SourceStep({
    sources,
    sourceId,
    onSourceChange,
    locatorBody,
    onLocatorBodyChange,
    sourceHelp,
    introspecting,
    introspectError,
    onNext,
}: SourceStepProps) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                Pick the source type and enter a locator. The introspector will sniff the
                columns and propose a mapping — no need to write a converter.
            </p>

            <div className="form-group">
                <label className="form-label" htmlFor="dsi-source">
                    Source
                </label>
                <select
                    id="dsi-source"
                    className="form-select"
                    value={sourceId}
                    onChange={(e) => onSourceChange(e.target.value)}
                >
                    {sources.map((s) => (
                        <option key={s} value={s}>
                            {SOURCE_HELP[s]?.label ?? s} ({s})
                        </option>
                    ))}
                </select>
            </div>

            <div className="form-group">
                <label className="form-label">Locator</label>
                <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center' }}>
                    <code
                        style={{
                            padding: '6px 10px',
                            background: 'var(--bg-secondary)',
                            borderRadius: 'var(--radius-sm)',
                            border: '1px solid var(--border-color)',
                            fontSize: '0.9rem',
                            whiteSpace: 'nowrap',
                        }}
                    >
                        {sourceId}:
                    </code>
                    <input
                        className="input"
                        style={{ flex: 1 }}
                        value={locatorBody}
                        onChange={(e) => onLocatorBodyChange(e.target.value)}
                        placeholder={sourceHelp.placeholder}
                        data-testid="locator-input"
                    />
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                    {sourceHelp.helpText}
                </div>
            </div>

            {sourceHelp.authNote && (
                <div
                    style={{
                        padding: 'var(--space-md)',
                        background: 'rgba(245, 158, 11, 0.08)',
                        border: '1px solid rgba(245, 158, 11, 0.3)',
                        borderRadius: 'var(--radius-md)',
                        fontSize: '0.9rem',
                    }}
                    role="note"
                >
                    {sourceHelp.authNote}
                </div>
            )}

            {introspectError && (
                <div className="error-banner" data-testid="introspect-error">
                    {introspectError}
                </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                    className="btn btn-primary"
                    onClick={onNext}
                    disabled={introspecting || !locatorBody.trim()}
                    data-testid="introspect-btn"
                >
                    {introspecting ? 'Sniffing…' : 'Introspect →'}
                </button>
            </div>
        </div>
    );
}

// ── Step 2: Map ──────────────────────────────────────────────────────

interface MapStepProps {
    introspection: IntrospectResponse;
    mappers: string[];
    selectedMapperId: string;
    onSelectedMapperChange: (id: string) => void;
    fieldMapJson: string;
    onFieldMapChange: (value: string) => void;
    fieldMapError: string;
    isBelowThreshold: boolean;
    forceLowConfidence: boolean;
    onForceChange: (value: boolean) => void;
    confidenceThreshold: number;
    previewing: boolean;
    previewError: string;
    canRunPreview: boolean;
    onBack: () => void;
    onPreview: () => void;
}

function MapStep({
    introspection,
    mappers,
    selectedMapperId,
    onSelectedMapperChange,
    fieldMapJson,
    onFieldMapChange,
    fieldMapError,
    isBelowThreshold,
    forceLowConfidence,
    onForceChange,
    confidenceThreshold,
    previewing,
    previewError,
    canRunPreview,
    onBack,
    onPreview,
}: MapStepProps) {
    const hypothesesById = useMemo(() => {
        const map = new Map<string, ShapeHypothesisDict>();
        for (const hyp of introspection.hypotheses) {
            map.set(hyp.mapper_id, hyp);
        }
        return map;
    }, [introspection]);

    // Mapper choices = ranked hypotheses + any registered mappers
    // the introspector didn't propose (so power users can hand-pick).
    const allMapperOptions = useMemo(() => {
        const seen = new Set<string>();
        const ordered: string[] = [];
        for (const hyp of introspection.hypotheses) {
            if (!seen.has(hyp.mapper_id)) {
                seen.add(hyp.mapper_id);
                ordered.push(hyp.mapper_id);
            }
        }
        for (const m of mappers) {
            if (!seen.has(m)) {
                seen.add(m);
                ordered.push(m);
            }
        }
        if (selectedMapperId && !seen.has(selectedMapperId)) {
            ordered.push(selectedMapperId);
        }
        return ordered;
    }, [introspection, mappers, selectedMapperId]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            {/* Column rundown */}
            <section>
                <h3 style={{ marginTop: 0, marginBottom: 'var(--space-sm)' }}>Column signatures</h3>
                <table className="table" style={{ width: '100%', fontSize: '0.9rem' }}>
                    <thead>
                        <tr>
                            <th style={{ textAlign: 'left' }}>Column</th>
                            <th style={{ textAlign: 'left' }}>Detected type</th>
                            <th style={{ textAlign: 'left' }}>Confidence</th>
                            <th style={{ textAlign: 'left' }}>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {introspection.column_signatures.map((sig) => (
                            <tr key={sig.name}>
                                <td><code>{sig.name}</code></td>
                                <td>{sig.column_type}</td>
                                <td>{(sig.confidence * 100).toFixed(0)}%</td>
                                <td style={{ color: 'var(--text-secondary)' }}>
                                    {sig.unique_values.length > 0 && (
                                        <span>
                                            unique: [{sig.unique_values.slice(0, 5).join(', ')}
                                            {sig.unique_values.length > 5 ? ', …' : ''}]
                                        </span>
                                    )}
                                    {sig.notes && <span>{sig.notes}</span>}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </section>

            {/* Mapper picker */}
            <section>
                <h3 style={{ marginTop: 0, marginBottom: 'var(--space-sm)' }}>Mapper</h3>
                {introspection.hypotheses.length === 0 && (
                    <div
                        style={{
                            padding: 'var(--space-md)',
                            background: 'rgba(245, 158, 11, 0.08)',
                            border: '1px solid rgba(245, 158, 11, 0.3)',
                            borderRadius: 'var(--radius-md)',
                            fontSize: '0.9rem',
                            marginBottom: 'var(--space-sm)',
                        }}
                        role="note"
                    >
                        No mapping hypothesis matched this dataset's shape. Pick a mapper from
                        the dropdown below and edit the field_map manually.
                    </div>
                )}
                <div className="form-group">
                    <label className="form-label">Use mapper</label>
                    <select
                        className="form-select"
                        value={selectedMapperId}
                        onChange={(e) => onSelectedMapperChange(e.target.value)}
                        data-testid="mapper-select"
                    >
                        <option value="">— pick a mapper —</option>
                        {allMapperOptions.map((m) => {
                            const hyp = hypothesesById.get(m);
                            const conf = hyp ? ` — ${Math.round(hyp.confidence * 100)}% confidence` : '';
                            return (
                                <option key={m} value={m}>
                                    {m}
                                    {conf}
                                </option>
                            );
                        })}
                    </select>
                </div>

                {selectedMapperId && hypothesesById.has(selectedMapperId) && (
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-sm)' }}>
                        <strong>Rationale:</strong> {hypothesesById.get(selectedMapperId)?.rationale}
                    </div>
                )}
            </section>

            {/* Confidence gate */}
            {isBelowThreshold && (
                <div
                    style={{
                        padding: 'var(--space-md)',
                        background: 'rgba(239, 68, 68, 0.08)',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: 'var(--radius-md)',
                        fontSize: '0.9rem',
                    }}
                    role="alert"
                    data-testid="confidence-warning"
                >
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>Low-confidence proposal</div>
                    <div style={{ marginBottom: 'var(--space-sm)' }}>
                        Confidence is below the {(confidenceThreshold * 100).toFixed(0)}% threshold.
                        Eyeball the column signatures + rationale carefully before proceeding.
                    </div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                        <input
                            type="checkbox"
                            checked={forceLowConfidence}
                            onChange={(e) => onForceChange(e.target.checked)}
                            data-testid="force-checkbox"
                        />
                        <span>I've reviewed the proposal — proceed anyway.</span>
                    </label>
                </div>
            )}

            {/* Field-map editor */}
            <section>
                <h3 style={{ marginTop: 0, marginBottom: 'var(--space-sm)' }}>Field map</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 0 }}>
                    JSON object passed through to the mapper. Edit to override the
                    introspector's suggestion (rename a source column, supply a nested
                    config like <code>entity_type_map</code>, etc.).
                </p>
                <textarea
                    className="input"
                    rows={8}
                    style={{ width: '100%', fontFamily: 'var(--font-monospace, monospace)' }}
                    value={fieldMapJson}
                    onChange={(e) => onFieldMapChange(e.target.value)}
                    data-testid="field-map-input"
                />
                {fieldMapError && (
                    <div className="error-banner" style={{ marginTop: 'var(--space-sm)' }}>
                        {fieldMapError}
                    </div>
                )}
            </section>

            {previewError && (
                <div className="error-banner" data-testid="preview-error">
                    {previewError}
                </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <button className="btn btn-ghost" onClick={onBack}>
                    ← Back
                </button>
                <button
                    className="btn btn-primary"
                    onClick={onPreview}
                    disabled={!canRunPreview || previewing}
                    data-testid="preview-btn"
                >
                    {previewing ? 'Previewing…' : 'Preview →'}
                </button>
            </div>
        </div>
    );
}

// ── Step 3: Preview & Confirm ────────────────────────────────────────

interface PreviewStepProps {
    previewResult: ImportResultDict;
    previewing: boolean;
    previewError: string;
    dropReasons: Set<string>;
    onToggleDropReason: (reason: string) => void;
    onRefreshPreview: () => void;
    running: boolean;
    runError: string;
    runResult: ImportResultDict | null;
    onBack: () => void;
    onRun: () => void;
    onClose: () => void;
    onSaveConfig: (name: string, description: string) => Promise<void>;
    savingConfig: boolean;
    saveError: string;
    savedConfigName: string | null;
}

function PreviewStep({
    previewResult,
    previewing,
    previewError,
    dropReasons,
    onToggleDropReason,
    onRefreshPreview,
    running,
    runError,
    runResult,
    onBack,
    onRun,
    onClose,
    onSaveConfig,
    savingConfig,
    saveError,
    savedConfigName,
}: PreviewStepProps) {
    const [showSaveForm, setShowSaveForm] = useState(false);
    const [saveName, setSaveName] = useState('');
    const [saveDescription, setSaveDescription] = useState('');
    const reasonEntries = useMemo(
        () =>
            Object.entries(previewResult.rejection_counts).sort(
                (a, b) => b[1] - a[1],
            ),
        [previewResult.rejection_counts],
    );

    if (runResult) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                <div
                    style={{
                        padding: 'var(--space-md)',
                        background: 'rgba(34, 197, 94, 0.08)',
                        border: '1px solid rgba(34, 197, 94, 0.3)',
                        borderRadius: 'var(--radius-md)',
                    }}
                    role="status"
                    data-testid="run-success-banner"
                >
                    <h3 style={{ marginTop: 0 }}>Imported.</h3>
                    <div>
                        <strong>{runResult.accepted_count}</strong> row(s) written to the
                        project's synthetic dataset.{' '}
                        {runResult.rejected_count > 0 && (
                            <>
                                <strong>{runResult.rejected_count}</strong> row(s) rejected (counts
                                preserved in the result for audit).
                            </>
                        )}
                    </div>
                    {runResult.written_path && (
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                            File: <code>{runResult.written_path}</code>
                        </div>
                    )}
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button className="btn btn-primary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            {/* Summary */}
            <section>
                <h3 style={{ marginTop: 0, marginBottom: 'var(--space-sm)' }}>Dry-run summary</h3>
                <div style={{ display: 'flex', gap: 'var(--space-lg)', flexWrap: 'wrap' }}>
                    <SummaryCard
                        label="Accepted"
                        value={previewResult.accepted_count}
                        tone="positive"
                    />
                    <SummaryCard
                        label="Rejected"
                        value={previewResult.rejected_count}
                        tone={previewResult.rejected_count > 0 ? 'warning' : 'neutral'}
                    />
                    <SummaryCard
                        label="Mapper"
                        value={previewResult.mapper_id}
                        tone="neutral"
                    />
                    <SummaryCard
                        label="Target profile"
                        value={previewResult.target_task_profile}
                        tone="neutral"
                    />
                </div>
            </section>

            {/* Rejection breakdown + bulk drop */}
            {reasonEntries.length > 0 && (
                <section>
                    <h3 style={{ marginTop: 0, marginBottom: 'var(--space-sm)' }}>
                        Rejected rows by reason
                    </h3>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 0 }}>
                        Tick a reason to bulk-drop those rejects on import. Counts are
                        preserved in the final result for audit. Refresh the preview after
                        toggling to see the updated sample.
                    </p>
                    <table className="table" style={{ width: '100%', fontSize: '0.9rem' }}>
                        <thead>
                            <tr>
                                <th style={{ width: 40 }}>Drop</th>
                                <th style={{ textAlign: 'left' }}>Reason</th>
                                <th style={{ textAlign: 'right' }}>Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {reasonEntries.map(([reason, count]) => (
                                <tr key={reason} data-testid={`reject-row-${reason}`}>
                                    <td>
                                        <input
                                            type="checkbox"
                                            checked={dropReasons.has(reason)}
                                            onChange={() => onToggleDropReason(reason)}
                                            data-testid={`drop-${reason}`}
                                        />
                                    </td>
                                    <td><code>{reason}</code></td>
                                    <td style={{ textAlign: 'right' }}>{count}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <div style={{ marginTop: 'var(--space-sm)' }}>
                        <button
                            className="btn btn-secondary"
                            onClick={onRefreshPreview}
                            disabled={previewing}
                            data-testid="refresh-preview-btn"
                        >
                            {previewing ? 'Refreshing…' : 'Refresh preview'}
                        </button>
                    </div>
                </section>
            )}

            {/* Accepted sample */}
            {previewResult.accepted_sample.length > 0 && (
                <section>
                    <h3 style={{ marginTop: 0, marginBottom: 'var(--space-sm)' }}>
                        Sample transformed rows
                    </h3>
                    <pre
                        style={{
                            background: 'var(--bg-secondary)',
                            padding: 'var(--space-md)',
                            borderRadius: 'var(--radius-md)',
                            fontSize: '0.85rem',
                            maxHeight: 240,
                            overflow: 'auto',
                            border: '1px solid var(--border-color)',
                        }}
                    >
                        {JSON.stringify(
                            previewResult.accepted_sample.map((r) => r.payload),
                            null,
                            2,
                        )}
                    </pre>
                </section>
            )}

            {previewError && <div className="error-banner">{previewError}</div>}
            {runError && (
                <div className="error-banner" data-testid="run-error">
                    {runError}
                </div>
            )}

            {/* Save-as-config affordance: lets the user persist this
                mapping for one-click re-runs later (Phase G). */}
            <section
                style={{
                    padding: 'var(--space-md)',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-md)',
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
                    <div>
                        <strong>Save this mapping</strong>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            Stores the locator + mapper + field_map + drop_reasons so you can
                            re-run against a refreshed source without re-introspecting.
                        </div>
                    </div>
                    {savedConfigName ? (
                        <span
                            style={{
                                fontSize: '0.85rem',
                                color: 'var(--text-secondary)',
                            }}
                            data-testid="saved-config-confirm"
                        >
                            Saved as <code>{savedConfigName}</code>.
                        </span>
                    ) : (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => setShowSaveForm((v) => !v)}
                            data-testid="toggle-save-form-btn"
                            disabled={running}
                        >
                            {showSaveForm ? 'Cancel' : 'Save mapping'}
                        </button>
                    )}
                </div>
                {showSaveForm && !savedConfigName && (
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 'var(--space-sm)',
                            marginTop: 'var(--space-md)',
                        }}
                    >
                        <div className="form-group">
                            <label className="form-label" htmlFor="dsi-save-name">
                                Name
                            </label>
                            <input
                                id="dsi-save-name"
                                className="input"
                                value={saveName}
                                onChange={(e) => setSaveName(e.target.value)}
                                placeholder="e.g. weekly PII refresh"
                                data-testid="save-name-input"
                                maxLength={120}
                            />
                        </div>
                        <div className="form-group">
                            <label className="form-label" htmlFor="dsi-save-desc">
                                Description (optional)
                            </label>
                            <input
                                id="dsi-save-desc"
                                className="input"
                                value={saveDescription}
                                onChange={(e) => setSaveDescription(e.target.value)}
                                placeholder="Why this mapping exists / when to re-run"
                                data-testid="save-desc-input"
                                maxLength={1000}
                            />
                        </div>
                        {saveError && (
                            <div className="error-banner" data-testid="save-error">
                                {saveError}
                            </div>
                        )}
                        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                            <button
                                type="button"
                                className="btn btn-primary"
                                onClick={() => onSaveConfig(saveName, saveDescription)}
                                disabled={savingConfig || !saveName.trim()}
                                data-testid="save-config-btn"
                            >
                                {savingConfig ? 'Saving…' : 'Save'}
                            </button>
                        </div>
                    </div>
                )}
            </section>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <button className="btn btn-ghost" onClick={onBack} disabled={running}>
                    ← Back
                </button>
                <button
                    className="btn btn-primary"
                    onClick={onRun}
                    disabled={running || previewResult.accepted_count === 0}
                    data-testid="run-btn"
                >
                    {running
                        ? 'Importing…'
                        : `Import ${previewResult.accepted_count} row(s) to project`}
                </button>
            </div>
        </div>
    );
}

interface SummaryCardProps {
    label: string;
    value: string | number;
    tone: 'positive' | 'warning' | 'neutral';
}

function SummaryCard({ label, value, tone }: SummaryCardProps) {
    const toneColor: Record<SummaryCardProps['tone'], string> = {
        positive: 'rgba(34, 197, 94, 0.08)',
        warning: 'rgba(245, 158, 11, 0.08)',
        neutral: 'var(--bg-secondary)',
    };
    return (
        <div
            style={{
                padding: 'var(--space-md)',
                background: toneColor[tone],
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                minWidth: 160,
            }}
        >
            <div
                style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-secondary)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                }}
            >
                {label}
            </div>
            <div style={{ fontWeight: 600, fontSize: '1.1rem', marginTop: 4 }}>{value}</div>
        </div>
    );
}
