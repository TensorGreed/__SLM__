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
import { X } from 'lucide-react';
import {
    introspectLocator,
    listMappers,
    listSources,
    previewImport,
    runImport,
    saveConfig,
    uploadImportFile,
    type ImportResultDict,
    type IntrospectResponse,
    type SavedConfig,
    type ShapeHypothesisDict,
} from '../../api/datasetImport';
import { applyRecipeToProject } from '../../api/recipes';
import type { Recipe, RecipeSuggestion } from '../../api/recipes';
import RecipePicker from './RecipePicker';
import TaskShapeConfirmCard from './TaskShapeConfirmCard';
import './DatasetImportWizard.css';

interface DatasetImportWizardProps {
    projectId: number;
    onClose: () => void;
    onSuccess?: (result: ImportResultDict) => void;
    onConfigSaved?: (config: SavedConfig) => void;
}

type WizardStep = 'source' | 'recipe' | 'map' | 'preview';

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
    file: {
        label: 'Server file (any row format)',
        placeholder: '/absolute/path/to/data.xlsx',
        helpText: 'CSV, TSV, Excel (.xlsx), JSON, JSONL or Parquet already on the server. Any encoding.',
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
    web: {
        label: 'Website page',
        placeholder: 'https://docs.example.com/guide/',
        helpText: 'One row per page: url, title, text. For a whole site as documents use "Documents from website / repo".',
    },
    git: {
        label: 'Git repository',
        placeholder: 'https://github.com/org/repo@v1.2',
        helpText: 'One row per code chunk (path, language, line range, symbol, text); doc files are one row each. Append @ref for a branch, tag or commit.',
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
    // Browser upload: the server stages the file and returns a ``file:``
    // locator, which then takes the place of the typed one.
    const [uploadedLocator, setUploadedLocator] = useState<string>('');
    const [uploadedName, setUploadedName] = useState<string>('');
    const [uploading, setUploading] = useState(false);

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

    // Recipe picker state (Theme 2). Recorded for downstream defaults +
    // shown as a summary chip in later steps. Empty when the user
    // clicks "Override".
    const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
    const [recipeSuggestion, setRecipeSuggestion] = useState<RecipeSuggestion | null>(null);
    const [recipeOverridden, setRecipeOverridden] = useState(false);
    const [recipePersistError, setRecipePersistError] = useState<string>('');

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

    const locator = useMemo(
        () => uploadedLocator || buildLocator(sourceId, locatorBody),
        [uploadedLocator, sourceId, locatorBody],
    );
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
            applyIntrospection(await introspectLocator(locator));
        } catch (err) {
            setIntrospectError(extractErrorMessage(err));
        } finally {
            setIntrospecting(false);
        }
    };

    const handleUploadFile = async (file: File) => {
        setUploading(true);
        setIntrospectError('');
        try {
            const res = await uploadImportFile(projectId, file);
            setUploadedLocator(res.locator);
            setUploadedName(res.filename);
            applyIntrospection(res.introspection);
        } catch (err) {
            setIntrospectError(extractErrorMessage(err));
        } finally {
            setUploading(false);
        }
    };

    const applyIntrospection = (res: IntrospectResponse) => {
        setIntrospection(res);
        // Default-select the top hypothesis if any; users can pick a
        // different one in the UI.
        const topMapper = res.proposal?.mapper_id || res.hypotheses[0]?.mapper_id || '';
        setSelectedMapperId(topMapper);
        const topFieldMap = res.proposal?.field_map || res.hypotheses[0]?.field_map || {};
        setFieldMapJson(fieldMapToJsonString(topFieldMap));
        setForceLowConfidence(false);
        // Reset recipe state on every fresh introspect; the picker
        // is shown next so the user lands on a sniffed task shape
        // before the dense mapper config.
        setSelectedRecipe(null);
        setRecipeSuggestion(null);
        setRecipeOverridden(false);
        setStep('recipe');
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
        recipe: '2. Task type',
        map: '3. Map',
        preview: '4. Preview & Confirm',
    };

    return (
        <div className="modal-backdrop" data-testid="dataset-import-wizard">
            <div className="modal diw-modal">
                <div className="modal-header">
                    <h2 className="u-m0">Import dataset</h2>
                    <button className="btn btn-ghost" onClick={onClose} aria-label="Close">
                        <X className="u-icon" aria-hidden="true" />
                    </button>
                </div>

                {/* Step indicator */}
                <div className="diw-steps">
                    {(Object.keys(stepLabel) as WizardStep[]).map((s) => (
                        <div
                            key={s}
                            className={s === step ? 'diw-step diw-step--active' : 'diw-step'}
                        >
                            {stepLabel[s]}
                        </div>
                    ))}
                </div>

                <div className="modal-body diw-body">
                    {step === 'source' && (
                        <SourceStep
                            sources={sources}
                            sourceId={sourceId}
                            onSourceChange={(id) => { setSourceId(id); setUploadedLocator(''); }}
                            locatorBody={locatorBody}
                            onLocatorBodyChange={(value) => { setLocatorBody(value); setUploadedLocator(''); }}
                            uploading={uploading}
                            uploadedName={uploadedName}
                            onUploadFile={handleUploadFile}
                            sourceHelp={sourceHelp}
                            introspecting={introspecting}
                            introspectError={introspectError}
                            onNext={handleIntrospect}
                        />
                    )}

                    {step === 'recipe' && introspection && (
                        <>
                            <TaskShapeConfirmCard
                                projectId={projectId}
                                detection={introspection.task_shape}
                                onConfirmed={(_result, candidate) => {
                                    // Confirmed server-side (adapter preset +
                                    // recipe). Carry the detector's mapping
                                    // into the map step.
                                    if (candidate?.mapper_id) {
                                        setSelectedMapperId(candidate.mapper_id);
                                        if (Object.keys(candidate.field_map || {}).length > 0) {
                                            setFieldMapJson(fieldMapToJsonString(candidate.field_map));
                                        }
                                    }
                                    setSelectedRecipe(null);
                                    setRecipeSuggestion(null);
                                    setRecipeOverridden(false);
                                    setStep('map');
                                }}
                            />
                            {introspection.task_shape && (
                                <p className="form-hint">Or choose the task type yourself:</p>
                            )}
                            <RecipePicker
                                headers={introspection.columns}
                                onSelect={(recipe, suggestion) => {
                                    setSelectedRecipe(recipe);
                                    setRecipeSuggestion(suggestion);
                                    setRecipeOverridden(false);
                                    setRecipePersistError('');
                                    // Best-effort persist: snapshot the
                                    // recipe + propagate the suggested
                                    // base model to the project so the
                                    // Training tab inherits sensible
                                    // defaults. The picker UX still
                                    // advances even if the call fails;
                                    // the import flow doesn't depend on
                                    // this side-effect.
                                    applyRecipeToProject(projectId, recipe.id).catch(
                                        (err) => {
                                            setRecipePersistError(extractErrorMessage(err));
                                        },
                                    );
                                    setStep('map');
                                }}
                                onOverride={() => {
                                    setSelectedRecipe(null);
                                    setRecipeSuggestion(null);
                                    setRecipeOverridden(true);
                                    setRecipePersistError('');
                                    setStep('map');
                                }}
                                onBack={() => setStep('source')}
                            />
                        </>
                    )}

                    {step === 'map' && introspection && (
                        <>
                            {selectedRecipe && (
                                <RecipeSummaryChip
                                    recipe={selectedRecipe}
                                    suggestion={recipeSuggestion}
                                    onChange={() => setStep('recipe')}
                                />
                            )}
                            {recipePersistError && (
                                <div
                                    role="alert"
                                    data-testid="recipe-persist-error"
                                    className="diw-notice"
                                >
                                    Task type selection didn't persist to the project
                                    record ({recipePersistError}). You can continue
                                    importing; training defaults will need to be set
                                    by hand later.
                                </div>
                            )}
                            {recipeOverridden && (
                                <div
                                    role="note"
                                    className="diw-notice"
                                >
                                    Task type suggestions skipped. Configuring mapping
                                    manually.{' '}
                                    <button
                                        type="button"
                                        className="btn btn-ghost diw-link-btn"
                                        onClick={() => setStep('recipe')}
                                    >
                                        Show task type suggestions →
                                    </button>
                                </div>
                            )}
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
                                onBack={() => setStep('recipe')}
                                onPreview={handlePreview}
                            />
                        </>
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

// ── Recipe summary chip (shown above the Map step) ──────────────────

interface RecipeSummaryChipProps {
    recipe: Recipe;
    suggestion: RecipeSuggestion | null;
    onChange: () => void;
}

function RecipeSummaryChip({ recipe, suggestion, onChange }: RecipeSummaryChipProps) {
    const confidencePct =
        suggestion && !suggestion.fallback ? Math.round(suggestion.confidence * 100) : null;
    return (
        <div
            data-testid="recipe-summary-chip"
            className="diw-chip"
        >
            <span className="diw-chip__icon" aria-hidden="true">
                {recipe.icon}
            </span>
            <div className="diw-chip__body">
                <strong>Task type:</strong> {recipe.name}{' '}
                {confidencePct !== null && (
                    <span className="diw-muted">
                        ({confidencePct}% match)
                    </span>
                )}
                <div className="diw-muted diw-small">
                    <code>{recipe.task_profile}</code> ·{' '}
                    <code>{recipe.scoring_mode}</code> · base{' '}
                    <code>{recipe.suggested_base_model}</code>
                </div>
            </div>
            <button
                type="button"
                className="btn btn-ghost diw-small"
                onClick={onChange}
            >
                Change task type
            </button>
        </div>
    );
}

// ── Step 1: Source ───────────────────────────────────────────────────

interface SourceStepProps {
    uploading: boolean;
    uploadedName: string;
    onUploadFile: (file: File) => void;
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
    uploading,
    uploadedName,
    onUploadFile,
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
        <div className="diw-stack">
            <div className="form-group" data-testid="upload-block">
                <label className="form-label" htmlFor="dsi-upload">
                    Upload a file from your computer
                </label>
                <input
                    id="dsi-upload"
                    type="file"
                    className="input"
                    accept=".csv,.tsv,.xlsx,.json,.jsonl,.parquet"
                    disabled={uploading}
                    data-testid="upload-input"
                    onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) onUploadFile(file);
                    }}
                />
                <div className="diw-hint">
                    {uploading
                        ? 'Uploading and reading columns…'
                        : uploadedName
                            ? `Loaded ${uploadedName}.`
                            : 'CSV, TSV, Excel (.xlsx), JSON, JSONL or Parquet — any encoding. We read the columns and propose how to use them.'}
                </div>
            </div>

            <p className="diw-muted u-m0">
                Or point at a source: pick the source type and enter a locator. The introspector
                will sniff the columns and propose a mapping — no need to write a converter.
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
                <div className="diw-row">
                    <code className="diw-prefix">
                        {sourceId}:
                    </code>
                    <input
                        className="input diw-grow"
                        value={locatorBody}
                        onChange={(e) => onLocatorBodyChange(e.target.value)}
                        placeholder={sourceHelp.placeholder}
                        data-testid="locator-input"
                    />
                </div>
                <div className="diw-hint">
                    {sourceHelp.helpText}
                </div>
            </div>

            {sourceHelp.authNote && (
                <div
                    className="diw-callout"
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

            <div className="diw-actions-end">
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
        <div className="diw-stack diw-stack--lg">
            {/* Column rundown */}
            <section>
                <h3 className="diw-h3">Column signatures</h3>
                <table className="table diw-table">
                    <thead>
                        <tr>
                            <th className="diw-th-left">Column</th>
                            <th className="diw-th-left">Detected type</th>
                            <th className="diw-th-left">Confidence</th>
                            <th className="diw-th-left">Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {introspection.column_signatures.map((sig) => (
                            <tr key={sig.name}>
                                <td><code>{sig.name}</code></td>
                                <td>{sig.column_type}</td>
                                <td>{(sig.confidence * 100).toFixed(0)}%</td>
                                <td className="diw-muted">
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
                <h3 className="diw-h3">Mapper</h3>
                {introspection.hypotheses.length === 0 && (
                    <div
                        className="diw-callout diw-callout--spaced"
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
                    <div className="diw-rationale">
                        <strong>Rationale:</strong> {hypothesesById.get(selectedMapperId)?.rationale}
                    </div>
                )}
            </section>

            {/* Confidence gate */}
            {isBelowThreshold && (
                <div
                    className="diw-callout diw-callout--error"
                    role="alert"
                    data-testid="confidence-warning"
                >
                    <div className="diw-strong-line">Low-confidence proposal</div>
                    <div className="diw-gate-note">
                        Confidence is below the {(confidenceThreshold * 100).toFixed(0)}% threshold.
                        Eyeball the column signatures + rationale carefully before proceeding.
                    </div>
                    <label className="diw-check">
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
                <h3 className="diw-h3">Field map</h3>
                <p className="diw-hint diw-hint--flush">
                    JSON object passed through to the mapper. Edit to override the
                    introspector's suggestion (rename a source column, supply a nested
                    config like <code>entity_type_map</code>, etc.).
                </p>
                <textarea
                    className="input diw-textarea"
                    rows={8}
                    value={fieldMapJson}
                    onChange={(e) => onFieldMapChange(e.target.value)}
                    data-testid="field-map-input"
                />
                {fieldMapError && (
                    <div className="error-banner diw-error-spaced">
                        {fieldMapError}
                    </div>
                )}
            </section>

            {previewError && (
                <div className="error-banner" data-testid="preview-error">
                    {previewError}
                </div>
            )}

            <div className="diw-actions-between">
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
            <div className="diw-stack">
                <div
                    className="diw-callout diw-callout--success"
                    role="status"
                    data-testid="run-success-banner"
                >
                    <h3 className="u-mt0">Imported.</h3>
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
                        <div className="diw-hint diw-hint--below">
                            File: <code>{runResult.written_path}</code>
                        </div>
                    )}
                </div>
                <div className="diw-actions-end">
                    <button className="btn btn-primary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="diw-stack diw-stack--lg">
            {/* Summary */}
            <section>
                <h3 className="diw-h3">Dry-run summary</h3>
                <div className="diw-cards">
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
                    <h3 className="diw-h3">
                        Rejected rows by reason
                    </h3>
                    <p className="diw-hint diw-hint--flush">
                        Tick a reason to bulk-drop those rejects on import. Counts are
                        preserved in the final result for audit. Refresh the preview after
                        toggling to see the updated sample.
                    </p>
                    <table className="table diw-table">
                        <thead>
                            <tr>
                                <th className="diw-th-narrow">Drop</th>
                                <th className="diw-th-left">Reason</th>
                                <th className="diw-th-right">Count</th>
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
                                    <td className="diw-th-right">{count}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <div className="u-mt-sm">
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
                    <h3 className="diw-h3">
                        Sample transformed rows
                    </h3>
                    <pre className="diw-pre">
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
            <section className="diw-save">
                <div className="diw-save__head">
                    <div>
                        <strong>Save this mapping</strong>
                        <div className="diw-muted diw-small">
                            Stores the locator + mapper + field_map + drop_reasons so you can
                            re-run against a refreshed source without re-introspecting.
                        </div>
                    </div>
                    {savedConfigName ? (
                        <span
                            className="diw-muted diw-small"
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
                    <div className="diw-save__form">
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
                        <div className="diw-actions-end">
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

            <div className="diw-actions-between">
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
    const toneClass: Record<SummaryCardProps['tone'], string> = {
        positive: 'diw-card diw-card--positive',
        warning: 'diw-card diw-card--warning',
        neutral: 'diw-card',
    };
    return (
        <div className={toneClass[tone]}>
            <div className="diw-card__label">
                {label}
            </div>
            <div className="diw-card__value">{value}</div>
        </div>
    );
}
