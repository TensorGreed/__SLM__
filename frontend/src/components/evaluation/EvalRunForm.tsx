/**
 * EvalRunForm — the short "Score this run" form on Results › Verdict (UI simplification pass, phase 2).
 *
 * Page one: what to score (built-in set or eval set, + version and prompt format for an eval set), which
 * model ("Fine-tuned run #N" / "Base model <name>" / "Both, base vs fine-tuned"), rows ("All N" / "First
 * N"), Run. Everything else (metric, max new tokens, temperature, a raw model path, the dialogue judge,
 * checked decoding, null as absent, the LLM judge provider and its local serve runtime incl. host / port)
 * sits in the shared ``AdvancedSection`` (``score-run-advanced``; ⌘K "Score this run: advanced options").
 *
 * The form holds every field's state itself (it is the parent of its Advanced section), so hidden
 * fields keep sending what they send today. The body is ``buildHeldoutPayload`` — the long form's body,
 * unchanged: "Base model" = Model Path set to the run's base model (the server files that result under
 * the baseline run), "Both" = the fine-tuned body then the base body. A raw Model path under Advanced
 * wins (one request, as before).
 *
 * UI-only prefills (decision 3; server defaults unchanged): "All N rows" for a set of ≤ 2,000 rows, max new
 * tokens suggested from the set's longest answer, and the last-used values per project (localStorage,
 * every access in try/catch). Each prefill says where it came from.
 */

import { useEffect, useState } from 'react';

import api from '../../api/client';
import { fetchEvalSetRows, fetchVariantSupport, isEvalSetName, type EvalSet } from '../../api/evalSets';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import AdvancedSection from '../shared/AdvancedSection';
import CheckedDecodingSettings from './CheckedDecodingSettings';
import NullAsAbsentSetting, { type NullRunChoice } from './NullAsAbsentSetting';
import {
    ALL_ROWS_PREFILL_MAX,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SAMPLES,
    MAX_NEW_TOKENS_CEILING,
    MAX_SAMPLES_CEILING,
    buildHeldoutPayload,
    longestAnswerChars,
    modelPathsFor,
    parseJsonl,
    readLastUsed,
    suggestMaxNewTokens,
    writeLastUsed,
    type HeldoutEvalType,
    type JudgeProvider,
    type ModelChoice,
} from './evalRunPayload';
import { isBaselineRun, type RunSummary } from './runSelection';
import './EvalRunForm.css';

export const SCORE_RUN_ADVANCED_ID = 'score-run-advanced';

interface LocalJudgeServeRun {
    run_id: string;
    status?: string;
    template_id?: string;
    template_name?: string;
}

interface EvalRunFormProps {
    projectId: number;
    /** The selected run (its ``base_model`` names the base model). */
    run: RunSummary & { base_model?: string | null };
    evalSets: EvalSet[];
    /** The page's ``?open=`` value (opens the Advanced section). */
    openSection?: string | null;
    openNonce?: number;
    /** Called once the eval(s) are queued. */
    onQueued?: () => void;
    /** "Add an eval set": opens Verdict › Test sets. */
    onOpenTestSets?: () => void;
}

const BUILT_IN_SETS: Array<{ value: string; label: string; manifestKey: string | null }> = [
    { value: 'test', label: 'Test examples', manifestKey: 'test' },
    { value: 'validation', label: 'Validation', manifestKey: 'val' },
    { value: 'gold_test', label: 'Final answer key', manifestKey: null },
    { value: 'gold_dev', label: 'Practice answer key', manifestKey: null },
];

function defaultJudgeModel(provider: JudgeProvider): string {
    if (provider === 'local_serve') return 'local-judge';
    if (provider === 'ollama') return 'llama3';
    if (provider === 'openai') return 'gpt-4o';
    return 'meta-llama/Meta-Llama-3-70B-Instruct';
}

function errorText(error: unknown): string {
    const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (detail && typeof detail === 'object') {
        const d = detail as { message?: unknown; actionable_fix?: unknown; error_code?: unknown };
        const message = typeof d.message === 'string' ? d.message.trim() : '';
        const fix = typeof d.actionable_fix === 'string' ? d.actionable_fix.trim() : '';
        if (message) return fix ? `${message} ${fix}` : message;
        if (typeof d.error_code === 'string' && d.error_code.trim()) return d.error_code.trim();
    }
    if (error instanceof Error) return error.message;
    return 'Failed to run evaluation';
}

/** Longest answer per set key: chars, null = no answer field found, 'error' = could not read the rows. */
type LongestCache = Record<string, number | null | 'error'>;

export default function EvalRunForm({ projectId, run, evalSets, openSection = null, openNonce = 0, onQueued, onOpenTestSets }: EvalRunFormProps) {
    // Last-used values, read once when the form opens.
    const [last] = useState(() => readLastUsed(projectId));
    const isBaseline = isBaselineRun(run);
    const baseModel = (run.base_model ?? '').trim() || null;

    const [datasetName, setDatasetName] = useState<string>(() => last.datasetName || 'test');
    const [evalSetVersion, setEvalSetVersion] = useState<number | null>(null);
    const [evalSetVariant, setEvalSetVariant] = useState('');
    const [modelChoice, setModelChoice] = useState<ModelChoice>(() => last.model ?? 'finetuned');
    const [rowsChoice, setRowsChoice] = useState<'all' | 'first' | null>(null);
    const [firstN, setFirstN] = useState<number>(() => last.firstN ?? DEFAULT_MAX_SAMPLES);
    const [evalType, setEvalType] = useState<HeldoutEvalType>(() => last.evalType ?? 'exact_match');
    const [maxNewTokens, setMaxNewTokens] = useState<number | null>(null);
    const [temperature, setTemperature] = useState<number>(() => last.temperature ?? 0);
    const [modelPath, setModelPath] = useState('');
    const [dialogueJudge, setDialogueJudge] = useState<boolean>(() => last.dialogueJudge ?? false);
    const [nullRunChoice, setNullRunChoice] = useState<NullRunChoice>('');
    const [provider, setProvider] = useState<JudgeProvider>(() => last.provider ?? 'hf');
    const [judgeModel, setJudgeModel] = useState<string>(() => last.judgeModel ?? defaultJudgeModel(last.provider ?? 'hf'));
    const [localServeRuns, setLocalServeRuns] = useState<LocalJudgeServeRun[]>([]);
    const [loadingLocalServeRuns, setLoadingLocalServeRuns] = useState(false);
    const [selectedLocalServeRunId, setSelectedLocalServeRunId] = useState('auto');
    const [localServeAutoStart, setLocalServeAutoStart] = useState(false);
    const [localServeAutoStop, setLocalServeAutoStop] = useState(true);
    const [localServeSource, setLocalServeSource] = useState<'export' | 'registry'>('export');
    const [localServeExportId, setLocalServeExportId] = useState('');
    const [localServeModelId, setLocalServeModelId] = useState('');
    const [localServeTemplateId, setLocalServeTemplateId] = useState('runner.vllm');
    const [localServeHost, setLocalServeHost] = useState('127.0.0.1');
    const [localServePort, setLocalServePort] = useState('8000');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // What is shown is what is sent: a remembered eval set that is not (yet) listed reads as the test examples.
    const pickedEvalSet = isEvalSetName(datasetName)
        ? evalSets.find((s) => `eval_set:${s.slug}` === datasetName) ?? null
        : null;
    const shownDataset = isEvalSetName(datasetName) && !pickedEvalSet ? 'test' : datasetName;
    const builtIn = BUILT_IN_SETS.find((s) => s.value === shownDataset) ?? null;
    const version = pickedEvalSet ? (evalSetVersion ?? pickedEvalSet.current_version) : null;

    // ── Set facts: row count (split manifest / eval-set version) and the longest answer ──────────
    const [manifestSplits, setManifestSplits] = useState<Record<string, number> | null>(null);
    useEffect(() => {
        let alive = true;
        api.get<{ splits?: Record<string, unknown> } | null>(`/projects/${projectId}/prepared-manifest`)
            .then((res) => {
                const splits = res?.data?.splits;
                if (!alive || !splits || typeof splits !== 'object') return;
                const counts: Record<string, number> = {};
                for (const [key, value] of Object.entries(splits)) {
                    if (typeof value === 'number' && Number.isFinite(value)) counts[key] = value;
                }
                setManifestSplits(counts);
            })
            .catch(() => { /* no manifest: row counts unknown, today's defaults apply */ });
        return () => { alive = false; };
    }, [projectId]);

    const setRows: number | null = pickedEvalSet
        ? (pickedEvalSet.versions.find((v) => v.version === version)?.rows ?? (version === pickedEvalSet.current_version ? pickedEvalSet.rows : null))
        : builtIn?.manifestKey && manifestSplits && typeof manifestSplits[builtIn.manifestKey] === 'number'
            ? manifestSplits[builtIn.manifestKey]
            : null;
    const setKey = pickedEvalSet ? `eval_set:${pickedEvalSet.slug}@v${version}` : shownDataset;

    const [longest, setLongest] = useState<LongestCache>({});
    const longestKnown = Object.prototype.hasOwnProperty.call(longest, setKey);
    useEffect(() => {
        if (longestKnown || setRows == null || setRows < 1 || setRows > ALL_ROWS_PREFILL_MAX) return undefined;
        let alive = true;
        const load = async (): Promise<unknown[]> => {
            if (pickedEvalSet && version != null) {
                const rows: unknown[] = [];
                for (let offset = 0; offset < setRows; offset += 200) {
                    const page = await fetchEvalSetRows(projectId, pickedEvalSet.slug, version, offset, 200);
                    const pageRows = Array.isArray(page?.rows) ? page.rows : [];
                    if (pageRows.length === 0) break;
                    rows.push(...pageRows.map((r) => r.row));
                }
                return rows;
            }
            const split = builtIn?.manifestKey;
            if (!split) return [];
            const res = await api.get<string>(`/projects/${projectId}/data-studio/dataset-versions/${split}/export`, { responseType: 'text' });
            return parseJsonl(res?.data);
        };
        load()
            .then((rows) => { if (alive) setLongest((prev) => ({ ...prev, [setKey]: longestAnswerChars(rows) })); })
            .catch(() => { if (alive) setLongest((prev) => ({ ...prev, [setKey]: 'error' })); });
        return () => { alive = false; };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the set
    }, [projectId, setKey, setRows, longestKnown]);

    // ── Rows ──────────────────────────────────────────────────────────────────────
    const canAll = setRows != null && setRows >= 1 && setRows <= MAX_SAMPLES_CEILING;
    const rowsSource: 'you' | 'last' | 'prefill' | 'default' = rowsChoice ? 'you' : last.rows ? 'last' : setRows != null ? 'prefill' : 'default';
    const wantedRows = rowsChoice ?? last.rows ?? (setRows != null && setRows <= ALL_ROWS_PREFILL_MAX ? 'all' : 'first');
    const rowsMode: 'all' | 'first' = wantedRows === 'all' && canAll ? 'all' : 'first';
    const maxSamples = rowsMode === 'all' && setRows != null ? setRows : firstN;

    // ── Max new tokens ────────────────────────────────────────────────────────────
    const longestChars = longest[setKey];
    const suggestion = typeof longestChars === 'number' ? suggestMaxNewTokens(longestChars) : null;
    const tokens = maxNewTokens ?? last.maxNewTokens ?? suggestion ?? DEFAULT_MAX_NEW_TOKENS;
    const tokensSource = maxNewTokens != null
        ? 'set under Advanced'
        : last.maxNewTokens != null
            ? 'last used'
            : suggestion != null
                ? `suggested from the longest answer in this set, ${longestChars} characters`
                : setRows != null && setRows > ALL_ROWS_PREFILL_MAX
                    ? `default; no suggestion for a set over ${ALL_ROWS_PREFILL_MAX.toLocaleString()} rows`
                    : longestChars === 'error'
                        ? 'default; the set rows could not be read'
                        : longestKnown
                            ? 'default; no answer field found in the rows'
                            : 'default';

    // ── Prompt-format variant support (the server says when the run renders no system message) ──
    const [variantBlockedReason, setVariantBlockedReason] = useState<string | null>(null);
    const pickedSetHasVariants = (pickedEvalSet?.variants.length ?? 0) > 0;
    useEffect(() => {
        let active = true;
        setVariantBlockedReason(null);
        if (!pickedSetHasVariants) return () => { active = false; };
        fetchVariantSupport(projectId, run.id)
            .then((support) => {
                if (!active) return;
                const blocked = support?.supported === false;
                setVariantBlockedReason(blocked ? (support.reason ?? 'Prompt-format variants do not apply to this run.') : null);
                if (blocked) setEvalSetVariant('');
            })
            .catch(() => { /* unknown: the server still refuses an unsupported variant with its reason */ });
        return () => { active = false; };
    }, [projectId, run.id, pickedSetHasVariants]);

    // ── LLM judge: local serve runs ─────────────────────────────────────────────────
    const loadLocalServeRuns = async () => {
        setLoadingLocalServeRuns(true);
        try {
            const res = await api.get<{ runs?: LocalJudgeServeRun[] }>(
                `/projects/${projectId}/evaluation/local-judge/serve-runs`,
                { params: { limit: 50 } },
            );
            setLocalServeRuns(Array.isArray(res.data?.runs) ? res.data.runs : []);
        } catch {
            setLocalServeRuns([]);
        } finally {
            setLoadingLocalServeRuns(false);
        }
    };
    useEffect(() => {
        if (evalType !== 'llm_judge' || provider !== 'local_serve') return;
        void loadLocalServeRuns();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId, evalType, provider]);

    const handleProviderChange = (next: JudgeProvider) => {
        setProvider(next);
        setJudgeModel(defaultJudgeModel(next));
    };

    // ── Which model ───────────────────────────────────────────────────────────────
    const rawPathSet = modelPath.trim() !== '';
    const effectiveChoice: ModelChoice = isBaseline || !baseModel ? 'finetuned' : modelChoice;
    const paths = modelPathsFor(effectiveChoice, baseModel, modelPath);

    const submit = async () => {
        setIsSubmitting(true);
        setError(null);
        const state = {
            experimentId: run.id,
            datasetName: shownDataset,
            pickedEvalSet,
            evalSetVersion,
            evalSetVariant,
            variantBlockedReason,
            evalType,
            maxSamples,
            maxNewTokens: tokens,
            temperature,
            modelPath,
            dialogueJudge,
            nullRunChoice,
            provider,
            judgeModel,
            selectedLocalServeRunId,
            localServeAutoStart,
            localServeAutoStop,
            localServeSource,
            localServeExportId,
            localServeModelId,
            localServeTemplateId,
            localServeHost,
            localServePort,
        };
        const jobIds: Array<number | undefined> = [];
        try {
            for (const path of paths) {
                // Each body is the long form's body for the same inputs with Model Path = ``path``.
                const res = await api.post<{ id: number }>(
                    `/projects/${projectId}/evaluation/run-heldout?async_job=true`,
                    buildHeldoutPayload({ ...state, modelPath: path }),
                );
                jobIds.push(res.data?.id);
            }
        } catch (err) {
            const queued = jobIds.length > 0
                ? `The fine-tuned eval was queued${jobIds[0] ? ` (job #${jobIds[0]})` : ''}; the base-model eval failed: `
                : '';
            setError(`${queued}${errorText(err)}`);
            if (jobIds.length > 0) void useJobsStore.getState().refreshAfterLocalChange();
            setIsSubmitting(false);
            return;
        }
        writeLastUsed(projectId, {
            datasetName: shownDataset,
            model: modelChoice,
            rows: rowsMode,
            firstN,
            evalType,
            maxNewTokens: maxNewTokens ?? last.maxNewTokens,
            temperature,
            dialogueJudge,
            provider,
            judgeModel,
        });
        const ids = jobIds.filter((id): id is number => typeof id === 'number');
        toast.info(
            paths.length > 1
                ? `2 evals queued: fine-tuned, then base model${ids.length ? ` (jobs ${ids.map((id) => `#${id}`).join(', ')})` : ''} — bell will notify when ready`
                : ids.length
                    ? `Eval queued — bell will notify when ready (job #${ids[0]})`
                    : 'Eval queued — bell will notify when ready',
            4000,
        );
        void useJobsStore.getState().refreshAfterLocalChange();
        setIsSubmitting(false);
        onQueued?.();
    };

    const setLabel = pickedEvalSet
        ? `${pickedEvalSet.name} v${version}${evalSetVariant && !variantBlockedReason ? ` (${evalSetVariant})` : ''}`
        : builtIn?.label ?? shownDataset;
    const rowsProvenance = setRows == null
        ? `${setLabel}: row count unknown`
        : `${setLabel}: ${setRows.toLocaleString()} rows${pickedEvalSet ? '' : ' (prepared split)'}`;
    const advancedCount = 7 + (evalType === 'llm_judge' ? 2 : 0) + (evalType === 'llm_judge' && provider === 'local_serve' ? 1 : 0);
    const metricLabel = evalType === 'exact_match' ? 'Exact match' : evalType === 'f1' ? 'F1' : 'LLM judge';

    return (
        <div className="card eval-panel__card--accent eval-run-form" data-testid="eval-run-form">
            <h3 className="u-card-title">Score this run</h3>
            <p className="u-hint u-mb-lg">
                Runs the model on rows it never trained on and stores the score on run #{run.id} · {run.name}.
            </p>

            <div className="eval-run-form__row">
                <label className="eval-run-form__label" htmlFor="eval-dataset-select">What</label>
                <div className="eval-run-form__control">
                    <select
                        id="eval-dataset-select"
                        className="input u-w-auto"
                        value={shownDataset}
                        onChange={(e) => { setDatasetName(e.target.value); setEvalSetVersion(null); setEvalSetVariant(''); }}
                        data-testid="eval-dataset-select"
                    >
                        {BUILT_IN_SETS.map((s) => (
                            <option key={s.value} value={s.value}>{s.label}</option>
                        ))}
                        {evalSets.length > 0 && (
                            <optgroup label="Eval sets">
                                {evalSets.map((s) => (
                                    <option key={s.slug} value={`eval_set:${s.slug}`}>
                                        {s.name} · v{s.current_version} · {s.rows} rows
                                    </option>
                                ))}
                            </optgroup>
                        )}
                    </select>
                    {pickedEvalSet && (
                        <select
                            id="eval-set-version"
                            className="input u-w-auto"
                            aria-label="Eval-set version"
                            value={version ?? pickedEvalSet.current_version}
                            onChange={(e) => setEvalSetVersion(Number(e.target.value))}
                            data-testid="eval-set-version-select"
                        >
                            {[...pickedEvalSet.versions].reverse().map((v) => (
                                <option key={v.version} value={v.version}>
                                    v{v.version} · {v.rows} rows{v.version === pickedEvalSet.current_version ? ' (current)' : ''}
                                </option>
                            ))}
                        </select>
                    )}
                    {pickedEvalSet && pickedSetHasVariants && (
                        <select
                            id="eval-set-variant"
                            className="input u-w-auto"
                            aria-label="Prompt format"
                            value={variantBlockedReason ? '' : evalSetVariant}
                            onChange={(e) => setEvalSetVariant(e.target.value)}
                            disabled={variantBlockedReason !== null}
                            data-testid="eval-set-variant-select"
                        >
                            <option value="">Prompt format: as in the rows</option>
                            {pickedEvalSet.variants.map((v) => (
                                <option key={v.name} value={v.name}>{v.name} · {v.label}</option>
                            ))}
                        </select>
                    )}
                    {onOpenTestSets && (
                        <button type="button" className="btn btn-link" onClick={onOpenTestSets} data-testid="eval-run-add-eval-set">
                            Add an eval set
                        </button>
                    )}
                    {variantBlockedReason && (
                        <p className="u-hint u-m0 eval-run-form__full" data-testid="eval-set-variant-blocked">{variantBlockedReason}</p>
                    )}
                </div>
            </div>

            <div className="eval-run-form__row">
                <span className="eval-run-form__label">Which model</span>
                {isBaseline ? (
                    <div className="eval-run-form__control">
                        <span className="u-meta" data-testid="eval-run-model-baseline">
                            This run is the base model (a lift-check baseline){baseModel ? `: ${baseModel}` : ''}.
                        </span>
                    </div>
                ) : (
                    <div className="eval-run-form__control" role="radiogroup" aria-label="Which model">
                        <label className="eval-run-form__radio">
                            <input type="radio" name="eval-run-model" checked={effectiveChoice === 'finetuned'} disabled={rawPathSet}
                                onChange={() => setModelChoice('finetuned')} data-testid="eval-run-model-finetuned" />
                            Fine-tuned run #{run.id}
                        </label>
                        <label className="eval-run-form__radio">
                            <input type="radio" name="eval-run-model" checked={effectiveChoice === 'base'} disabled={rawPathSet || !baseModel}
                                onChange={() => setModelChoice('base')} data-testid="eval-run-model-base" />
                            Base model{baseModel ? <code>{baseModel}</code> : ''}
                        </label>
                        <label className="eval-run-form__radio">
                            <input type="radio" name="eval-run-model" checked={effectiveChoice === 'both'} disabled={rawPathSet || !baseModel}
                                onChange={() => setModelChoice('both')} data-testid="eval-run-model-both" />
                            Both, base vs fine-tuned
                            <small>two evals</small>
                        </label>
                        {!baseModel && (
                            <span className="u-meta eval-run-form__full">The run records no base model; to score another model use Advanced › Model path.</span>
                        )}
                        {rawPathSet && (
                            <span className="u-meta eval-run-form__full" data-testid="eval-run-model-raw-path">
                                Advanced › Model path is set: one eval answered by {modelPath.trim()}.
                            </span>
                        )}
                        {!rawPathSet && effectiveChoice !== 'finetuned' && (
                            <span className="u-meta eval-run-form__full" data-testid="eval-run-model-note">
                                The base model&apos;s result is filed under its baseline run, so the Verdict card reads base vs fine-tuned.
                            </span>
                        )}
                    </div>
                )}
            </div>

            <div className="eval-run-form__row">
                <span className="eval-run-form__label">Rows</span>
                <div className="eval-run-form__control" role="radiogroup" aria-label="Rows">
                    {canAll && setRows != null && (
                        <label className="eval-run-form__radio">
                            <input type="radio" name="eval-run-rows" checked={rowsMode === 'all'} onChange={() => setRowsChoice('all')}
                                data-testid="eval-run-rows-all" />
                            All {setRows.toLocaleString()}
                        </label>
                    )}
                    <label className="eval-run-form__radio">
                        <input type="radio" name="eval-run-rows" checked={rowsMode === 'first'} onChange={() => setRowsChoice('first')}
                            data-testid="eval-run-rows-first" />
                        First
                    </label>
                    <input
                        className="input eval-run-form__number"
                        type="number"
                        min={1}
                        max={MAX_SAMPLES_CEILING}
                        aria-label="First N rows"
                        value={firstN}
                        onChange={(e) => { setFirstN(Number(e.target.value)); setRowsChoice('first'); }}
                        data-testid="eval-run-rows-first-n"
                    />
                    <span className="u-meta eval-run-form__full" data-testid="eval-run-rows-provenance">
                        {rowsProvenance}
                        {rowsSource === 'prefill' && rowsMode === 'all' ? ` · all rows prefilled (sets of up to ${ALL_ROWS_PREFILL_MAX.toLocaleString()} rows)` : ''}
                        {rowsSource === 'last' ? ' · your last choice' : ''}
                        {rowsMode === 'first' && setRows != null && firstN < setRows ? ` · partial: ${Math.max(1, firstN)} of ${setRows.toLocaleString()} rows` : ''}
                    </span>
                </div>
            </div>

            <p className="u-meta eval-run-form__sends" data-testid="eval-run-sends">
                Also sent: metric {metricLabel} · max {tokens} new tokens ({tokensSource}) · temperature {temperature}
                {dialogueJudge ? ' · every assistant turn judged' : ''}
                {nullRunChoice ? ` · null counts as absent: ${nullRunChoice}` : ''} — change under Advanced.
            </p>

            <AdvancedSection
                pageKey="results-verdict"
                id={SCORE_RUN_ADVANCED_ID}
                count={advancedCount}
                hint="metric, max new tokens, temperature, model path, judges, decoding, JSON scoring"
                keepMounted
                openSection={openSection}
                openNonce={openNonce}
            >
                <div className="eval-panel__grid-3">
                    <div className="form-group">
                        <label className="form-label" htmlFor="eval-run-metric">Metric</label>
                        <select id="eval-run-metric" className="input" value={evalType}
                            onChange={(e) => setEvalType(e.target.value as HeldoutEvalType)} data-testid="eval-run-metric">
                            <option value="exact_match">Exact Match</option>
                            <option value="f1">F1</option>
                            <option value="llm_judge">LLM Judge</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="eval-run-max-new-tokens">Max New Tokens</label>
                        <input id="eval-run-max-new-tokens" className="input" type="number" min={1} max={MAX_NEW_TOKENS_CEILING}
                            value={tokens} onChange={(e) => setMaxNewTokens(Number(e.target.value))} data-testid="eval-run-max-new-tokens" />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="eval-run-temperature">Temperature</label>
                        <input id="eval-run-temperature" className="input" type="number" min={0} max={2} step={0.1}
                            value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} data-testid="eval-run-temperature" />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="eval-run-model-path">Model path (optional)</label>
                        <input id="eval-run-model-path" className="input" value={modelPath} onChange={(e) => setModelPath(e.target.value)}
                            placeholder="Leave blank to use the run's own model" data-testid="eval-run-model-path" />
                        <p className="u-hint u-mt0">Any model id or path; overrides Which model (one eval).</p>
                    </div>
                </div>

                <div className="form-group">
                    <label className="eval-panel__toggle">
                        <input type="checkbox" checked={dialogueJudge} onChange={(e) => setDialogueJudge(e.target.checked)}
                            data-testid="eval-dialogue-judge-toggle" />
                        Judge every assistant turn (multi-turn rows)
                    </label>
                    <p className="u-hint u-mt0">
                        Replays each assistant turn from the gold history and judges it against that turn&apos;s
                        reference. One generation per turn, so it takes longer and costs more with a cloud judge.
                    </p>
                </div>

                <CheckedDecodingSettings projectId={projectId} />

                <NullAsAbsentSetting projectId={projectId} runChoice={nullRunChoice} onRunChoice={setNullRunChoice} />

                {evalType === 'llm_judge' && (
                    <div className="u-grid-2">
                        <div className="form-group">
                            <label className="form-label" htmlFor="eval-run-judge-provider">Judge Provider</label>
                            <select id="eval-run-judge-provider" className="input" value={provider}
                                onChange={(e) => handleProviderChange(e.target.value as JudgeProvider)} data-testid="eval-run-judge-provider">
                                <option value="local_serve">Local Serve Runtime</option>
                                <option value="hf">HuggingFace (vLLM)</option>
                                <option value="ollama">Local (Ollama)</option>
                                <option value="openai">Cloud (OpenAI)</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label" htmlFor="eval-run-judge-model">
                                {provider === 'local_serve' ? 'Judge Model Alias (served)' : 'Judge Model'}
                            </label>
                            <input id="eval-run-judge-model" className="input" value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)}
                                placeholder={provider === 'openai' ? 'gpt-4o' : provider === 'ollama' ? 'llama3' : provider === 'local_serve' ? 'local-judge' : 'meta-llama/...'} />
                        </div>
                    </div>
                )}

                {evalType === 'llm_judge' && provider === 'local_serve' && (
                    <div className="eval-pack-card u-mt-md" data-testid="eval-run-local-serve">
                        <div className="eval-pack-header">
                            <div>
                                <h4 className="eval-panel__h4">Local Judge Serve Runtime</h4>
                                <p className="eval-pack-subtitle">Select an existing serve run or auto-start one for judge evaluation.</p>
                            </div>
                            <button className="btn btn-secondary btn-sm" onClick={() => void loadLocalServeRuns()} disabled={loadingLocalServeRuns}>
                                {loadingLocalServeRuns ? 'Refreshing...' : 'Refresh Runs'}
                            </button>
                        </div>
                        <div className="u-grid-2">
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-serve-run">Serve Run</label>
                                <select id="eval-run-serve-run" className="input" value={selectedLocalServeRunId} onChange={(e) => setSelectedLocalServeRunId(e.target.value)}>
                                    <option value="auto">Auto (latest compatible run)</option>
                                    {localServeRuns.map((r) => (
                                        <option key={r.run_id} value={r.run_id}>
                                            {r.run_id} • {r.template_id || r.template_name || 'template'} • {r.status || 'unknown'}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-auto-start">Auto Start Missing Run</label>
                                <select id="eval-run-auto-start" className="input" value={localServeAutoStart ? 'yes' : 'no'}
                                    onChange={(e) => setLocalServeAutoStart(e.target.value === 'yes')}>
                                    <option value="no">No</option>
                                    <option value="yes">Yes</option>
                                </select>
                            </div>
                        </div>
                        <div className="u-grid-2">
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-auto-stop">Auto Stop After Eval</label>
                                <select id="eval-run-auto-stop" className="input" value={localServeAutoStop ? 'yes' : 'no'}
                                    onChange={(e) => setLocalServeAutoStop(e.target.value === 'yes')} disabled={!localServeAutoStart}>
                                    <option value="yes">Yes</option>
                                    <option value="no">No</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-auto-source">Auto Start Source</label>
                                <select id="eval-run-auto-source" className="input" value={localServeSource}
                                    onChange={(e) => setLocalServeSource(e.target.value as 'export' | 'registry')} disabled={!localServeAutoStart}>
                                    <option value="export">Export</option>
                                    <option value="registry">Registry</option>
                                </select>
                            </div>
                        </div>
                        <div className="u-grid-2">
                            {localServeSource === 'export' ? (
                                <div className="form-group">
                                    <label className="form-label" htmlFor="eval-run-export-id">Export ID (auto-start)</label>
                                    <input id="eval-run-export-id" className="input" value={localServeExportId} onChange={(e) => setLocalServeExportId(e.target.value)}
                                        disabled={!localServeAutoStart} placeholder="e.g. 12" />
                                </div>
                            ) : (
                                <div className="form-group">
                                    <label className="form-label" htmlFor="eval-run-registry-id">Registry Model ID (auto-start)</label>
                                    <input id="eval-run-registry-id" className="input" value={localServeModelId} onChange={(e) => setLocalServeModelId(e.target.value)}
                                        disabled={!localServeAutoStart} placeholder="e.g. 7" />
                                </div>
                            )}
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-template-id">Template ID (optional)</label>
                                <input id="eval-run-template-id" className="input" value={localServeTemplateId} onChange={(e) => setLocalServeTemplateId(e.target.value)}
                                    disabled={!localServeAutoStart} placeholder="runner.vllm" />
                            </div>
                        </div>
                        <div className="u-grid-2">
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-host">Host (auto-start)</label>
                                <input id="eval-run-host" className="input" value={localServeHost} onChange={(e) => setLocalServeHost(e.target.value)}
                                    disabled={!localServeAutoStart} placeholder="127.0.0.1" />
                            </div>
                            <div className="form-group">
                                <label className="form-label" htmlFor="eval-run-port">Port (auto-start)</label>
                                <input id="eval-run-port" className="input" value={localServePort} onChange={(e) => setLocalServePort(e.target.value)}
                                    disabled={!localServeAutoStart} placeholder="8000" />
                            </div>
                        </div>
                    </div>
                )}
            </AdvancedSection>

            {error && (
                <div className="eval-panel__error-card u-mt-md" role="alert" data-testid="eval-run-error">{error}</div>
            )}

            <div className="u-mt-lg">
                <button type="button" className="btn btn-primary" onClick={() => void submit()} disabled={isSubmitting} data-testid="eval-run-submit">
                    {isSubmitting ? 'Queuing…' : paths.length > 1 ? 'Run both' : 'Run'}
                </button>
            </div>
        </div>
    );
}
