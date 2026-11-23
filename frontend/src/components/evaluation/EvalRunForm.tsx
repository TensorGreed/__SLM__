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
 * wins (one request, as before). When the base leg of "Both" fails after the fine-tuned leg was queued,
 * "Retry base only" posts the base body alone.
 *
 * UI-only prefills (decision 3; server defaults unchanged): "All N rows" for a set of ≤ 2,000 rows, max new
 * tokens suggested from the set's longest answer (floor 128; a tool-call row's answer is its expected call), and the
 * last-used values per project (localStorage, every access in try/catch; set, prompt format, model choice, tokens per
 * set; never the LLM judge). Each prefill says where it came from.
 *
 * Phase 7b: "Both" queues both legs (the server runs them one after the other); a set with prompt-format variants
 * offers "Every prompt format" (one eval per variant, per chosen model) and, for "Both", "Base model on" another
 * prompt format (a labelled cross-prompt pair: fine-tuned on X vs base on Y).
 *
 * Phase 7c (UI-only): with nothing picked or remembered, "Which model" starts on Both when the run records a base
 * model, is not a baseline run and no base-model result is stored on this set + prompt format (the form says why and
 * that it means two evals); and "What" starts on the project's only eval set when there is no prepared test split.
 * Neither default is remembered as a choice.
 *
 * Comparability: with "Fine-tuned" picked and a base-model result already stored for the set (the lift's
 * baseline for this run on this set, from the lift summary), rows and max new tokens MATCH that result
 * ("matching the base result #N"); changing them away shows a warning ("pick Both or match them"). Run waits
 * for that check, and for a remembered eval set while the sets load.
 *
 * "Scored as: <mode>" (read-only) when a scoring mode decides the score: the newest result on this run that
 * stored one (``metrics.scoring_mode``). Not the prepared manifest's ``output_schema.scoring_mode``: a plain Q&A
 * split is stamped ``field_match`` there yet scored as free text (seen on the main-path E2E). The metric select
 * still sends what it sends today.
 */

import { useEffect, useRef, useState } from 'react';

import api from '../../api/client';
import { evalSetDatasetName, fetchEvalSetRows, fetchVariantSupport, isEvalSetName, type EvalSet } from '../../api/evalSets';
import { fetchSftLiftSummary } from '../../api/sftLift';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';
import AdvancedSection from '../shared/AdvancedSection';
import CheckedDecodingSettings from './CheckedDecodingSettings';
import NullAsAbsentSetting, { type NullRunChoice } from './NullAsAbsentSetting';
import {
    ALL_ROWS_PREFILL_MAX,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SAMPLES,
    ALL_PROMPT_FORMATS,
    MAX_NEW_TOKENS_CEILING,
    ROWS_AS_WRITTEN,
    STOP_IF_UNANSWERED_ROWS,
    STRUCTURED_UNSIZED_TOKENS,
    MAX_SAMPLES_CEILING,
    buildHeldoutPayload,
    effectiveRows,
    evalLegs,
    longestAnswerChars,
    modelPathsFor,
    parseJsonl,
    readLastUsed,
    resultRunSettings,
    suggestMaxNewTokens,
    writeLastUsed,
    type HeldoutEvalRequest,
    type HeldoutEvalType,
    type JudgeProvider,
    type ModelChoice,
} from './evalRunPayload';
import { isBaselineRun, type RunSummary } from './runSelection';
import { TASK_SCORING_MODES, scoringModeLabel, scoringModeOf, type VerdictResultLike } from './verdictScore';
import type { LiftFetcher } from './VerdictCard';
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
    /** The eval sets are still loading (a remembered eval set is not listed yet): Run waits. */
    evalSetsLoading?: boolean;
    /** The run's stored results (newest first): the scoring mode when the prepared data names none. */
    results?: VerdictResultLike[];
    /** Lift-summary fetch (the panel passes its shared one); finds the base result for the chosen set. */
    fetchLift?: LiftFetcher;
    /** The page's ``?open=`` value (opens the Advanced section). */
    openSection?: string | null;
    openNonce?: number;
    /** Called once the eval(s) are queued, with one line saying what was queued ("Queued 4 evals: …"; phase 7g G2:
     *  the panel collapses the form and shows it in the form's place). */
    onQueued?: (summary: string) => void;
    /** "Add an eval set": opens Verdict › Test sets. */
    onOpenTestSets?: () => void;
}

const BUILT_IN_SETS: Array<{ value: string; label: string; manifestKey: string | null }> = [
    { value: 'test', label: 'Test examples', manifestKey: 'test' },
    { value: 'validation', label: 'Validation', manifestKey: 'val' },
    { value: 'gold_test', label: 'Final answer key', manifestKey: null },
    { value: 'gold_dev', label: 'Practice answer key', manifestKey: null },
];

const PROVIDER_LABEL: Record<JudgeProvider, string> = {
    local_serve: 'local serve runtime',
    hf: 'HuggingFace (vLLM)',
    ollama: 'Ollama',
    openai: 'OpenAI',
};

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

/**
 * The Run button says what one click queues (phase 7e, gap G1): "Run" (one eval), "Run both" only for one fine-tuned
 * leg + one base leg, else "Run N evals: <who>, <K> prompt formats".
 */
function runLabelFor(legs: Array<{ modelPath: string; variant: string }>, baseModel: string | null): string {
    if (legs.length <= 1) return 'Run';
    const tuned = legs.filter((l) => l.modelPath === '').length;
    const base = baseModel ? legs.filter((l) => l.modelPath === baseModel).length : 0;
    if (legs.length === 2 && tuned === 1 && base === 1) return 'Run both';
    const who = tuned > 0 && base > 0 ? 'fine-tuned + base' : tuned > 0 ? 'fine-tuned' : base > 0 ? 'base model' : 'model path';
    const formats = new Set(legs.map((l) => l.variant)).size;
    return `Run ${legs.length} evals: ${who}${formats > 1 ? `, ${formats} prompt formats` : ''}`;
}

/** Longest answer per set key: chars, null = no answer field found, 'error' = could not read the rows. */
type LongestCache = Record<string, number | null | 'error'>;

/** The base-model result the lift would pair with this run on the chosen set, and what it ran with. */
interface BaseResultRef {
    resultId: number;
    runId: number;
    rowsRequested: number | null;
    rowsAnswered: number | null;
    tokens: number | null;
    /** False when the stored result names none of its rows / tokens: it still counts as a base result (no
     *  "Both" default), but there is nothing to match. */
    hasSettings: boolean;
}

/** Per lookup key: the base result, null = none stored, 'error' = could not be checked. */
type BaseCache = Record<string, BaseResultRef | null | 'error'>;

function hasKey(record: object, key: string): boolean {
    return Object.prototype.hasOwnProperty.call(record, key);
}

export default function EvalRunForm({
    projectId, run, evalSets, evalSetsLoading = false, results = [], fetchLift = fetchSftLiftSummary,
    openSection = null, openNonce = 0, onQueued, onOpenTestSets,
}: EvalRunFormProps) {
    // Last-used values, read once when the form opens.
    const [last] = useState(() => readLastUsed(projectId));
    const isBaseline = isBaselineRun(run);
    const baseModel = (run.base_model ?? '').trim() || null;

    // null = not picked (here or last time): the set prefill below applies, else the test examples.
    const [pickedDataset, setDatasetName] = useState<string | null>(() => last.datasetName || null);
    const [evalSetVersion, setEvalSetVersion] = useState<number | null>(null);
    // Remembered by name: shown only when the picked set has that variant (else "as in the rows"). null = not
    // picked (here or last time): a set with 2+ prompt formats starts on "Every prompt format" (phase 7e).
    const [evalSetVariant, setEvalSetVariant] = useState<string | null>(() => last.variant ?? null);
    // "Both": the base leg's prompt format when it is not the fine-tuned leg's ('' = the same).
    const [baseVariant, setBaseVariant] = useState<string>(() => last.baseVariant ?? '');
    // null = not picked (here or last time): the first-eval default below applies (Both or Fine-tuned).
    const [modelChoice, setModelChoice] = useState<ModelChoice | null>(() => last.model ?? null);
    const [rowsChoice, setRowsChoice] = useState<'all' | 'first' | null>(null);
    const [firstN, setFirstN] = useState<number>(() => last.firstN ?? DEFAULT_MAX_SAMPLES);
    const [evalType, setEvalType] = useState<HeldoutEvalType>(() => last.evalType ?? 'exact_match');
    // Max new tokens the user set in this form, per set key; remembered ones (per set) from earlier runs.
    const [editedTokens, setEditedTokens] = useState<Record<string, number>>({});
    const [rememberedTokens, setRememberedTokens] = useState<Record<string, number>>(() => ({ ...(last.maxNewTokensBySet ?? {}) }));
    const [temperature, setTemperature] = useState<number>(() => last.temperature ?? 0);
    const [modelPath, setModelPath] = useState('');
    const [dialogueJudge, setDialogueJudge] = useState<boolean>(() => last.dialogueJudge ?? false);
    const [nullRunChoice, setNullRunChoice] = useState<NullRunChoice>('');
    // Opt-in early cutoff (gap G7): off on every visit, never remembered.
    const [stopIfUnanswered, setStopIfUnanswered] = useState(false);
    // The LLM judge is never remembered (it can spend money): today's defaults every time.
    const [provider, setProvider] = useState<JudgeProvider>('hf');
    const [judgeModel, setJudgeModel] = useState<string>(() => defaultJudgeModel('hf'));
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
    // "Both" whose base leg failed after the fine-tuned leg was queued: the bodies still to post.
    const [retryBodies, setRetryBodies] = useState<HeldoutEvalRequest[] | null>(null);

    // ── Set facts: does the project have a prepared test split? (null = not known: no set prefill) ──────────
    const [manifestSplits, setManifestSplits] = useState<Record<string, number> | null>(null);
    const [hasTestSplit, setHasTestSplit] = useState<boolean | null>(null);
    useEffect(() => {
        let alive = true;
        api.get<{ splits?: Record<string, unknown> } | null>(`/projects/${projectId}/prepared-manifest`)
            .then((res) => {
                if (!alive) return;
                const data = res?.data;
                // The server answers {} when the project has no prepared manifest (so no test split).
                if (data && typeof data === 'object') {
                    const test = (data.splits && typeof data.splits === 'object') ? data.splits.test : undefined;
                    setHasTestSplit(test != null && !(typeof test === 'number' && test < 1));
                }
                const splits = data?.splits;
                if (!splits || typeof splits !== 'object') return;
                const counts: Record<string, number> = {};
                for (const [key, value] of Object.entries(splits)) {
                    if (typeof value === 'number' && Number.isFinite(value)) counts[key] = value;
                }
                setManifestSplits(counts);
            })
            .catch(() => { /* no manifest: row counts unknown, today's defaults apply */ });
        return () => { alive = false; };
    }, [projectId]);

    // Eval-set prefill (phase 7c; UI-only): only when it is unambiguous — the project's only eval set and no
    // prepared test split. Never over a test split (the built-in held-out split stays the score of record), never
    // over a set you picked (here or last time). There is no "benchmark" mark on an eval set, so no other case.
    const prefillEvalSet = pickedDataset === null && !evalSetsLoading && evalSets.length === 1 && hasTestSplit === false
        ? evalSets[0]
        : null;
    const datasetName = pickedDataset ?? (prefillEvalSet ? `eval_set:${prefillEvalSet.slug}` : 'test');

    // What is shown is what is sent: a remembered eval set that is not (yet) listed reads as the test examples
    // (and Run waits while the sets are still loading).
    const pickedEvalSet = isEvalSetName(datasetName)
        ? evalSets.find((s) => `eval_set:${s.slug}` === datasetName) ?? null
        : null;
    const waitingForEvalSets = isEvalSetName(datasetName) && !pickedEvalSet && evalSetsLoading;
    const shownDataset = isEvalSetName(datasetName) && !pickedEvalSet ? 'test' : datasetName;
    const builtIn = BUILT_IN_SETS.find((s) => s.value === shownDataset) ?? null;
    const version = pickedEvalSet ? (evalSetVersion ?? pickedEvalSet.current_version) : null;

    // ── Set facts: row count (split manifest / eval-set version) and the longest answer ──────────
    const setRows: number | null = pickedEvalSet
        ? (pickedEvalSet.versions.find((v) => v.version === version)?.rows ?? (version === pickedEvalSet.current_version ? pickedEvalSet.rows : null))
        : builtIn?.manifestKey && manifestSplits && typeof manifestSplits[builtIn.manifestKey] === 'number'
            ? manifestSplits[builtIn.manifestKey]
            : null;
    const setKey = pickedEvalSet ? `eval_set:${pickedEvalSet.slug}@v${version}` : shownDataset;

    const [longest, setLongest] = useState<LongestCache>({});
    const longestKnown = hasKey(longest, setKey);
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

    // ── Prompt-format variant support (the server says when the run renders no system message) ──
    const [variantBlockedReason, setVariantBlockedReason] = useState<string | null>(null);
    // Review of 7e (S3): true once the server answered that this run takes prompt-format variants. Until then (or on
    // an error) the "Every prompt format" default does not apply, so Run never queues per-format legs the server
    // would refuse; a pick still does (the server then answers with its reason).
    const [variantsSupported, setVariantsSupported] = useState(false);
    const pickedSetHasVariants = (pickedEvalSet?.variants.length ?? 0) > 0;
    useEffect(() => {
        let active = true;
        setVariantBlockedReason(null);
        setVariantsSupported(false);
        if (!pickedSetHasVariants) return () => { active = false; };
        fetchVariantSupport(projectId, run.id)
            .then((support) => {
                if (!active) return;
                const blocked = support?.supported === false;
                setVariantBlockedReason(blocked ? (support.reason ?? 'Prompt-format variants do not apply to this run.') : null);
                setVariantsSupported(!blocked);
                if (blocked) setEvalSetVariant(null);
            })
            .catch(() => { /* unknown: the server still refuses an unsupported variant with its reason */ });
        return () => { active = false; };
    }, [projectId, run.id, pickedSetHasVariants]);

    // The prompt format shown and sent: a remembered / kept name only when this set has it.
    const variantNames = (pickedEvalSet?.variants ?? []).map((v) => v.name);
    // Phase 7e (UI-only, the server's default is unchanged): with no prompt format picked (here or remembered), a set
    // with 2+ prompt formats starts on "Every prompt format" (one eval per format); a single-format set keeps "as in
    // the rows". A pick (or a remembered one) always wins; the default is never remembered.
    const allFormatsByDefault = evalSetVariant === null && !!pickedEvalSet && variantsSupported && !variantBlockedReason
        && variantNames.length >= 2;
    const wantedVariant = evalSetVariant ?? (allFormatsByDefault ? ALL_PROMPT_FORMATS : '');
    const shownVariant = !pickedEvalSet || variantBlockedReason
        ? ''
        : wantedVariant === ALL_PROMPT_FORMATS
            ? (variantNames.length > 0 ? ALL_PROMPT_FORMATS : '')
            : variantNames.includes(wantedVariant) ? wantedVariant : '';
    const allFormats = shownVariant === ALL_PROMPT_FORMATS;

    const rawPathSet = modelPath.trim() !== '';

    // ── The base-model result the lift would pair with (the lift summary's baseline on this set) ──────
    // Eval sets: the lift on exactly this set + version + prompt format; built-in sets: the built-in lift,
    // whose baseline counts only when it scored this split. Only the run's own base model qualifies.
    const liftKey = pickedEvalSet && version != null
        ? evalSetDatasetName(pickedEvalSet.slug, version, shownVariant && !allFormats ? shownVariant : null)
        : '';
    const baseLookupKey = `${run.id}|${liftKey || `builtin:${shownDataset}`}`;
    // "Every prompt format": the Both default reads the base result of each format (phase 7e, gap G1).
    const formatLiftKeys = allFormats && pickedEvalSet && version != null
        ? variantNames.map((name) => evalSetDatasetName(pickedEvalSet.slug, version, name))
        : [];
    const baseApplies = !isBaseline && baseModel !== null && !waitingForEvalSets;
    const [baseCache, setBaseCache] = useState<BaseCache>({});
    // Review of 7e (S2): lookups started, by cache key. A lookup that lands changes the missing keys and re-runs the
    // effect below; the others keep going (their result is kept) and are not started again. A key stays here once
    // settled (its answer, or 'error', is cached for good, as before), so an effect that runs before that answer is
    // rendered does not start it again.
    const baseRequestedRef = useRef<Set<string>>(new Set());
    const mountedRef = useRef(true);
    useEffect(() => {
        mountedRef.current = true;
        return () => { mountedRef.current = false; };
    }, []);
    // The lift keys to look up ('' = the built-in set): this set + prompt format, and each format under "Every".
    const lookupKeys = [liftKey, ...formatLiftKeys.filter((k) => k !== liftKey)];
    const missingKeys = baseApplies ? lookupKeys.filter((k) => !hasKey(baseCache, `${run.id}|${k || `builtin:${shownDataset}`}`)) : [];
    const missingSignature = `${shownDataset}|${missingKeys.join('\n')}`;
    useEffect(() => {
        if (!baseApplies || missingKeys.length === 0) return;
        const lookup = async (key: string): Promise<BaseResultRef | null> => {
            const summary = key ? await fetchLift(projectId, run.id, key) : await fetchLift(projectId, run.id);
            const base = summary && typeof summary === 'object' ? summary.baseline : null;
            if (!base || typeof base.eval_result_id !== 'number' || typeof base.experiment_id !== 'number') return null;
            if (base.base_model && base.base_model !== baseModel) return null;
            if ((key || shownDataset) !== base.dataset_name) return null;
            const res = await api.get<VerdictResultLike[]>(`/projects/${projectId}/evaluation/results/${base.experiment_id}`);
            const rows = Array.isArray(res?.data) ? res.data : [];
            const settings = resultRunSettings(rows.find((r) => r.id === base.eval_result_id) ?? null);
            const hasSettings = !(settings.rowsRequested == null && settings.rowsAnswered == null && settings.tokens == null);
            return { resultId: base.eval_result_id, runId: base.experiment_id, ...settings, hasSettings };
        };
        for (const key of missingKeys) {
            // The cache key names the run and the set, so a result that lands after the form moved on is still right.
            const cacheKey = `${run.id}|${key || `builtin:${shownDataset}`}`;
            if (baseRequestedRef.current.has(cacheKey)) continue;
            baseRequestedRef.current.add(cacheKey);
            const settle = (entry: BaseResultRef | null | 'error') => {
                if (mountedRef.current) setBaseCache((prev) => ({ ...prev, [cacheKey]: entry }));
            };
            lookup(key).then(settle).catch(() => settle('error'));
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the missing lookup keys
    }, [projectId, missingSignature, baseApplies]);
    const baseKnown = hasKey(baseCache, baseLookupKey);
    const baseEntry = baseApplies ? baseCache[baseLookupKey] : undefined;
    // Under "Every prompt format": every format checked, and none has a base result (null = none stored).
    const formatEntries = formatLiftKeys.map((k) => (baseApplies ? baseCache[`${run.id}|${k}`] : undefined));
    const formatsChecking = allFormats && baseApplies && formatEntries.some((e) => e === undefined);
    const noBaseOnAnyFormat = formatEntries.length > 0 && formatEntries.every((e) => e === null);
    // Both by default only on a checked "no base result" for this set + prompt format (not while checking, not on
    // an error); under "Every prompt format", only when no format has a base result (phase 7e, gap G1: the default
    // used to drop to Fine-tuned there, so the Run queued no base legs).
    const bothByDefault = modelChoice === null && !rawPathSet && baseApplies
        && (allFormats ? noBaseOnAnyFormat : baseEntry === null);

    // ── Which model ───────────────────────────────────────────────────────────────
    // The first-eval default (phase 7c, review of 7b; UI-only, server defaults unchanged): with no choice picked
    // (here or remembered), Both when the run records a base model, is not a baseline run, and no base-model
    // result is stored yet on this set + prompt format (a checked "none", not an unknown); else Fine-tuned. While
    // the lookup runs (Run waits) the form shows Fine-tuned.
    const effectiveChoice: ModelChoice = isBaseline || !baseModel
        ? 'finetuned'
        : modelChoice ?? (bothByDefault ? 'both' : 'finetuned');
    const paths = modelPathsFor(effectiveChoice, baseModel, modelPath);
    const scoresOwnWeightsOnly = !rawPathSet && effectiveChoice === 'finetuned' && !allFormats;
    // "Both" on a set with variants: the base leg may run on another prompt format (cross-prompt).
    const canCrossPrompt = !rawPathSet && effectiveChoice === 'both' && pickedSetHasVariants && !variantBlockedReason && !allFormats;
    const baseVariantValid = baseVariant === ''
        || (baseVariant === ROWS_AS_WRITTEN && shownVariant !== '')
        || (variantNames.includes(baseVariant) && baseVariant !== shownVariant);
    const shownBaseVariant = canCrossPrompt && baseVariantValid ? baseVariant : '';
    const legs = evalLegs(paths, shownVariant, variantNames, shownBaseVariant);
    function legText(leg: { modelPath: string; variant: string } | undefined): string {
        if (!leg) return 'eval';
        const who = leg.modelPath === '' ? 'fine-tuned' : leg.modelPath === baseModel ? 'base model' : leg.modelPath;
        return pickedEvalSet ? `${who} on ${leg.variant || 'the rows as written'}` : who;
    }

    const baseRef: BaseResultRef | null = baseEntry && baseEntry !== 'error' && baseEntry.hasSettings ? baseEntry : null;
    // Run waits for the check that decides the default (Fine-tuned: the settings to match; Every prompt format with
    // no model picked: Both or Fine-tuned).
    const checkingBase = baseApplies && ((!baseKnown && scoresOwnWeightsOnly)
        || (formatsChecking && modelChoice === null && !rawPathSet));
    // Match the base result's settings only for "Fine-tuned" (Both scores both sides with the same settings).
    const matchBase = scoresOwnWeightsOnly && baseRef !== null;
    const baseRows = baseRef ? (effectiveRows(baseRef.rowsRequested, setRows) ?? baseRef.rowsAnswered) : null;

    // ── Rows ──────────────────────────────────────────────────────────────────────
    const canAll = setRows != null && setRows >= 1 && setRows <= MAX_SAMPLES_CEILING;
    const matchedRowsMode: 'all' | 'first' | null = matchBase && baseRows != null
        ? (canAll && setRows != null && baseRows >= setRows ? 'all' : 'first')
        : null;
    const rowsSource: 'you' | 'base' | 'last' | 'prefill' | 'default' = rowsChoice
        ? 'you'
        : matchedRowsMode ? 'base' : last.rows ? 'last' : setRows != null ? 'prefill' : 'default';
    const wantedRows = rowsChoice ?? matchedRowsMode ?? last.rows ?? (setRows != null && setRows <= ALL_ROWS_PREFILL_MAX ? 'all' : 'first');
    const rowsMode: 'all' | 'first' = wantedRows === 'all' && canAll ? 'all' : 'first';
    const shownFirstN = rowsSource === 'base' && matchedRowsMode === 'first' && baseRows != null ? baseRows : firstN;
    const maxSamples = rowsMode === 'all' && setRows != null ? setRows : shownFirstN;
    const sentRows = effectiveRows(Math.max(1, maxSamples), setRows);

    // ── Scored as (a scoring mode decides the score) ───────────────────────────────────────
    const modeResult = results.find((r) => {
        const mode = scoringModeOf(r);
        return !!mode && TASK_SCORING_MODES.includes(mode);
    }) ?? null;
    const scoredAs: { mode: string; from: string } | null = modeResult
        ? { mode: scoringModeOf(modeResult) as string, from: `as result #${modeResult.id} on this run was` }
        : null;

    // ── Max new tokens ────────────────────────────────────────────────────────────
    const longestChars = longest[setKey];
    const suggestion = typeof longestChars === 'number' ? suggestMaxNewTokens(longestChars) : null;
    const edited = hasKey(editedTokens, setKey) ? editedTokens[setKey] : null;
    const remembered = hasKey(rememberedTokens, setKey) ? rememberedTokens[setKey] : null;
    const matchedTokens = matchBase && baseRef?.tokens != null ? baseRef.tokens : null;
    // No answer field to size from while a scoring mode decides the score: a structured answer cut short fails
    // silently, so start higher and say so (phase 7b, gap N4).
    const unsizedStructured = suggestion == null && longestKnown && longestChars === null && scoredAs !== null;
    const tokens = edited ?? matchedTokens ?? remembered ?? suggestion
        ?? (unsizedStructured ? STRUCTURED_UNSIZED_TOKENS : DEFAULT_MAX_NEW_TOKENS);
    const tokensSource = edited != null
        ? 'set under Advanced'
        : matchedTokens != null
            ? `matching the base result #${baseRef?.resultId}`
            : remembered != null
                ? 'last used for this set'
                : suggestion != null
                    ? `suggested from the longest answer in this set, ${longestChars} characters`
                    : setRows != null && setRows > ALL_ROWS_PREFILL_MAX
                        ? `default; no suggestion for a set over ${ALL_ROWS_PREFILL_MAX.toLocaleString()} rows`
                        : longestChars === 'error'
                            ? 'default; the set rows could not be read'
                            : unsizedStructured
                                ? `could not size: no answer field found in the rows; ${STRUCTURED_UNSIZED_TOKENS} for ${scoringModeLabel(scoredAs?.mode ?? '')} scoring, check max new tokens`
                                : longestKnown
                                    ? 'default; no answer field found in the rows'
                                    : 'default';
    // "use suggestion": back from a value you set (now or last time) for this set.
    const canUseSuggestion = edited != null || (matchedTokens == null && remembered != null);
    const useSuggestion = () => {
        setEditedTokens((prev) => { const next = { ...prev }; delete next[setKey]; return next; });
        setRememberedTokens((prev) => { const next = { ...prev }; delete next[setKey]; return next; });
    };

    // ── Comparability with the stored base result ("Fine-tuned" only) ─────────────────────────
    const rowsDiffer = baseRows != null && sentRows != null && baseRows !== sentRows;
    const tokensDiffer = baseRef?.tokens != null && baseRef.tokens !== tokens;
    const baseMismatch = scoresOwnWeightsOnly && baseRef !== null && (rowsDiffer || tokensDiffer);
    const baseShownRows = baseRef ? (baseRef.rowsAnswered ?? baseRows) : null;
    const matchTheBase = () => {
        setRowsChoice(null);
        setEditedTokens((prev) => { const next = { ...prev }; delete next[setKey]; return next; });
    };

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

    const postBodies = async (bodies: HeldoutEvalRequest[]): Promise<{ ids: Array<number | undefined>; failed: unknown | null }> => {
        const ids: Array<number | undefined> = [];
        for (const body of bodies) {
            try {
                const res = await api.post<{ id: number }>(`/projects/${projectId}/evaluation/run-heldout?async_job=true`, body);
                ids.push(res.data?.id);
            } catch (err) {
                return { ids, failed: err };
            }
        }
        return { ids, failed: null };
    };

    const submit = async () => {
        setIsSubmitting(true);
        setError(null);
        setRetryBodies(null);
        const state = {
            experimentId: run.id,
            datasetName: shownDataset,
            pickedEvalSet,
            evalSetVersion,
            evalSetVariant: shownVariant,
            variantBlockedReason,
            evalType,
            maxSamples,
            maxNewTokens: tokens,
            temperature,
            modelPath,
            dialogueJudge,
            nullRunChoice,
            stopIfUnanswered,
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
        // Each body is the long form's body for the same inputs with Model Path = the leg's path and Prompt format =
        // the leg's variant.
        const bodies = legs.map((leg) => buildHeldoutPayload({ ...state, modelPath: leg.modelPath, evalSetVariant: leg.variant }));
        const { ids: jobIds, failed } = await postBodies(bodies);
        const remember = () => writeLastUsed(projectId, {
            datasetName: shownDataset,
            // The prompt format only when picked (here or last time): the "Every prompt format" default is not a choice.
            ...(evalSetVariant === null ? {} : { variant: pickedEvalSet ? shownVariant : evalSetVariant }),
            baseVariant: pickedEvalSet ? shownBaseVariant : baseVariant,
            // Base-only is almost always a one-off (review S5): it is never remembered, and the earlier remembered
            // choice (Fine-tuned or Both) stays.
            // A default (no pick) is not a choice: it is never remembered, so the next eval decides afresh.
            ...(modelChoice !== null && modelChoice !== 'base' ? { model: modelChoice } : last.model ? { model: last.model } : {}),
            rows: rowsMode,
            firstN,
            ...(evalType !== 'llm_judge' ? { evalType } : {}),
            maxNewTokensBySet: { ...rememberedTokens, ...editedTokens },
            temperature,
            dialogueJudge,
        });
        if (failed) {
            if (jobIds.length > 0) {
                // Something was queued: remember the choices (the next visit starts from them, gap N2).
                remember();
                const queued = legs.slice(0, jobIds.length).map(legText).join(', ');
                setError(`Queued ${queued}${jobIds[0] ? ` (job ${jobIds.filter(Boolean).map((id) => `#${id}`).join(', ')})` : ''}; the next eval (${legText(legs[jobIds.length])}) failed: ${errorText(failed)}`);
                setRetryBodies(bodies.slice(jobIds.length));
                void useJobsStore.getState().refreshAfterLocalChange();
            } else {
                setError(errorText(failed));
            }
            setIsSubmitting(false);
            return;
        }
        remember();
        const ids = jobIds.filter((id): id is number => typeof id === 'number');
        toast.info(
            legs.length > 1
                ? `${legs.length} evals queued, run one after the other: ${legs.map(legText).join(', ')}${ids.length ? ` (jobs ${ids.map((id) => `#${id}`).join(', ')})` : ''} — bell will notify when ready`
                : ids.length
                    ? `Eval queued — bell will notify when ready (job #${ids[0]})`
                    : 'Eval queued — bell will notify when ready',
            4000,
        );
        void useJobsStore.getState().refreshAfterLocalChange();
        setIsSubmitting(false);
        const jobsText = ids.length ? ` (job${ids.length === 1 ? '' : 's'} ${ids.map((id) => `#${id}`).join(', ')})` : '';
        onQueued?.(`Queued ${legs.length} eval${legs.length === 1 ? '' : 's'}: ${legs.map(legText).join(', ')}${jobsText}. The bell says when ${legs.length === 1 ? 'it is' : 'they are'} ready.`);
    };

    const retryBase = async () => {
        if (!retryBodies?.length) return;
        setIsSubmitting(true);
        const { ids, failed } = await postBodies(retryBodies);
        if (failed) {
            if (ids.length > 0) setRetryBodies(retryBodies.slice(ids.length));
            setError(`The eval failed again: ${errorText(failed)}`);
            setIsSubmitting(false);
            return;
        }
        setError(null);
        setRetryBodies(null);
        const id = ids.find((x): x is number => typeof x === 'number');
        toast.info(`Eval queued — bell will notify when ready${id ? ` (job #${id})` : ''}`, 4000);
        void useJobsStore.getState().refreshAfterLocalChange();
        setIsSubmitting(false);
        const known = ids.filter((x): x is number => typeof x === 'number');
        onQueued?.(`Queued ${ids.length} more eval${ids.length === 1 ? '' : 's'}${known.length ? ` (job${known.length === 1 ? '' : 's'} ${known.map((x) => `#${x}`).join(', ')})` : ''}. The bell says when ready.`);
    };

    const setLabel = pickedEvalSet
        ? `${pickedEvalSet.name} v${version}${allFormats ? ' (every prompt format)' : shownVariant ? ` (${shownVariant})` : ''}`
        : builtIn?.label ?? shownDataset;
    const rowsProvenance = setRows == null
        ? `${setLabel}: row count unknown`
        : `${setLabel}: ${setRows.toLocaleString()} rows${pickedEvalSet ? '' : ' (prepared split)'}`;
    const advancedCount = 8 +(evalType === 'llm_judge' ? 2 : 0) + (evalType === 'llm_judge' && provider === 'local_serve' ? 1 : 0);
    const metricLabel = evalType === 'exact_match' ? 'Exact match' : evalType === 'f1' ? 'F1' : 'LLM judge';
    const judgeText = evalType === 'llm_judge'
        ? `judge ${PROVIDER_LABEL[provider]} ${judgeModel.trim() || defaultJudgeModel(provider)}${provider === 'local_serve' ? ` (serve run ${selectedLocalServeRunId || 'auto'})` : ''}`
        : '';
    const metricPart = scoredAs
        ? (judgeText ? `${judgeText} · ` : '')
        : `metric ${metricLabel}${judgeText ? ` (${judgeText})` : ''} · `;
    const runDisabled = isSubmitting || waitingForEvalSets || checkingBase;
    const runLabel = isSubmitting
        ? 'Queuing…'
        : waitingForEvalSets
            ? 'Loading eval sets…'
            : checkingBase
                ? 'Checking for a base result…'
                : runLabelFor(legs, baseModel);

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
                        onChange={(e) => { setDatasetName(e.target.value); setEvalSetVersion(null); }}
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
                            value={shownVariant}
                            onChange={(e) => setEvalSetVariant(e.target.value)}
                            disabled={variantBlockedReason !== null}
                            data-testid="eval-set-variant-select"
                        >
                            <option value="">Prompt format: as in the rows</option>
                            {pickedEvalSet.variants.map((v) => (
                                <option key={v.name} value={v.name}>{v.name} · {v.label}</option>
                            ))}
                            <option value={ALL_PROMPT_FORMATS}>
                                Every prompt format ({pickedEvalSet.variants.length}): one eval per format
                            </option>
                        </select>
                    )}
                    {onOpenTestSets && (
                        <button type="button" className="btn btn-link" onClick={onOpenTestSets} data-testid="eval-run-add-eval-set">
                            Add an eval set
                        </button>
                    )}
                    {prefillEvalSet && (
                        <p className="u-meta u-m0 eval-run-form__full" data-testid="eval-run-set-prefill">
                            Prefilled: {prefillEvalSet.name} is the only eval set in this project, and there is no prepared test split.
                        </p>
                    )}
                    {waitingForEvalSets && (
                        <p className="u-meta u-m0 eval-run-form__full" data-testid="eval-run-sets-loading">
                            Loading eval sets (your last choice was {datasetName.slice('eval_set:'.length)})…
                        </p>
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
                            <small>two evals, one after the other</small>
                        </label>
                        {bothByDefault && effectiveChoice === 'both' && (
                            <span className="u-meta eval-run-form__full" data-testid="eval-run-model-default-both">
                                {allFormats ? (
                                    <>
                                        Both is the default for a first eval: no base-model score on any prompt format of this set yet,
                                        so this runs {legs.length} evals, one after the other (per prompt format: fine-tuned, then the base model). Pick Fine-tuned for {variantNames.length} evals.
                                    </>
                                ) : (
                                    <>
                                        Both is the default for a first eval: no base-model score on {pickedEvalSet ? 'this set and prompt format' : 'this set'} yet,
                                        so this runs two evals, one after the other (fine-tuned, then the base model). Pick Fine-tuned for one eval.
                                    </>
                                )}
                            </span>
                        )}
                        {!baseModel && (
                            <span className="u-meta eval-run-form__full">The run records no base model; to score another model use Advanced › Model path.</span>
                        )}
                        {rawPathSet && (
                            <span className="u-meta eval-run-form__full" data-testid="eval-run-model-raw-path">
                                Advanced › Model path is set: one eval answered by {modelPath.trim()}.
                            </span>
                        )}
                        {canCrossPrompt && (
                            <span className="eval-run-form__full">
                                <select
                                    className="input u-w-auto"
                                    aria-label="Base model on"
                                    value={shownBaseVariant}
                                    onChange={(e) => setBaseVariant(e.target.value)}
                                    data-testid="eval-run-base-variant"
                                >
                                    <option value="">Base model on the same prompt format</option>
                                    {shownVariant !== '' && (
                                        <option value={ROWS_AS_WRITTEN}>Base model on the rows as written (cross-prompt: the prompt differs too)</option>
                                    )}
                                    {variantNames.filter((n) => n !== shownVariant).map((n) => (
                                        <option key={n} value={n}>Base model on {n} (cross-prompt: the prompt differs too)</option>
                                    ))}
                                </select>
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
                        <input type="radio" name="eval-run-rows" checked={rowsMode === 'first'} onChange={() => { setFirstN(shownFirstN); setRowsChoice('first'); }}
                            data-testid="eval-run-rows-first" />
                        First
                    </label>
                    <input
                        className="input eval-run-form__number"
                        type="number"
                        min={1}
                        max={MAX_SAMPLES_CEILING}
                        aria-label="First N rows"
                        value={shownFirstN}
                        onChange={(e) => { setFirstN(Number(e.target.value)); setRowsChoice('first'); }}
                        data-testid="eval-run-rows-first-n"
                    />
                    <span className="u-meta eval-run-form__full" data-testid="eval-run-rows-provenance">
                        {rowsProvenance}
                        {allFormatsByDefault && allFormats ? ` · Every prompt format is the default for a set with ${variantNames.length} prompt formats (${variantNames.join(', ')}); pick one for fewer evals` : ''}
                        {rowsSource === 'base' ? ` · matching the base result #${baseRef?.resultId}` : ''}
                        {rowsSource === 'prefill' && rowsMode === 'all' ? ` · all rows prefilled (sets of up to ${ALL_ROWS_PREFILL_MAX.toLocaleString()} rows)` : ''}
                        {rowsSource === 'last' ? ' · your last choice' : ''}
                        {rowsMode === 'first' && setRows != null && shownFirstN < setRows ? ` · partial: ${Math.max(1, shownFirstN)} of ${setRows.toLocaleString()} rows` : ''}
                    </span>
                </div>
            </div>

            {matchBase && !baseMismatch && baseRef && (
                <p className="u-meta eval-run-form__sends" data-testid="eval-run-base-match">
                    Matching the base result #{baseRef.resultId} (run #{baseRef.runId}): {baseShownRows ?? '?'} rows, {baseRef.tokens ?? tokens} max new
                    tokens, so base vs fine-tuned compares like for like.
                </p>
            )}
            {baseMismatch && baseRef && (
                <div className="eval-run-form__warn" role="status" data-testid="eval-run-base-mismatch">
                    The base scored {baseShownRows ?? '?'} rows at {baseRef.tokens ?? '?'} max new tokens (result #{baseRef.resultId}, run
                    #{baseRef.runId}); this eval sends {sentRows ?? maxSamples} rows at {tokens}. Base vs fine-tuned would compare different
                    settings: pick Both, or match them.{' '}
                    <button type="button" className="btn btn-link" onClick={matchTheBase} data-testid="eval-run-base-match-button">
                        Match the base
                    </button>
                </div>
            )}

            {scoredAs && (
                <p className="u-meta eval-run-form__scored-as" data-testid="eval-run-scored-as">
                    Scored as: {scoringModeLabel(scoredAs.mode)} ({scoredAs.from}; the scoring mode decides the score, not the metric
                    under Advanced).
                </p>
            )}
            <p className="u-meta eval-run-form__sends" data-testid="eval-run-sends">
                Also sent: {metricPart}max {tokens} new tokens ({tokensSource}
                {canUseSuggestion && (
                    <>
                        {' · '}
                        <button type="button" className="btn btn-link" onClick={useSuggestion} data-testid="eval-run-tokens-use-suggestion">
                            {suggestion != null ? `use suggestion (${suggestion})` : `use the default (${DEFAULT_MAX_NEW_TOKENS})`}
                        </button>
                    </>
                )}
                ) · temperature {temperature}
                {dialogueJudge ? ' · every assistant turn judged' : ''}
                {nullRunChoice ? ` · null counts as absent: ${nullRunChoice}` : ''}
                {stopIfUnanswered ? ` · stops a prompt format with no parsable answer after ${STOP_IF_UNANSWERED_ROWS} rows` : ''} — change under Advanced.
            </p>
            {scoresOwnWeightsOnly && baseEntry === 'error' && (
                <p className="u-meta eval-run-form__sends" data-testid="eval-run-base-unknown">
                    Could not check for a stored base-model result on this set; pick Both to score both sides with the same settings.
                </p>
            )}

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
                            value={tokens} onChange={(e) => { const n = Number(e.target.value); setEditedTokens((prev) => ({ ...prev, [setKey]: n })); }} data-testid="eval-run-max-new-tokens" />
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

                <NullAsAbsentSetting projectId={projectId} runChoice={nullRunChoice} onRunChoice={setNullRunChoice} showProjectToggle={false} />

                <div className="form-group">
                    <label className="eval-panel__toggle">
                        <input type="checkbox" checked={stopIfUnanswered} onChange={(e) => setStopIfUnanswered(e.target.checked)}
                            data-testid="eval-run-stop-unanswered" />
                        Stop a prompt format the model never answers in (after {STOP_IF_UNANSWERED_ROWS} rows with no parsable answer)
                    </label>
                    <p className="u-hint u-mt0" data-testid="eval-run-stop-unanswered-hint">
                        {scoredAs
                            ? `Applies here: this run is scored as ${scoringModeLabel(scoredAs.mode)}.`
                            : 'JSON / tool-call scoring only; ignored for other scoring.'}
                        {' '}If none of the first {STOP_IF_UNANSWERED_ROWS} answers parses, the eval stops there and is stored as a
                        partial result that says why. If any parses, every row runs as usual.
                    </p>
                </div>

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
                <div className="eval-panel__error-card u-mt-md" role="alert" data-testid="eval-run-error">
                    {error}
                    {retryBodies && retryBodies.length > 0 && (
                        <>
                            {' '}
                            <button type="button" className="btn btn-secondary btn-sm" onClick={() => void retryBase()} disabled={isSubmitting}
                                data-testid="eval-run-retry-base">
                                {retryBodies.every((b) => (b.model_path ?? '') === baseModel) && retryBodies.length === 1 ? 'Retry base only' : 'Retry the rest'}
                            </button>
                        </>
                    )}
                </div>
            )}

            <div className="u-mt-lg">
                <button type="button" className="btn btn-primary" onClick={() => void submit()} disabled={runDisabled} data-testid="eval-run-submit">
                    {runLabel}
                </button>
            </div>
        </div>
    );
}
