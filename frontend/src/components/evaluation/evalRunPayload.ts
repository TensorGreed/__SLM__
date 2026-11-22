/**
 * The "Score this run" form's request body and its UI-only prefills (UI simplification pass, phase 2).
 *
 * ``buildHeldoutPayload`` is the body the eval form has always posted to ``run-heldout`` (moved here
 * unchanged from ``EvalPanel.runHeldoutEval``): the short form maps its choices onto the same fields,
 * so a choice sends exactly what the long form sent for the same inputs. "Base model" = the long
 * form's Model Path set to the run's own base model (the server files that result under the
 * baseline run); "Both" = the fine-tuned body, then the base body.
 *
 * Prefills (decision 3, UI only; server defaults unchanged): "All N rows" when the set has ≤ 2,000
 * rows, max new tokens suggested from the set's longest answer (never under 128, so a preamble or a thinking
 * block is not cut off), last-used values per project. With "Fine-tuned" picked and a base-model result already
 * stored for the set (the lift's baseline), rows and max new tokens match that result instead
 * (``baseResultSettings``), so base vs fine-tuned compares like for like.
 */

import { evalSetDatasetName, type EvalSet } from '../../api/evalSets';
import { nullAsAbsentPayload, type NullRunChoice } from './NullAsAbsentSetting';

export type JudgeProvider = 'hf' | 'ollama' | 'openai' | 'local_serve';
export type HeldoutEvalType = 'exact_match' | 'f1' | 'llm_judge';

export interface HeldoutEvalRequest {
    experiment_id: number;
    dataset_name: string;
    eval_type: HeldoutEvalType;
    max_samples: number;
    max_new_tokens: number;
    temperature: number;
    model_path?: string;
    judge_model?: string;
    dialogue_judge?: boolean;
    prompt_variant?: string;
    null_as_absent?: boolean;
}

/** Every input the long form read when it built the body (names as in the old ``EvalPanel`` state). */
export interface HeldoutFormState {
    experimentId: number;
    datasetName: string;
    pickedEvalSet: Pick<EvalSet, 'slug' | 'current_version'> | null;
    evalSetVersion: number | null;
    evalSetVariant: string;
    variantBlockedReason: string | null;
    evalType: HeldoutEvalType;
    maxSamples: number;
    maxNewTokens: number;
    temperature: number;
    modelPath: string;
    dialogueJudge: boolean;
    nullRunChoice: NullRunChoice;
    provider: JudgeProvider;
    judgeModel: string;
    selectedLocalServeRunId: string;
    localServeAutoStart: boolean;
    localServeAutoStop: boolean;
    localServeSource: 'export' | 'registry';
    localServeExportId: string;
    localServeModelId: string;
    localServeTemplateId: string;
    localServeHost: string;
    localServePort: string;
}

export function buildHeldoutPayload(s: HeldoutFormState): HeldoutEvalRequest {
    const payload: HeldoutEvalRequest = {
        experiment_id: s.experimentId,
        dataset_name: s.pickedEvalSet
            ? evalSetDatasetName(s.pickedEvalSet.slug, s.evalSetVersion ?? s.pickedEvalSet.current_version)
            : s.datasetName,
        eval_type: s.evalType,
        max_samples: Math.max(1, s.maxSamples),
        max_new_tokens: Math.max(1, s.maxNewTokens),
        temperature: Math.max(0, s.temperature),
    };

    if (s.modelPath.trim()) {
        payload.model_path = s.modelPath.trim();
    }
    if (s.pickedEvalSet && s.evalSetVariant && !s.variantBlockedReason) {
        payload.prompt_variant = s.evalSetVariant;
    }
    if (s.dialogueJudge) {
        payload.dialogue_judge = true;
    }
    const nullPayload = nullAsAbsentPayload(s.nullRunChoice);
    if (nullPayload !== undefined) {
        payload.null_as_absent = nullPayload;
    }
    if (s.evalType === 'llm_judge') {
        if (s.provider === 'local_serve') {
            const runToken = s.selectedLocalServeRunId.trim() || 'auto';
            const params = new URLSearchParams();
            const judgeModelToken = s.judgeModel.trim();
            if (judgeModelToken) {
                params.set('model', judgeModelToken);
            }
            if (s.localServeAutoStart) {
                params.set('auto_start', '1');
                params.set('source', s.localServeSource);
                const exportId = Number.parseInt(s.localServeExportId, 10);
                const modelId = Number.parseInt(s.localServeModelId, 10);
                if (s.localServeSource === 'export' && Number.isFinite(exportId) && exportId > 0) {
                    params.set('export_id', String(exportId));
                }
                if (s.localServeSource === 'registry' && Number.isFinite(modelId) && modelId > 0) {
                    params.set('model_id', String(modelId));
                }
                if (s.localServeTemplateId.trim()) {
                    params.set('template_id', s.localServeTemplateId.trim());
                }
                if (s.localServeHost.trim()) {
                    params.set('host', s.localServeHost.trim());
                }
                const port = Number.parseInt(s.localServePort, 10);
                if (Number.isFinite(port) && port > 0) {
                    params.set('port', String(port));
                }
                if (s.localServeAutoStop) {
                    params.set('auto_stop', '1');
                }
            }
            const suffix = params.toString();
            payload.judge_model = `local_serve:${runToken}${suffix ? `?${suffix}` : ''}`;
        } else {
            payload.judge_model = s.judgeModel;
        }
    }
    return payload;
}

// ── Which model answers ─────────────────────────────────────────────────────────

/** "finetuned" = the run's own weights; "base" = its base model; "both" = two evals, base vs fine-tuned. */
export type ModelChoice = 'finetuned' | 'base' | 'both';

/** The model path each request carries for a choice ('' = the run's own weights), in posting order. A raw
 *  Model path typed under Advanced wins (one request, exactly as the long form sent it). */
export function modelPathsFor(choice: ModelChoice, baseModel: string | null | undefined, rawModelPath: string): string[] {
    if (rawModelPath.trim()) return [rawModelPath];
    const base = (baseModel ?? '').trim();
    if (choice === 'base' && base) return [base];
    if (choice === 'both' && base) return ['', base];
    return [''];
}

// ── Prompt formats ───────────────────────────────────────────────────────────

/** The prompt-format select's "every prompt format of this set" choice: one Run queues one eval per variant. */
export const ALL_PROMPT_FORMATS = '*';
/** The "Base model on" select's "the rows as written" choice (no variant) while the fine-tuned leg uses one. */
export const ROWS_AS_WRITTEN = '__rows__';

export interface EvalLeg {
    /** '' = the run's own weights, else the model path (the base model). */
    modelPath: string;
    /** '' = the rows as written. */
    variant: string;
}

/**
 * The evals one Run queues, in posting order. ``paths`` is ``modelPathsFor``'s answer ('' = fine-tuned). ``variant``
 * is the picked prompt format ('' / a name / ``ALL_PROMPT_FORMATS``); ``baseVariant`` moves the base leg of "Both"
 * to another prompt format ('' = the same, ``ROWS_AS_WRITTEN`` = none): a cross-prompt pair.
 */
export function evalLegs(paths: string[], variant: string, variants: string[], baseVariant = ''): EvalLeg[] {
    if (variant === ALL_PROMPT_FORMATS && variants.length > 0) {
        return variants.flatMap((v) => paths.map((modelPath) => ({ modelPath, variant: v })));
    }
    const single = variant === ALL_PROMPT_FORMATS ? '' : variant;
    return paths.map((modelPath, i) => {
        const isBaseLegOfBoth = paths.length > 1 && i > 0 && modelPath !== '';
        if (!isBaseLegOfBoth || !baseVariant) return { modelPath, variant: single };
        return { modelPath, variant: baseVariant === ROWS_AS_WRITTEN ? '' : baseVariant };
    });
}

// ── Prefills ─────────────────────────────────────────────────────────────────

/** Sets at or under this many rows prefill "All rows" (decision 3). */
export const ALL_ROWS_PREFILL_MAX = 2000;
/** The server's ``max_samples`` ceiling (``HeldoutEvalRequest`` in ``api/evaluation.py``). */
export const MAX_SAMPLES_CEILING = 5000;
export const DEFAULT_MAX_SAMPLES = 100;
export const DEFAULT_MAX_NEW_TOKENS = 128;
/** The form's max new tokens input bound. */
export const MAX_NEW_TOKENS_CEILING = 1024;
/** A suggestion never goes under this: room for a preamble or a thinking block before the answer. */
export const SUGGESTED_TOKENS_FLOOR = 128;
/** When a set's rows name no answer the form can size from and a scoring mode (tool call, JSON fields, …) decides
 *  the score, max new tokens starts here instead of the 128 default: a structured answer cut short fails silently. */
export const STRUCTURED_UNSIZED_TOKENS = 256;

function asText(value: unknown): string {
    if (value == null) return '';
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

function contentText(content: unknown): string {
    if (Array.isArray(content)) {
        return content
            .map((part) => (part && typeof part === 'object' ? asText((part as { text?: unknown }).text) : asText(part)))
            .join('');
    }
    return asText(content);
}

const ANSWER_KEYS = [
    'reference', 'target_text', 'expected', 'output', 'answer', 'structured_output', 'completion', 'response', 'target', 'label',
    'json',
] as const;

/** A tool-call row's expected call as the model writes it (``{"tool": …, "args": …}``, the tool-call scoring mode's
 *  target), from ``tool_name`` + ``tool_args`` (or ``arguments``). '' when the row names no tool. */
function toolCallAnswerText(record: Record<string, unknown>): string {
    const name = record.tool_name;
    if (typeof name !== 'string' || !name.trim()) return '';
    const args = record.tool_args ?? record.arguments ?? {};
    return asText({ tool: name, args });
}

/** The reference answer of one row: the final assistant turn of a chat row (its text plus any tool calls), else a
 *  tool-call row's expected call (``tool_name`` + ``tool_args``), else the first answer-like field. '' when the row
 *  has none. */
export function rowAnswerText(row: unknown): string {
    if (!row || typeof row !== 'object') return '';
    const record = row as Record<string, unknown>;
    const messages = Array.isArray(record.messages) ? record.messages : Array.isArray(record.conversations) ? record.conversations : null;
    if (messages) {
        for (let i = messages.length - 1; i >= 0; i -= 1) {
            const m = messages[i] as Record<string, unknown> | null;
            if (!m || typeof m !== 'object') continue;
            const role = String(m.role ?? m.from ?? '').toLowerCase();
            if (role !== 'assistant' && role !== 'gpt' && role !== 'model') continue;
            const text = contentText(m.content ?? m.value ?? m.text);
            const calls = m.tool_calls != null ? asText(m.tool_calls) : '';
            return `${text}${calls}`;
        }
    }
    const toolCall = toolCallAnswerText(record);
    if (toolCall) return toolCall;
    for (const key of ANSWER_KEYS) {
        const text = asText(record[key]);
        if (text.trim()) return text;
    }
    return '';
}

/** The longest reference answer (in characters) over the rows, or null when no row has one. */
export function longestAnswerChars(rows: unknown[]): number | null {
    let longest: number | null = null;
    for (const row of rows) {
        const text = rowAnswerText(row);
        if (!text) continue;
        longest = Math.max(longest ?? 0, text.length);
    }
    return longest;
}

/**
 * Max new tokens suggested for answers up to ``chars`` characters: about one token per 3 characters (generous
 * for English and JSON), +50% headroom, rounded up to a multiple of 16, between 128 (``SUGGESTED_TOKENS_FLOOR``:
 * a short answer key must not truncate a preamble or a thinking block) and the form's 1024.
 */
export function suggestMaxNewTokens(chars: number): number {
    const tokens = Math.ceil(Math.max(0, chars) / 3) * 1.5;
    const rounded = Math.ceil(tokens / 16) * 16;
    return Math.min(MAX_NEW_TOKENS_CEILING, Math.max(SUGGESTED_TOKENS_FLOOR, rounded));
}

// ── A stored result's run settings (rows, max new tokens) ─────────────────────────────

export interface ResultRunSettings {
    /** ``details.dataset.max_samples``: the row limit the eval asked for. */
    rowsRequested: number | null;
    /** ``details.inference.samples`` (``metrics.evaluated_samples`` on older results): rows answered. */
    rowsAnswered: number | null;
    /** ``details.inference.max_new_tokens_requested`` (else ``max_new_tokens``): the max new tokens asked for. */
    tokens: number | null;
}

function finiteNumber(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** What a stored eval result says it ran with (read-only; null when not stored). */
export function resultRunSettings(result: { metrics?: Record<string, unknown> | null; details?: Record<string, unknown> | null } | null | undefined): ResultRunSettings {
    const details = (result?.details ?? null) as Record<string, unknown> | null;
    const dataset = (details?.dataset ?? null) as Record<string, unknown> | null;
    const inference = (details?.inference ?? null) as Record<string, unknown> | null;
    return {
        rowsRequested: finiteNumber(dataset?.max_samples),
        rowsAnswered: finiteNumber(inference?.samples) ?? finiteNumber(result?.metrics?.evaluated_samples),
        tokens: finiteNumber(inference?.max_new_tokens_requested) ?? finiteNumber(inference?.max_new_tokens),
    };
}

/** Rows an eval asking for ``requested`` rows scores on a set of ``setRows`` rows (null = unknown). */
export function effectiveRows(requested: number | null, setRows: number | null): number | null {
    if (requested == null) return null;
    return setRows != null ? Math.min(requested, setRows) : requested;
}

/** Parse a JSONL body (a prepared split file); unparseable lines are skipped. */
export function parseJsonl(text: unknown): unknown[] {
    if (typeof text !== 'string') return [];
    const rows: unknown[] = [];
    for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
            rows.push(JSON.parse(trimmed));
        } catch {
            // skip
        }
    }
    return rows;
}

// ── Last-used values (per project, localStorage; every access in try/catch) ─────────────────

/**
 * Remembered per project. Never the LLM judge (metric, provider, model): a judge can spend money, so it is
 * picked again on purpose each time.
 */
export interface LastUsedScoreRun {
    datasetName?: string;
    /** The prompt-format variant name ('' = as in the rows, ``ALL_PROMPT_FORMATS`` = every one); applied only when the
     *  picked set has it. */
    variant?: string;
    /** "Both": the base leg's prompt format when it differs from the fine-tuned one ('' = the same). */
    baseVariant?: string;
    model?: ModelChoice;
    rows?: 'all' | 'first';
    firstN?: number;
    /** Exact match or F1 only (an LLM judge is never remembered). */
    evalType?: Exclude<HeldoutEvalType, 'llm_judge'>;
    /** Max new tokens the user set, per set key (else that set's suggestion applies). */
    maxNewTokensBySet?: Record<string, number>;
    temperature?: number;
    dialogueJudge?: boolean;
}

function lastUsedKey(projectId: number): string {
    return `brewslm.scoreRun.${projectId}`;
}

export function readLastUsed(projectId: number): LastUsedScoreRun {
    try {
        const raw = window.localStorage.getItem(lastUsedKey(projectId));
        const parsed = raw ? JSON.parse(raw) : null;
        if (!parsed || typeof parsed !== 'object') return {};
        const value = { ...(parsed as Record<string, unknown>) };
        // Values an earlier build stored that are no longer remembered: the LLM judge, one tokens value for every set.
        delete value.provider;
        delete value.judgeModel;
        delete value.maxNewTokens;
        if (value.evalType !== 'exact_match' && value.evalType !== 'f1') delete value.evalType;
        if (typeof value.variant !== 'string') delete value.variant;
        if (typeof value.baseVariant !== 'string') delete value.baseVariant;
        if (value.model !== 'finetuned' && value.model !== 'base' && value.model !== 'both') delete value.model;
        const bySet = value.maxNewTokensBySet;
        if (bySet && typeof bySet === 'object') {
            const clean: Record<string, number> = {};
            for (const [key, n] of Object.entries(bySet as Record<string, unknown>)) {
                if (typeof n === 'number' && Number.isFinite(n) && n > 0) clean[key] = n;
            }
            value.maxNewTokensBySet = clean;
        } else {
            delete value.maxNewTokensBySet;
        }
        return value as LastUsedScoreRun;
    } catch {
        return {};
    }
}

export function writeLastUsed(projectId: number, value: LastUsedScoreRun): void {
    try {
        window.localStorage.setItem(lastUsedKey(projectId), JSON.stringify(value));
    } catch {
        // Storage blocked or full: the form still works with today's defaults.
    }
}
