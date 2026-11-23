/**
 * Multi-view evaluation hub with scorecards, failure analysis, and quality gates.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Check, X } from 'lucide-react';
import api from '../../api/client';
import { useJobsStore } from '../../stores/jobsStore';
import { cannotBeEvaluated } from '../../api/evalSummary';
import EmptyState from '../shared/EmptyState';
import LoadingState from '../shared/LoadingState';
import StepFooter from '../shared/StepFooter';
import { Term } from '../shared/Term';
import ClassificationChartsPanel from './ClassificationChartsPanel';
import RemediationMetricsPanel, { RemediationRowView, isRemediationMetrics, isRemediationRow, type RemediationMetrics, type RemediationRow } from './RemediationMetricsPanel';
import StructuredPlanMetricsPanel, { StructuredPlanRowView, isStructuredPlanMetrics, isStructuredPlanRow, type StructuredPlanMetrics, type StructuredPlanRow } from './StructuredPlanMetricsPanel';
import MultiOptionMetricsPanel, { MultiOptionRowView, isMultiOptionMetrics, isMultiOptionRow, type MultiOptionMetrics, type MultiOptionRow } from './MultiOptionMetricsPanel';
import RowContextChip from '../shared/RowContextChip';
import type { EvalRowContext, RowContextMetrics } from '../../types/rowContext';
import ToolCallMetricsPanel, { ToolCallRowView, isToolCallMetrics, isToolCallRow, type ToolCallMetrics, type ToolCallRow } from './ToolCallMetricsPanel';
import SliceBreakdownPanel, { sliceBreakdownOf } from './SliceBreakdownPanel';
import { gateReportStatus, measuredFailedGateIds, notMeasuredText } from './gateStatus';
import SqlExecutionMetricsPanel, { SqlExecutionRowView, hasSqlExecution, isSqlExecRow, type SqlExecRow, type SqlExecutionMetrics } from './SqlExecutionMetricsPanel';
import FindingSetMetricsPanel, { FindingSetRowView, isFindingSetMetrics, type FindingSetMetrics, type FindingSetRow } from './FindingSetMetricsPanel';
import DialogueJudgePanel, { findDialogueJudge, hasJudgedRows } from './DialogueJudgePanel';
import PreferencePairsFromEvalCard from './PreferencePairsFromEvalCard';
import FieldMatchMetricsPanel, { hasFieldMatchBreakdown, type FieldMatchMetrics } from './FieldMatchMetricsPanel';
import './ClassificationChartsPanel.css';
import ScorecardPanel from './ScorecardPanel';
import { evaluatedOnText, type GateEvaluatedOn } from './liftEvidence';
import AggregateRunBadge from './AggregateRunBadge';
import PerSliceMetricsTable from './PerSliceMetricsTable';
import BehavioralResultsTable from './BehavioralResultsTable';
import GoldSetWorkbenchPanel from './GoldSetWorkbenchPanel';
import EvalPackScaffoldPanel from './EvalPackScaffoldPanel';
import DriftReviewQueuePanel from './DriftReviewQueuePanel';
import FailureClustersPanel from './FailureClustersPanel';
import ActiveLearningPanel from './ActiveLearningPanel';
import SftLiftPanel from './SftLiftPanel';
import StudentTeacherComparisonPanel from './StudentTeacherComparisonPanel';
import FrontierComparisonPanel from './FrontierComparisonPanel';
import AutoRagComparisonPanel from './AutoRagComparisonPanel';
import RerouteRecommendationPanel from './RerouteRecommendationPanel';
import EvalGapsPanel from './EvalGapsPanel';
import ProbePackPanel from './ProbePackPanel';
import EvalSummaryCard from './EvalSummaryCard';
import { scoredNullAsAbsent, type JsonScoringRecord } from '../../api/nullScoring';
import EvalSetsCard from './EvalSetsCard';
import EvalRunForm, { SCORE_RUN_ADVANCED_ID } from './EvalRunForm';
import { displayPassRate, evalResultDatasetLabel, isEvalSetName, latestBuiltInResult, listEvalSets, type EvalSet } from '../../api/evalSets';
import './EvalPanel.css';
import ExternalBenchmarksPanel from './ExternalBenchmarksPanel';
import RedactionTryBox from './RedactionTryBox';
import AnswerKeyReviewPanel from '../data/AnswerKeyReviewPanel';
import { projectPath, resultsPath } from '../../utils/workspaceRoutes';
import { defaultRunId, isBaselineRun, runLabel, type RunSummary } from './runSelection';
import AdvancedSection from '../shared/AdvancedSection';
import VerdictCard from './VerdictCard';
import HowItsScoredCard, { HOW_SCORED_SETTINGS_ID } from './HowItsScoredCard';
import { isStringMatchAside, liftKeys, verdictResultsPerSet, type VerdictResultLike } from './verdictScore';
import { createSharedLiftFetch } from '../../api/sftLift';


/** Results workflows (Epic U phase U3). ``null`` = the one-page layout. */
export type EvalWorkflow = 'verdict' | 'failures' | 'compare' | 'try';

interface EvalPanelProps {
    projectId: number;
    /** The Results workflow to render; the panel stays mounted across tab
     *  switches so the selected run and its results persist. */
    workflow: EvalWorkflow;
    onNextStep?: () => void;
    /** The page's ``?open=`` value (or a legacy anchor mapped to it): opens that Advanced section. */
    openSection?: string | null;
    openNonce?: number;
}

type ExperimentSummary = RunSummary;

interface ScoredPrediction {
    prompt: string;
    reference: string;
    prediction: string;
    judge_score: number;
    judge_rationale: string;
}

interface InferenceMetrics {
    engine?: string;
    device?: string;
    average_latency_ms?: number;
    token_throughput_tps?: number;
}

/**
 * Story 1.5 Gate 2 — the eval finalize step populates this field when
 * ≥80% of prediction/gold pairs have disjoint top-level JSON keys.
 * Task-agnostic: covers PII span-extraction, classification,
 * summarization, alignment pairs, anything with a structured
 * output schema.
 */
interface SchemaMismatchReport {
    ratio: number;
    sample_size: number;
    pairs_mismatched: number;
    expected_top_keys: Array<{ keys: string[]; count: number }>;
    observed_top_keys: Array<{ keys: string[]; count: number }>;
    hint: string;
}

interface EvalMetrics {
    /** Eval prompt format; absent = 1 (v2 renders a row's own system message as in training). */
    prompt_format_version?: number;
    row_context?: RowContextMetrics;
    /** Opt-in "null counts as absent" JSON scoring (present only when it was on). */
    json_scoring?: JsonScoringRecord | null;
    /** Rows that carried their own output schema, checked against it (present only then). */
    output_schema?: { rows?: number; json_valid_rate?: number; schema_valid_rate?: number } | null;
    scored_predictions?: ScoredPrediction[];
    inference?: InferenceMetrics;
    schema_mismatch?: SchemaMismatchReport;
    // V1 of the ML-native visualisations arc — the classification
    // handler emits these per result; ClassificationChartsPanel renders
    // the bars + confusion-matrix heatmap when both are present.
    accuracy?: number;
    macro_f1?: number;
    per_class?: Record<string, {
        precision: number;
        recall: number;
        f1: number;
        support: number;
    }>;
    confusion_matrix?: Record<string, Record<string, number>>;
    candidate_set?: string[];
    // Typed JSON-record (field_match) breakdown: per-field null precision /
    // recall, out-of-vocab + type-violation rates; FieldMatchMetricsPanel.
    scoring_mode?: string;
    per_field?: FieldMatchMetrics['per_field'];
    null_precision?: number | null;
    null_recall?: number | null;
    hallucinated?: number;
    abstained_wrongly?: number;
    schema_valid_rate?: number | null;
    json_validity_rate?: number | null;
    overall_em?: number | null;
}

interface PredictionPreviewRow extends FindingSetRow, RemediationRow, StructuredPlanRow, ToolCallRow, MultiOptionRow, SqlExecRow {
    prompt?: string;
    reference?: string;
    prediction?: string;
    row_context?: EvalRowContext | null;
    row_tool_list_source?: 'row' | 'project' | null;
    /** Rows with their own output schema: whether the answer matched it (and why not). */
    row_output_schema_valid?: boolean;
    row_output_schema_errors?: string[] | null;
    formatted_prompt?: string;
    // Phase 5.3.2: QAHandler enriches predictions with these per-row
    // diagnostics. When the handler is the QA / instruction-following
    // family, every row carries:
    //   - answer_span: what got scored (extracted from CoT outputs when
    //     a marker like "Final answer:" or "Therefore:" matched).
    //   - span_marker: the regex that matched (null when no marker).
    //   - row_exact_match / row_f1: this row's per-row scores so the
    //     UI can paint a green/yellow/red status per row.
    answer_span?: string | null;
    span_marker?: string | null;
    row_exact_match?: number | null;
    row_f1?: number | null;
    // Phase 5.3.4: StructuredExtractionHandler enrichments. JSON
    // validity status + per-field comparison drive an inline note
    // ("JSON: valid · 3/3 fields") and a "Show field-by-field
    // comparison" disclosure beneath the prediction.
    is_valid_json?: boolean | null;
    parsed_prediction?: Record<string, unknown> | null;
    parsed_reference?: Record<string, unknown> | null;
    missing_required_fields?: string[] | null;
    row_field_results?: Record<string, { em?: number; f1?: number }> | null;
    // Phase 5.3.4b: span_set scoring mode adds per-row entity-level
    // diagnostics for PII / NER / span-extraction tasks. When
    // scoring_mode === "span_set", the UI renders an entity-by-entity
    // breakdown (matched / missed / hallucinated) instead of (or
    // alongside) the per-field comparison.
    scoring_mode?: string | null;
    row_matched_entities?: Array<{
        type?: string;
        start?: number;
        end?: number;
        text?: string;
    }> | null;
    row_missed_entities?: Array<{
        type?: string;
        start?: number;
        end?: number;
        text?: string;
    }> | null;
    row_hallucinated_entities?: Array<{
        type?: string;
        start?: number;
        end?: number;
        text?: string;
    }> | null;
    row_precision?: number | null;
    row_recall?: number | null;
    // Phase 5.3.5: RAGHandler enrichments. Faithfulness = fraction of
    // prediction tokens that appear in the context; context_recall =
    // fraction of gold tokens present in the context (retriever-side
    // diagnostic); unsupported_rate = fraction of pred tokens NOT in
    // context. is_faithful is the binary at the handler's threshold.
    // rag_context flows through so the UI can render a "Show context"
    // disclosure for grounding-mismatch debugging.
    rag_context?: string | null;
    rag_has_context?: boolean | null;
    rag_faithfulness?: number | null;
    rag_context_recall?: number | null;
    rag_unsupported_rate?: number | null;
    rag_is_faithful?: boolean | null;
    // Phase 5.3.6: AlignmentHandler enrichments for DPO / ORPO. The
    // model's output is compared to BOTH the chosen and rejected
    // completions; the per-row preference_correct flag indicates
    // whether the model preferred the right one.
    alignment_chosen?: string | null;
    alignment_rejected?: string | null;
    alignment_has_pair?: boolean | null;
    alignment_chosen_sim?: number | null;
    alignment_rejected_sim?: number | null;
    alignment_margin?: number | null;
    alignment_preference_correct?: boolean | null;
    latency_ms?: number;
    input_modality?: string;
}

interface EvalResult {
    id: number;
    dataset_name: string;
    eval_type: string;
    pass_rate: number | null;
    metrics: EvalMetrics;
    details?: {
        predictions_preview?: PredictionPreviewRow[];
        [key: string]: unknown;
    } | null;
    // Quality-Lift phase 8 slice 1 — multi-seed surfacing. The
    // EvalResultResponse schema started carrying these fields in
    // slice 1; the AggregateRunBadge in the result header reads
    // them to decide whether to render at all + which seed_group_id
    // to drill into.
    is_aggregate?: boolean;
    seed_group_id?: string | null;
}

interface SafetyScorecard {
    overall_risk: string;
    red_flags?: string[];
}

interface EvaluationPackSummary {
    pack_id: string;
    display_name: string;
    description: string;
    gate_count: number;
}

interface EvaluationPackListResponse {
    packs: EvaluationPackSummary[];
}

interface EvaluationPackPreferenceResponse {
    preferred_pack_id: string | null;
    active_pack_id: string | null;
    active_pack_source: string;
    active_pack?: {
        display_name?: string;
    };
}

interface GateCheck {
    gate_id: string;
    metric_id: string;
    resolved_metric_key?: string | null;
    operator: 'gte' | 'lte' | string;
    threshold: number | null;
    required: boolean;
    actual: number | null;
    passed: boolean;
    reason: string;
    // Multi-seed variance (Quality-Lift phase 1, slice 3). Present only
    // when the gate's metric was sourced from an aggregate EvalResult
    // (seed-group rollup). The dedicated drill-down lives in
    // ScorecardPanel; here we just want the inline ``mean ± std (n=N)``
    // surfacing so the eval-pack table doesn't silently underreport
    // spread.
    actual_std?: number;
    actual_min?: number;
    actual_max?: number;
    actual_n?: number;
    gate_value?: number;
    variance_policy?: 'lower_bound' | 'mean' | 'scalar';
    seed_group_id?: string | null;
}

interface GateReport {
    captured_at: string;
    passed: boolean;
    failed_gate_ids: string[];
    /** Required rules that failed on a measured value (absent on older backends). */
    measured_failed_gate_ids?: string[];
    /** pass / fail (a measured rule failed) / not_measured (only missing metrics). */
    status?: 'pass' | 'fail' | 'not_measured';
    missing_required_metrics: string[];
    missing_required_schema_metrics?: string[];
    evaluated_on?: GateEvaluatedOn | null;
    task_profile?: string;
    task_profile_source?: string;
    task_profile_selected?: string;
    task_profile_fallback_used?: boolean;
    task_spec?: {
        task_profile?: string;
        display_name?: string;
        required_metric_ids?: string[];
    };
    checks: GateCheck[];
    pack?: {
        pack_id?: string;
        display_name?: string;
    };
}

interface RemediationPlanGenerateRequest {
    experiment_id: number;
    evaluation_result_id?: number;
    max_failures: number;
}

interface RemediationPlanIndexEntry {
    plan_id: string;
    created_at?: string | null;
    experiment_id?: number | null;
    evaluation_result_id?: number | null;
    eval_type?: string | null;
    dataset_name?: string | null;
    root_causes?: string[];
    summary?: {
        total_failures_analyzed?: number | null;
        cluster_count?: number | null;
        recommendation_count?: number | null;
        dominant_root_cause?: string | null;
    };
}

interface RemediationPlanIndexResponse {
    project_id: number;
    count: number;
    plans: RemediationPlanIndexEntry[];
}

interface RemediationPlanCluster {
    cluster_id?: string;
    root_cause?: string;
    error_type?: string;
    slice?: string;
    failure_count?: number;
    confidence?: number;
}

interface RemediationPlanRecommendation {
    recommendation_id?: string;
    root_cause?: string;
    title?: string;
    confidence?: number;
    rationale?: string;
    expected_impact?: {
        metric?: string;
        estimated_delta?: number | null;
        confidence?: number | null;
        horizon?: string | null;
    };
    data_operations?: Array<string | { action?: string; description?: string }>;
    training_config_changes?: Array<string | { key?: string; from?: unknown; to?: unknown; reason?: string }>;
}

interface RemediationPlanDetail {
    plan_id: string;
    created_at?: string | null;
    source_evaluation?: {
        id?: number;
        eval_type?: string;
        dataset_name?: string;
        pass_rate?: number | null;
    };
    summary?: {
        total_failures_analyzed?: number | null;
        cluster_count?: number | null;
        recommendation_count?: number | null;
        dominant_root_cause?: string | null;
    };
    clusters?: RemediationPlanCluster[];
    recommendations?: RemediationPlanRecommendation[];
    analysis_evidence?: {
        failures_analyzed?: number;
    };
}

const AUTO_PACK_VALUE = '__auto__';

/** Loads the eval sets when Verdict › Test sets opens (its body mounts only while open). */
function TestSetsBody({ onMount, children }: { onMount: () => void; children: ReactNode }) {
    useEffect(() => {
        onMount();
    }, [onMount]);
    return <>{children}</>;
}

function getErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const maybeDetail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof maybeDetail === 'string' && maybeDetail.trim()) {
            return maybeDetail;
        }
        if (typeof maybeDetail === 'object' && maybeDetail !== null) {
            const detail = maybeDetail as {
                message?: unknown;
                actionable_fix?: unknown;
                error_code?: unknown;
            };
            const message = typeof detail.message === 'string' ? detail.message.trim() : '';
            const actionableFix = typeof detail.actionable_fix === 'string' ? detail.actionable_fix.trim() : '';
            if (message && actionableFix) {
                return `${message} ${actionableFix}`;
            }
            if (message) {
                return message;
            }
            const errorCode = typeof detail.error_code === 'string' ? detail.error_code.trim() : '';
            if (errorCode) {
                return errorCode;
            }
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Failed to run evaluation';
}

export default function EvalPanel({ projectId, onNextStep, workflow, openSection = null, openNonce = 0 }: EvalPanelProps) {
    const navigate = useNavigate();
    const wf = (name: EvalWorkflow) => workflow === name;
    // A same-page request to open an Advanced section (the Verdict card's "details").
    const [localOpen, setLocalOpen] = useState<{ id: string; nonce: number } | null>(null);
    const sectionRequest = (id: string) => ({
        openSection: localOpen?.id === id ? id : openSection,
        openNonce: openNonce + (localOpen?.id === id ? localOpen.nonce : 0),
    });
    const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
    const [selectedExp, setSelectedExp] = useState<number | null>(null);
    // True once the user picked a run (picker, probe sparkline): the
    // summary card's server-resolved run then no longer overrides it.
    const userPickedRunRef = useRef(false);
    const selectedExpRef = useRef<number | null>(null);
    const [evalResults, setEvalResults] = useState<EvalResult[]>([]);
    const selectedRun = useMemo(() => experiments.find((e) => e.id === selectedExp) ?? null, [experiments, selectedExp]);
    // Held-out eval Jobs for the selected run that have finished (phase 7b, gap N8): the base leg of "Both" files its
    // result under the baseline run, so the run's own result count does not change when it lands. This key does: the
    // lift cache, the Verdict card's lift lines and the eval-sets list refresh when it changes.
    const jobs = useJobsStore((state) => state.jobs);
    const evalJobsDoneKey = useMemo(() => jobs
        .filter((j) => (j.status === 'succeeded' || j.status === 'failed')
            && (j.kind === 'heldout_evaluation' || j.kind === 'llm_judge_evaluation')
            && (j.params as { experiment_id?: unknown } | null)?.experiment_id === selectedExp)
        .map((j) => j.id)
        .sort((a, b) => a - b)
        .join(','), [jobs, selectedExp]);
    // One lift-summary fetch per (run, set) shared by the Verdict card and Details' Base vs fine-tuned panel;
    // a new result list (or run, or a finished eval for it) starts a fresh cache.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the deps are the cache's lifetime, not inputs
    const sharedLiftFetch = useMemo(() => createSharedLiftFetch(), [projectId, selectedExp, evalResults.length, evalJobsDoneKey]);
    const [scorecard, setScorecard] = useState<SafetyScorecard | null>(null);

    // The "Score this run" form (``EvalRunForm`` holds its own fields); ``?open=score-run-advanced`` opens it.
    const [showRunForm, setShowRunForm] = useState(() => openSection === SCORE_RUN_ADVANCED_ID);
    const [seenFormRequest, setSeenFormRequest] = useState(`${openSection}#${openNonce}`);
    if (`${openSection}#${openNonce}` !== seenFormRequest) {
        setSeenFormRequest(`${openSection}#${openNonce}`);
        if (openSection === SCORE_RUN_ADVANCED_ID) setShowRunForm(true);
    }
    const formProjectRef = useRef(projectId);
    // Phase 7e (gap G3): a trained run with no evals opens the form by default (the card already says "evaluate
    // it now"); a run with any eval keeps it closed. Decided once per run, when its results first load; the button
    // (a user pick) always wins. ``formAutoOpened``: the form is open by that default, not by the user.
    // Review of 7e (M1): not on a run that cannot be evaluated at all: the summary card's own source
    // (``cannotBeEvaluated``: simulated, no model weights) or a multi-seed leader (its seeds hold the weights).
    // ``summaryFor``: the summary card's answer for the run it was asked about (``evaluable`` = may auto-open).
    const [resultsFor, setResultsFor] = useState<number | null>(null);
    const [summaryFor, setSummaryFor] = useState<{ runId: number; evaluable: boolean } | null>(null);
    const formDecidedRef = useRef<Set<number>>(new Set());
    const [formAutoOpened, setFormAutoOpened] = useState(false);
    useEffect(() => {
        if (!selectedRun || resultsFor !== selectedRun.id || formDecidedRef.current.has(selectedRun.id)) return;
        const trained = !isBaselineRun(selectedRun) && String(selectedRun.status ?? '').toLowerCase() === 'completed';
        if (evalResults.length === 0 && !trained) return;  // not trained yet: decide once it is
        // No evals: wait for the summary of this run (it is refetched when the run finishes training).
        if (evalResults.length === 0 && summaryFor?.runId !== selectedRun.id) return;
        formDecidedRef.current.add(selectedRun.id);
        if (evalResults.length === 0) {
            if (!showRunForm && summaryFor?.evaluable) {
                setShowRunForm(true);
                setFormAutoOpened(true);
            }
        } else if (formAutoOpened) {
            // Opened by the default for another run: this run has evals, so it starts closed.
            setShowRunForm(false);
            setFormAutoOpened(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- decided when a run's results first load
    }, [selectedRun, resultsFor, evalResults, summaryFor]);
    // The project's eval sets: listed in the form and managed under Verdict › Test sets.
    const [evalSets, setEvalSets] = useState<EvalSet[]>([]);
    // The form's Run waits for a remembered eval set while the list loads.
    const [evalSetsLoaded, setEvalSetsLoaded] = useState(false);

    const [loadingExperiments, setLoadingExperiments] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [evaluationPacks, setEvaluationPacks] = useState<EvaluationPackSummary[]>([]);
    const [preferredPackId, setPreferredPackId] = useState<string | null>(null);
    const [activePackId, setActivePackId] = useState<string | null>(null);
    const [activePackLabel, setActivePackLabel] = useState('');
    const [activePackSource, setActivePackSource] = useState('default');
    const [loadingPackState, setLoadingPackState] = useState(false);
    const [isSavingPackPreference, setIsSavingPackPreference] = useState(false);
    const [gateReport, setGateReport] = useState<GateReport | null>(null);
    const [isLoadingGateReport, setIsLoadingGateReport] = useState(false);
    const [gateErrorMessage, setGateErrorMessage] = useState<string | null>(null);
    // The gate report request itself failed (the Verdict card says "could not be loaded", not "not checked yet").
    const [gateLoadFailed, setGateLoadFailed] = useState(false);
    const [remediationPlans, setRemediationPlans] = useState<RemediationPlanIndexEntry[]>([]);
    const [selectedRemediationPlanId, setSelectedRemediationPlanId] = useState('');
    const [selectedRemediationEvalResultId, setSelectedRemediationEvalResultId] = useState('');
    const [remediationPlan, setRemediationPlan] = useState<RemediationPlanDetail | null>(null);
    const [isLoadingRemediationPlans, setIsLoadingRemediationPlans] = useState(false);
    const [isLoadingRemediationPlan, setIsLoadingRemediationPlan] = useState(false);
    const [isGeneratingRemediationPlan, setIsGeneratingRemediationPlan] = useState(false);
    const [remediationErrorMessage, setRemediationErrorMessage] = useState<string | null>(null);
    const [remediationMaxFailures, setRemediationMaxFailures] = useState(200);

    useEffect(() => {
        setSelectedExp(null);
        selectedExpRef.current = null;
        userPickedRunRef.current = false;
        setEvalResults([]);
        setScorecard(null);
        setErrorMessage(null);
        setGateErrorMessage(null);
        setEvaluationPacks([]);
        setPreferredPackId(null);
        setActivePackId(null);
        setActivePackLabel('');
        setActivePackSource('default');
        setGateReport(null);
        // Another project closes the form; the first mount keeps a `?open=score-run-advanced` request.
        if (formProjectRef.current !== projectId) {
            formProjectRef.current = projectId;
            setShowRunForm(false);
            setFormAutoOpened(false);
            setResultsFor(null);
            setSummaryFor(null);
            formDecidedRef.current = new Set();
            setEvalSets([]);
            setEvalSetsLoaded(false);
        }
        setRemediationPlans([]);
        setSelectedRemediationPlanId('');
        setSelectedRemediationEvalResultId('');
        setRemediationPlan(null);
        setRemediationErrorMessage(null);
        setRemediationMaxFailures(200);
    }, [projectId]);

    // Review of 7e (S1): this project's finished training Jobs (from the jobs store the top bar already polls). When
    // one finishes while Results is open, the run list reloads: the run's status turns "completed", the default run
    // moves to it (unless a run was picked), and the Score-form default above decides once for it.
    const trainingDoneKey = useMemo(() => jobs
        .filter((j) => j.kind === 'training_start' && j.project_id === projectId
            && (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled'))
        .map((j) => j.id)
        .sort((a, b) => a - b)
        .join(','), [jobs, projectId]);
    const trainingDoneSeenRef = useRef<{ projectId: number; key: string } | null>(null);
    useEffect(() => {
        let active = true;
        const seen = trainingDoneSeenRef.current;
        const refresh = seen !== null && seen.projectId === projectId && seen.key !== trainingDoneKey;
        trainingDoneSeenRef.current = { projectId, key: trainingDoneKey };
        // A Job already finished when the page loaded is not a refresh (the first load reads the runs anyway).
        if (seen !== null && seen.projectId === projectId && !refresh) return undefined;
        // The summary read before training finished is stale (the card refetches it with the new refresh token).
        if (refresh) setSummaryFor(null);
        const loadExperiments = async () => {
            if (!refresh) setLoadingExperiments(true);
            try {
                const res = await api.get<ExperimentSummary[]>(`/projects/${projectId}/training/experiments`);
                if (active) {
                    const runs = Array.isArray(res.data) ? res.data : [];
                    setExperiments(runs);
                    // Open on the summary's run (latest completed non-baseline)
                    // on every tab, not only once the Verdict card resolves.
                    const initial = defaultRunId(runs);
                    if (initial != null && (selectedExpRef.current == null || (refresh && !userPickedRunRef.current))) {
                        void loadResults(initial);
                    }
                }
            } catch (error) {
                if (active) {
                    setErrorMessage(getErrorMessage(error));
                }
            } finally {
                if (active) {
                    setLoadingExperiments(false);
                }
            }
        };

        void loadExperiments();
        return () => {
            active = false;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- loads per project and again when a training Job finishes
    }, [projectId, trainingDoneKey]);

    const loadPackState = async () => {
        setLoadingPackState(true);
        try {
            const [packsRes, prefRes] = await Promise.all([
                api.get<EvaluationPackListResponse>(`/projects/${projectId}/evaluation/packs`),
                api.get<EvaluationPackPreferenceResponse>(`/projects/${projectId}/evaluation/pack-preference`),
            ]);
            setEvaluationPacks(Array.isArray(packsRes.data?.packs) ? packsRes.data.packs : []);
            setPreferredPackId(prefRes.data?.preferred_pack_id ?? null);
            setActivePackId(prefRes.data?.active_pack_id ?? null);
            setActivePackSource(prefRes.data?.active_pack_source || 'default');
            setActivePackLabel(prefRes.data?.active_pack?.display_name || '');
        } catch {
            // Keep evaluation flow functional even if pack endpoints are unavailable.
            setEvaluationPacks([]);
            setPreferredPackId(null);
            setActivePackId(null);
            setActivePackSource('default');
            setActivePackLabel('');
        } finally {
            setLoadingPackState(false);
        }
    };

    const loadGateReport = async (expId: number) => {
        setIsLoadingGateReport(true);
        setGateErrorMessage(null);
        setGateLoadFailed(false);
        try {
            const res = await api.get<GateReport>(`/projects/${projectId}/evaluation/gates/${expId}`);
            setGateReport(res.data);
        } catch (error) {
            setGateReport(null);
            setGateLoadFailed(true);
            setGateErrorMessage(getErrorMessage(error));
        } finally {
            setIsLoadingGateReport(false);
        }
    };

    const loadRemediationPlan = async (planId: string) => {
        const token = String(planId || '').trim();
        if (!token) {
            setRemediationPlan(null);
            setSelectedRemediationPlanId('');
            return;
        }
        setIsLoadingRemediationPlan(true);
        setRemediationErrorMessage(null);
        try {
            const res = await api.get<RemediationPlanDetail>(
                `/projects/${projectId}/evaluation/remediation-plans/${encodeURIComponent(token)}`,
            );
            setRemediationPlan(res.data || null);
            setSelectedRemediationPlanId(token);
        } catch (error) {
            setRemediationPlan(null);
            setRemediationErrorMessage(getErrorMessage(error));
        } finally {
            setIsLoadingRemediationPlan(false);
        }
    };

    const loadRemediationPlans = async (expId: number) => {
        setIsLoadingRemediationPlans(true);
        setRemediationErrorMessage(null);
        try {
            const res = await api.get<RemediationPlanIndexResponse>(
                `/projects/${projectId}/evaluation/remediation-plans`,
                {
                    params: { experiment_id: expId, limit: 30 },
                },
            );
            const plans = Array.isArray(res.data?.plans) ? res.data.plans : [];
            setRemediationPlans(plans);
            if (plans.length === 0) {
                setSelectedRemediationPlanId('');
                setRemediationPlan(null);
                return;
            }
            const preferredPlanId = String(selectedRemediationPlanId || '').trim();
            const nextPlan = plans.find((item) => item.plan_id === preferredPlanId) || plans[0];
            if (nextPlan?.plan_id) {
                await loadRemediationPlan(nextPlan.plan_id);
            }
        } catch (error) {
            setRemediationPlans([]);
            setRemediationPlan(null);
            setRemediationErrorMessage(getErrorMessage(error));
        } finally {
            setIsLoadingRemediationPlans(false);
        }
    };

    useEffect(() => {
        void loadPackState();
    }, [projectId]);

    const savePackPreference = async (packId: string | null) => {
        setIsSavingPackPreference(true);
        setGateErrorMessage(null);
        try {
            const res = await api.put<EvaluationPackPreferenceResponse>(
                `/projects/${projectId}/evaluation/pack-preference`,
                { pack_id: packId },
            );
            setPreferredPackId(res.data?.preferred_pack_id ?? null);
            setActivePackId(res.data?.active_pack_id ?? null);
            setActivePackSource(res.data?.active_pack_source || 'default');
            setActivePackLabel(res.data?.active_pack?.display_name || '');
            if (selectedExp) {
                await loadGateReport(selectedExp);
            }
        } catch (error) {
            setErrorMessage(getErrorMessage(error));
        } finally {
            setIsSavingPackPreference(false);
        }
    };

    const listResults = async (expId: number) => {
        const [res, sc] = await Promise.all([
            api.get<EvalResult[]>(`/projects/${projectId}/evaluation/results/${expId}`),
            api.get<SafetyScorecard>(`/projects/${projectId}/evaluation/safety-scorecard/${expId}`),
        ]);
        setEvalResults(res.data);
        setResultsFor(expId);
        setScorecard(sc.data);
    };

    const loadResults = async (expId: number) => {
        setErrorMessage(null);
        selectedExpRef.current = expId;
        setSelectedExp(expId);
        try {
            await Promise.all([listResults(expId), loadGateReport(expId)]);
            void loadRemediationPlans(expId);
        } catch (error) {
            setErrorMessage(getErrorMessage(error));
        }
    };

    useEffect(() => {
        if (!selectedExp) {
            return;
        }
        if (!evalResults.length) {
            setSelectedRemediationEvalResultId('');
            return;
        }
        const selectedToken = String(selectedRemediationEvalResultId || '').trim();
        const hasSelected = evalResults.some((row) => String(row.id) === selectedToken);
        if (hasSelected) {
            return;
        }
        // Default: the newest eval on a built-in set (decision D3), as the backend picks with no id.
        setSelectedRemediationEvalResultId(String((latestBuiltInResult(evalResults) ?? evalResults[0]).id));
    }, [evalResults, selectedExp, selectedRemediationEvalResultId]);

    // Auto-refresh the eval results when a heldout/llm-judge Job for
    // the currently-selected experiment finishes. Without this the user has
    // to reload to see the new EvalResult after the bell toast fires. Keyed
    // on the finished Jobs' ids, so a polling tick that changes nothing
    // reloads nothing; the first key seen for a run (already-finished Jobs
    // on page load) is not a change.
    const seenEvalJobsKeyRef = useRef<{ exp: number | null; key: string } | null>(null);
    useEffect(() => {
        if (!selectedExp) return;
        const seen = seenEvalJobsKeyRef.current;
        seenEvalJobsKeyRef.current = { exp: selectedExp, key: evalJobsDoneKey };
        if (!seen || seen.exp !== selectedExp || seen.key === evalJobsDoneKey || !evalJobsDoneKey) return;
        // Re-fetch in the background; ignore failures (the user can
        // still see the bell toast even if the reload errors).
        void listResults(selectedExp).catch(() => {});
        void loadGateReport(selectedExp).catch(() => {});
        if (evalSetsLoaded || showRunForm) void reloadEvalSets();
    }, [evalJobsDoneKey, selectedExp]);  // eslint-disable-line react-hooks/exhaustive-deps

    const generateRemediationPlan = async () => {
        if (!selectedExp) {
            return;
        }
        setIsGeneratingRemediationPlan(true);
        setRemediationErrorMessage(null);
        try {
            const payload: RemediationPlanGenerateRequest = {
                experiment_id: selectedExp,
                max_failures: Math.max(1, Math.min(1000, Math.floor(remediationMaxFailures || 200))),
            };
            const evalResultId = Number.parseInt(String(selectedRemediationEvalResultId || '').trim(), 10);
            if (Number.isFinite(evalResultId) && evalResultId > 0) {
                payload.evaluation_result_id = evalResultId;
            }
            const res = await api.post<RemediationPlanDetail>(
                `/projects/${projectId}/evaluation/remediation-plans/generate`,
                payload,
            );
            const detail = res.data || null;
            setRemediationPlan(detail);
            setSelectedRemediationPlanId(String(detail?.plan_id || ''));
            await loadRemediationPlans(selectedExp);
        } catch (error) {
            setRemediationErrorMessage(getErrorMessage(error));
        } finally {
            setIsGeneratingRemediationPlan(false);
        }
    };

    const reloadEvalSets = useCallback(async () => {
        try {
            setEvalSets(await listEvalSets(projectId));
        } catch {
            setEvalSets([]);
        } finally {
            setEvalSetsLoaded(true);
        }
    }, [projectId]);

    useEffect(() => {
        if (showRunForm) void reloadEvalSets();
    }, [showRunForm, reloadEvalSets]);
    // The Verdict card's cross-prompt lines read the eval sets' ``last_evals``: load them once the run has a
    // result on an eval set (phase 7b, gap N3).
    const runHasEvalSetResults = evalResults.some((r) => isEvalSetName(r.dataset_name));
    useEffect(() => {
        if (runHasEvalSetResults) void reloadEvalSets();
    }, [runHasEvalSetResults, evalResults.length, reloadEvalSets]);

    const riskColor = (risk: string) => {
        if (risk === 'low') return 'badge-success';
        if (risk === 'medium') return 'badge-warning';
        if (risk === 'high') return 'badge-error';
        return 'badge-info';
    };

    const radarData = useMemo(
        () =>
            evalResults
                .filter((result) => result.eval_type === 'llm_judge')
                .map((result) => ({
                    subject: result.dataset_name,
                    A: (result.pass_rate ?? 0) * 100,
                    fullMark: 100,
                })),
        [evalResults],
    );

    const llmJudgeResults = useMemo(
        () => evalResults.filter((result) => result.eval_type === 'llm_judge'),
        [evalResults],
    );

    const scoredPredictions = llmJudgeResults[0]?.metrics.scored_predictions ?? [];

    // Phase 5.2a: surface the per-row predictions the eval runner already
    // stores in result.details.predictions_preview. The API returns results
    // newest-first (order_by created_at.desc), so iterate forward — picking
    // the FIRST result that carries a preview gives us the most recent
    // sample. Falling back to older runs would let the card mismatch the
    // headline metric without warning.
    const samplePredictionsResult = useMemo(() => {
        for (let i = 0; i < evalResults.length; i += 1) {
            const preview = evalResults[i].details?.predictions_preview;
            if (Array.isArray(preview) && preview.length > 0) {
                return evalResults[i];
            }
        }
        return null;
    }, [evalResults]);

    const samplePredictions = useMemo<PredictionPreviewRow[]>(() => {
        const preview = samplePredictionsResult?.details?.predictions_preview;
        return Array.isArray(preview) ? (preview as PredictionPreviewRow[]) : [];
    }, [samplePredictionsResult]);

    const samplePredictionsEvalType = samplePredictionsResult?.eval_type ?? '';
    const samplePredictionsDatasetName =
        samplePredictionsResult?.dataset_name ?? '';
    const samplePredictionsResultId = samplePredictionsResult?.id ?? null;
    const samplePredictionsIsLatestRun =
        samplePredictionsResult != null &&
        evalResults.length > 0 &&
        samplePredictionsResult.id === evalResults[0].id;
    // Phase 5.3.2: when the QAHandler ran, every preview row carries
    // row_exact_match / row_f1 so we can show a status badge column.
    // Detect at the run level so the table shape is consistent across
    // all rows in the same run.
    const samplePredictionsHasRowScores = samplePredictions.some(
        (row) =>
            (row.row_exact_match !== undefined && row.row_exact_match !== null) ||
            (row.row_f1 !== undefined && row.row_f1 !== null),
    );

    const latestInference = useMemo(() => {
        for (let i = 0; i < evalResults.length; i += 1) {
            const inference = evalResults[i].metrics.inference;
            if (inference) {
                return inference;
            }
        }
        return null;
    }, [evalResults]);

    const activePackDisplayName = useMemo(() => {
        if (activePackLabel.trim()) {
            return activePackLabel;
        }
        if (!activePackId) {
            return 'Auto';
        }
        const found = evaluationPacks.find((pack) => pack.pack_id === activePackId);
        return found?.display_name || activePackId;
    }, [activePackId, activePackLabel, evaluationPacks]);

    const gateStatusBadge = useMemo(() => {
        if (!gateReport) {
            return { className: 'badge-info', label: 'NO REPORT' };
        }
        const status = gateReportStatus(gateReport);
        if (status === 'pass') return { className: 'badge-success', label: 'PASS' };
        if (status === 'not_measured') return { className: 'badge-warning', label: 'NOT MEASURED' };
        return { className: 'badge-error', label: 'FAIL' };
    }, [gateReport]);

    const remediationDataOps = (recommendation: RemediationPlanRecommendation): string[] => {
        const rows: string[] = [];
        for (const item of recommendation.data_operations || []) {
            if (typeof item === 'string' && item.trim()) {
                rows.push(item.trim());
                continue;
            }
            if (typeof item === 'object' && item !== null) {
                const action = String((item as { action?: unknown }).action || '').trim();
                const description = String((item as { description?: unknown }).description || '').trim();
                if (action && description) {
                    rows.push(`${action}: ${description}`);
                } else if (description) {
                    rows.push(description);
                } else if (action) {
                    rows.push(action);
                }
            }
        }
        return rows;
    };

    const remediationTrainingChanges = (recommendation: RemediationPlanRecommendation): string[] => {
        const rows: string[] = [];
        for (const item of recommendation.training_config_changes || []) {
            if (typeof item === 'string' && item.trim()) {
                rows.push(item.trim());
                continue;
            }
            if (typeof item === 'object' && item !== null) {
                const key = String((item as { key?: unknown }).key || '').trim();
                const from = (item as { from?: unknown }).from;
                const to = (item as { to?: unknown }).to;
                const reason = String((item as { reason?: unknown }).reason || '').trim();
                const base = key
                    ? `${key}: ${String(from ?? 'current')} -> ${String(to ?? 'recommended')}`
                    : reason || 'Adjust training configuration';
                rows.push(reason ? `${base} (${reason})` : base);
            }
        }
        return rows;
    };

    const onPackPreferenceSelect = (value: string) => {
        const nextPackId = value === AUTO_PACK_VALUE ? null : value;
        void savePackPreference(nextPackId);
    };

    // The Verdict card shows base vs fine-tuned lines for this run: the summary card then drops its own
    // base-vs-fine-tuned headline (it keeps the seed table, judge trust, threshold tuning and failing rows).
    const selectedIsBaseline = experiments.some((e) => e.id === selectedExp && isBaselineRun(e));
    const verdictShowsLift = selectedExp != null && !selectedIsBaseline
        && liftKeys(verdictResultsPerSet(evalResults as unknown as VerdictResultLike[])
            .filter((r, _i, all) => !isStringMatchAside(r, all))).length > 0;

    // O2: plain string-match results next to task-scored ones sit under "Other stored metrics".
    const historyAside = evalResults.filter((r) => isStringMatchAside(r as unknown as VerdictResultLike, evalResults as unknown as VerdictResultLike[]));
    const historyMain = evalResults.filter((r) => !historyAside.includes(r));

    // Quiet "prompt format" label in the history, only once some result is on a newer format.
    const showFormatLabels = evalResults.some((r) => (r.metrics?.prompt_format_version ?? 1) >= 2);

    return (
        <div className="animate-fade-in u-stack-xl">
            {wf('verdict') && (
                <VerdictCard
                    projectId={projectId}
                    experimentId={selectedExp}
                    runLabel={runLabel(experiments, selectedExp)}
                    isBaseline={selectedIsBaseline}
                    results={evalResults as unknown as VerdictResultLike[]}
                    gate={gateReport ? {
                        status: gateReportStatus(gateReport),
                        failed: measuredFailedGateIds(gateReport),
                        missing: gateReport.missing_required_metrics,
                        ruleSet: gateReport.pack?.display_name || gateReport.pack?.pack_id || null,
                        evaluatedOn: gateReport.evaluated_on ?? null,
                    } : null}
                    gateError={gateLoadFailed ? (gateErrorMessage || 'could not be loaded') : null}
                    fetchLift={sharedLiftFetch}
                    evalSets={evalSets}
                    baseModel={(selectedRun as { base_model?: string | null } | null)?.base_model ?? null}
                    refreshKey={evalJobsDoneKey}
                    safetyRisk={scorecard?.overall_risk ?? null}
                    onOpenDetails={() => setLocalOpen((prev) => ({ id: 'verdict-details', nonce: (prev?.nonce ?? 0) + 1 }))}
                    slicesHref={resultsPath(projectId, 'failures', 'open=failures-tools')}
                />
            )}
            {wf('verdict') && (
                <EvalSummaryCard
                    projectId={projectId}
                    experimentId={selectedExp}
                    refreshToken={`${evalResults.length}|${trainingDoneKey}`}
                    hideTryBox
                    verdictShowsLift={verdictShowsLift}
                    onSummary={(expId, summary) => {
                        if (expId != null) setSummaryFor({ runId: expId, evaluable: !cannotBeEvaluated(summary) && !summary?.seed_group_leader });
                    }}
                    onResolved={(expId) => {
                        if (!userPickedRunRef.current && expId !== selectedExpRef.current) void loadResults(expId);
                    }}
                />
            )}
            {wf('compare') && <ExternalBenchmarksPanel projectId={projectId} experimentId={selectedExp} />}
            {(<>
            <div className="card">
                <h3 className="u-card-title">Run</h3>
                {loadingExperiments ? (
                    <LoadingState label="Loading runs" lines={2} />
                ) : experiments.length === 0 ? (
                    <EmptyState
                        title="No runs to evaluate"
                        description="Train a run first — then come back here, pick an answer key, pick a pass/fail rule set, and run a scored evaluation."
                        docsHref="http://localhost:3001/docs/workflows/evaluation-and-remediation"
                    />
                ) : (
                    <div className="u-row">
                        {experiments.map((experiment) => (
                            <button
                                key={experiment.id}
                                type="button"
                                className={`btn ${selectedExp === experiment.id ? 'btn-primary' : 'btn-secondary'} eval-panel__run-btn`}
                                data-testid={`eval-run-${experiment.id}`}
                                onClick={() => {
                                    userPickedRunRef.current = true;
                                    void loadResults(experiment.id);
                                }}
                            >
                                #{experiment.id} · {experiment.name}
                                {isBaselineRun(experiment) && <span className="badge badge-info eval-panel__run-baseline">baseline</span>}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {errorMessage && (
                <div className="card eval-panel__error-card">
                    {errorMessage}
                </div>
            )}
            </>)}
            {wf('try') && (
                <div className="card" data-testid="eval-try-redact">
                    <h3 className="eval-panel__h3 eval-panel__h3--sm">Redact a text</h3>
                    {selectedExp && evalResults.some((r) => (r.metrics as { scoring_mode?: string | null } | null)?.scoring_mode === 'span_set') ? (
                        <RedactionTryBox projectId={projectId} experimentId={selectedExp} />
                    ) : (
                        <p className="eval-pack-note">Available for a span-tagging run once it has been evaluated.</p>
                    )}
                </div>
            )}
            {(<>
            {(<>

            {selectedExp && wf('verdict') && (
                <div className="u-row u-between">
                    <h2 className="eval-panel__h2">Evaluate</h2>
                    <div className="u-row">
                        {experiments.length >= 2 && (() => {
                            // E3 — quick deep-link to the side-by-side eval compare page.
                            // Pairs the currently-selected experiment (B) with the
                            // chronologically prior one (A). When the user wants a
                            // different pair they can still tweak the URL params
                            // by hand — this affordance covers the most common
                            // "did my latest run regress against the previous one?"
                            // question with one click.
                            // Baseline runs are the base model, not a prior run
                            // (the lift panel compares against it already).
                            const sorted = [...experiments]
                                .filter((e) => e.id === selectedExp || !isBaselineRun(e))
                                .sort((a, b) => Number(b.id) - Number(a.id));
                            const selectedIdx = sorted.findIndex((e) => e.id === selectedExp);
                            const prior = sorted[selectedIdx + 1] ?? sorted[1];
                            if (!prior || prior.id === selectedExp) return null;
                            return (
                                <button
                                    type="button"
                                    className="btn btn-secondary"
                                    data-testid="eval-panel-compare-to-prior"
                                    onClick={() =>
                                        navigate(`${projectPath(projectId, 'results')}?tab=compare&a=${prior.id}&b=${selectedExp}`)
                                    }
                                    title={`Side-by-side eval comparison: ${prior.name} → currently selected`}
                                >
                                    Compare to #{prior.id}
                                </button>
                            );
                        })()}
                        <button
                            className="btn btn-primary"
                            onClick={() => { setFormAutoOpened(false); setShowRunForm((open) => !open); }}
                            data-testid="eval-score-run-open"
                        >
                            {showRunForm ? 'Close Score this run' : '+ Score this run'}
                        </button>
                    </div>
                </div>
            )}



            {showRunForm && wf('verdict') && selectedRun && (
                <EvalRunForm
                    key={`${projectId}:${selectedRun.id}`}
                    projectId={projectId}
                    run={selectedRun}
                    evalSets={evalSets}
                    evalSetsLoading={!evalSetsLoaded}
                    results={evalResults as unknown as VerdictResultLike[]}
                    fetchLift={sharedLiftFetch}
                    {...sectionRequest(SCORE_RUN_ADVANCED_ID)}
                    onQueued={() => { setShowRunForm(false); setFormAutoOpened(false); }}
                    onOpenTestSets={() => setLocalOpen((prev) => ({ id: 'verdict-test-sets', nonce: (prev?.nonce ?? 0) + 1 }))}
                />
            )}

            {/* Test sets: named eval sets (upload, versions, prompt-format variants, base vs fine-tuned per set).
                Was "More options: eval sets" inside the eval form. ⌘K: Results › Test sets. */}
            {wf('verdict') && (
                <AdvancedSection
                    pageKey="results-verdict"
                    id="verdict-test-sets"
                    title="Test sets"
                    count={evalSets.length || undefined}
                    hint="named test sets: upload, versions, prompt-format variants, base vs fine-tuned per set"
                    {...sectionRequest('verdict-test-sets')}
                >
                    <TestSetsBody onMount={reloadEvalSets}>
                        <EvalSetsCard projectId={projectId} sets={evalSets} onChanged={reloadEvalSets} />
                    </TestSetsBody>
                </AdvancedSection>
            )}

            {/* Quality-Lift phase 8 slice 1 — render the
                AggregateRunBadge above the scorecard for every
                multi-seed aggregate EvalResult so the user sees
                ``F1: 0.83 ± 0.04 (n=3)`` before any scalar metric.
                The scorecard's gate rows still surface their own
                per-seed breakdown; this is the result-header
                surface (phase 1's mean ± std promise visible at a
                glance, not just inside failing gates). */}
            {wf('verdict') && evalResults
                .filter((r) => r.is_aggregate && r.seed_group_id)
                .map((r) => (
                    <AggregateRunBadge
                        key={`agg-${r.id}`}
                        projectId={projectId}
                        seedGroupId={r.seed_group_id as string}
                        datasetName={r.dataset_name}
                        evalType={r.eval_type}
                        metrics={r.metrics as Record<string, unknown>}
                    />
                ))}




            {/* Scores: the per-result score panels (Metric History, tool-call / JSON / slice panels…). Collapsed;
                the score of record is on the Verdict card above. ⌘K: Results › Scores. */}
            {wf('verdict') && selectedExp != null && evalResults.length > 0 && (
                <AdvancedSection
                    pageKey="results-verdict"
                    id="verdict-scores"
                    title="Scores"
                    count={evalResults.length}
                    hint="every stored result of this run: metric history, per-result panels, slices"
                    keepMounted
                    {...sectionRequest('verdict-scores')}
                >
                    <div className="u-stack-lg">
                            <div className="card">
                                <h3 className="u-card-title">Metric History</h3>
                                <div className="table-container eval-panel__history-scroll">
                                    <table className="docs-table">
                                        <thead>
                                            <tr>
                                                <th>Dataset</th>
                                                <th>Type</th>
                                                <th>Pass Rate</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {historyMain.map((result) => (
                                                <tr key={result.id}>
                                                    <td data-testid={`eval-history-dataset-${result.id}`}>{evalResultDatasetLabel(result)}</td>
                                                    <td>
                                                        <span className="badge badge-accent u-caps">
                                                            {isToolCallMetrics(result.metrics as { scoring_mode?: string | null }) ? 'tool call' : result.eval_type.replace('_', ' ')}
                                                        </span>
                                                        {showFormatLabels && (
                                                            <span className="u-meta" data-testid={`eval-format-${result.id}`}>
                                                                {' '}prompt rendering v{result.metrics?.prompt_format_version ?? 1}
                                                            </span>
                                                        )}
                                                        {scoredNullAsAbsent(result.metrics) && (
                                                            <span className="u-meta" data-testid={`eval-null-as-absent-${result.id}`}>
                                                                {' '}null counts as absent
                                                            </span>
                                                        )}
                                                    </td>
                                                    <td>
                                                        <strong
                                                            className={
                                                                (displayPassRate(result) ?? 0) > 0.8
                                                                    ? 'u-text-success'
                                                                    : (displayPassRate(result) ?? 0) > 0.5
                                                                      ? 'u-text-warning'
                                                                      : 'u-text-error'
                                                            }
                                                            data-testid={`eval-pass-rate-${result.id}`}
                                                        >
                                                            {displayPassRate(result) != null ? `${((displayPassRate(result) ?? 0) * 100).toFixed(1)}%` : '—'}
                                                        </strong>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                {historyAside.length > 0 && (
                                    <details className="eval-collapsible" data-testid="eval-history-string-match">
                                        <summary>Other stored metrics (string match) ({historyAside.length})</summary>
                                        <p className="u-hint">
                                            Plain exact-match results on this run, stored before or beside the task scoring. The
                                            task-scored results above are the score of record; these stay stored.
                                        </p>
                                        <table className="docs-table">
                                            <tbody>
                                                {historyAside.map((result) => (
                                                    <tr key={result.id} data-testid={`eval-history-aside-${result.id}`}>
                                                        <td>{evalResultDatasetLabel(result)} · result #{result.id}</td>
                                                        <td>exact match (string)</td>
                                                        <td data-testid={`eval-pass-rate-${result.id}`}>
                                                            {displayPassRate(result) != null ? `${((displayPassRate(result) ?? 0) * 100).toFixed(1)}%` : '—'}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </details>
                                )}

                                {/* V1 ML-native visualisations — classification per-class bars +
                                    confusion-matrix heatmap. Renders for every eval result that
                                    carries both `per_class` and `confusion_matrix` (the
                                    classification handler always emits both); silent on other
                                    task profiles. Each result that qualifies gets its own block
                                    so projects with both gold-dev and gold-test results show two
                                    distinct charts. */}
                                {evalResults
                                    .filter((r) => !isFindingSetMetrics(r.metrics) && !isRemediationMetrics(r.metrics) && !isStructuredPlanMetrics(r.metrics) && !isToolCallMetrics(r.metrics) && r.metrics?.per_class && Object.keys(r.metrics.per_class).length > 0)
                                    .map((r) => (
                                        <ClassificationChartsPanel
                                            key={`cls-charts-${r.id}`}
                                            perClass={r.metrics.per_class!}
                                            confusionMatrix={r.metrics.confusion_matrix || {}}
                                            candidates={r.metrics.candidate_set}
                                            macroF1={r.metrics.macro_f1 ?? null}
                                            accuracy={r.metrics.accuracy ?? null}
                                        />
                                    ))}

                                {/* Findings evals (finding_set): strict P/R/F1, clean-chunk
                                    false positives, malformed output, per-label table. */}
                                {evalResults
                                    .filter((r) => isFindingSetMetrics(r.metrics))
                                    .map((r) => (
                                        <FindingSetMetricsPanel
                                            key={`fs-metrics-${r.id}`}
                                            metrics={r.metrics as unknown as FindingSetMetrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Code-remediation evals (remediation): pass / applies / parses /
                                    resolved / citation rates, patch failure reasons, judge score. */}
                                {evalResults
                                    .filter((r) => isRemediationMetrics(r.metrics))
                                    .map((r) => (
                                        <RemediationMetricsPanel
                                            key={`rem-metrics-${r.id}`}
                                            metrics={r.metrics as unknown as RemediationMetrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Structured-plan evals (structured_plan): schema validity, pointer
                                    matches and the rubric per criterion. */}
                                {evalResults
                                    .filter((r) => isStructuredPlanMetrics(r.metrics))
                                    .map((r) => (
                                        <StructuredPlanMetricsPanel
                                            key={`sp-metrics-${r.id}`}
                                            metrics={r.metrics as unknown as StructuredPlanMetrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Multi-option evals (multi_option): options passing the operator's verifier, top
                                    option, distinct strategies, cross-format agreement, per-format / per-check bars. */}
                                {evalResults
                                    .filter((r) => isMultiOptionMetrics(r.metrics))
                                    .map((r) => (
                                        <MultiOptionMetricsPanel
                                            key={`mo-metrics-${r.id}`}
                                            metrics={r.metrics as unknown as MultiOptionMetrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Tool-call evals (tool_call): pass rate, tool pick, validity against the catalog. */}
                                {evalResults
                                    .filter((r) => isToolCallMetrics(r.metrics))
                                    .map((r) => (
                                        <ToolCallMetricsPanel
                                            key={`tc-metrics-${r.id}`}
                                            metrics={r.metrics as unknown as ToolCallMetrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                            rowsAvailable={(r.details?.dataset as { rows_available?: number | null } | undefined)?.rows_available ?? null}
                                        />
                                    ))}

                                {/* Per-slice pass (and strict) counts with Wilson intervals, from row tags / project slices. Collapsed. */}
                                {evalResults
                                    .filter((r) => sliceBreakdownOf(r.metrics as Record<string, unknown>))
                                    .map((r) => (
                                        <SliceBreakdownPanel
                                            key={`slices-${r.id}`}
                                            breakdown={sliceBreakdownOf(r.metrics as Record<string, unknown>)!}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Opt-in SQL execution match: predicted vs answer-key query results on the project's databases. */}
                                {evalResults
                                    .filter((r) => hasSqlExecution(r.metrics as SqlExecutionMetrics))
                                    .map((r) => (
                                        <SqlExecutionMetricsPanel
                                            key={`sx-metrics-${r.id}`}
                                            metrics={r.metrics as unknown as SqlExecutionMetrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Multi-turn evals: per-turn replay judge summary. */}
                                {evalResults
                                    .filter((r) => findDialogueJudge(r.metrics))
                                    .map((r) => (
                                        <DialogueJudgePanel
                                            key={`dj-metrics-${r.id}`}
                                            summary={findDialogueJudge(r.metrics)!}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Judged results: build preference pairs from wrong / partial answers. */}
                                {evalResults
                                    .filter((r) => hasJudgedRows(r.metrics))
                                    .map((r) => (
                                        <PreferencePairsFromEvalCard
                                            key={`pairs-${r.id}`}
                                            projectId={projectId}
                                            evalResultId={r.id}
                                            split={r.dataset_name}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}

                                {/* Typed JSON-record evals (field_match): per-field null
                                    precision / recall, made-up values, out-of-vocabulary and
                                    type-violation rates. Each block names its run + dataset. */}
                                {evalResults
                                    .filter((r) => hasFieldMatchBreakdown(r.metrics))
                                    .map((r) => (
                                        <FieldMatchMetricsPanel
                                            key={`fm-metrics-${r.id}`}
                                            metrics={r.metrics}
                                            provenance={`Run #${selectedExp ?? '?'} · ${evalResultDatasetLabel(r)} · result ${r.id}`}
                                        />
                                    ))}
                            </div>
                    {/* Quality-Lift phase 8 slice 2 — per-slice metrics table.
                        Reads ``metrics["per_slice"]`` off each EvalResult and
                        cross-references the live gate report so failing cells
                        light up red — only against the user's actual gate
                        thresholds (honest gates only, never an invented bar).
                        One table per EvalResult that carries per_slice data;
                        silent when neither slice_definitions nor per_slice
                        payload exist (caller flips ``showEmptyStateWhenNoSlices``
                        if they want a nudge). */}
                    {evalResults
                        .filter((r) => {
                            const ps = (r.metrics as Record<string, unknown>)?.['per_slice'];
                            return typeof ps === 'object' && ps !== null;
                        })
                        .map((r) => (
                            <PerSliceMetricsTable
                                key={`ps-${r.id}`}
                                projectId={projectId}
                                metrics={r.metrics as Record<string, unknown>}
                                gateChecks={gateReport?.checks ?? []}
                            />
                        ))}
                    </div>
                </AdvancedSection>
            )}
            {/* How it's scored + the project's scoring settings (phase 4: moved from Train › More options). */}
            {wf('verdict') && (
                <HowItsScoredCard
                    projectId={projectId}
                    runLabel={selectedExp != null ? runLabel(experiments, selectedExp) : null}
                    results={selectedExp != null ? evalResults as unknown as VerdictResultLike[] : []}
                    {...sectionRequest(HOW_SCORED_SETTINGS_ID)}
                />
            )}
            {/* Verdict details: the rule report, base vs fine-tuned in full, scorecard, runtime and safety.
                The Verdict card above carries their one-line summary. ⌘K: Results › Verdict details. */}
            {wf('verdict') && selectedExp != null && (
                <AdvancedSection
                    pageKey="results-verdict"
                    id="verdict-details"
                    title="Details"
                    hint="pass/fail rule report, base vs fine-tuned in full, scorecard, runtime, safety"
                    {...sectionRequest('verdict-details')}
                >
                    <div className="u-stack-lg">
                    {wf('verdict') && selectedExp != null && gateReport && (
                        <div className="card eval-panel__gate" data-testid="eval-verdict-gate">
                            <div className="eval-panel__gate-head">
                                <h3>Pass/fail rules</h3>
                                <span className={`badge ${gateStatusBadge.className}`}>{gateStatusBadge.label}</span>
                            </div>
                            <div className="eval-panel__gate-meta">
                                <span data-testid="eval-verdict-gate-run"><strong>{runLabel(experiments, selectedExp)}</strong></span>
                                <span>Rule set <strong>{gateReport.pack?.display_name || gateReport.pack?.pack_id || 'auto'}</strong></span>
                                {gateReport.captured_at && <span>Checked <strong>{new Date(gateReport.captured_at).toLocaleString()}</strong></span>}
                            </div>
                            {gateReport.evaluated_on && (
                                <div className="eval-panel__gate-meta" data-testid="eval-verdict-gate-evaluated-on">
                                    <span>{evaluatedOnText(gateReport.evaluated_on)}</span>
                                    <button
                                        type="button"
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => {
                                            userPickedRunRef.current = true;
                                            void loadResults(gateReport.evaluated_on!.experiment_id);
                                        }}
                                    >
                                        Open run #{gateReport.evaluated_on.experiment_id}
                                    </button>
                                </div>
                            )}
                            {measuredFailedGateIds(gateReport).length > 0 && (
                                <div className="eval-panel__gate-warn" data-testid="eval-verdict-gate-failed">Failed required rules: {measuredFailedGateIds(gateReport).join(', ')}</div>
                            )}
                            {gateReport.missing_required_metrics.length > 0 && (
                                <div className="eval-panel__gate-warn" data-testid="eval-verdict-gate-missing">
                                    Not measured: {gateReport.missing_required_metrics.join(', ')}. {notMeasuredText(gateReport.missing_required_metrics.length)}
                                </div>
                            )}
                        </div>
                    )}
                    {selectedExp && evalResults.length > 0 && wf('verdict') && (
                        <SftLiftPanel
                            projectId={projectId}
                            experimentId={selectedExp}
                            refreshToken={evalResults.length}
                            fetchLift={sharedLiftFetch}
                        />
                    )}
                    {selectedExp && wf('verdict') && (
                        <div id="experiment-scorecard">
                            <ScorecardPanel projectId={projectId} experimentId={selectedExp} runLabel={runLabel(experiments, selectedExp)} />
                        </div>
                    )}
                    {latestInference && wf('verdict') && (
                        <div className="card eval-panel__stats">
                            <div>
                                <div className="u-meta">Runtime</div>
                                <strong>{latestInference.engine ?? 'unknown'}</strong>
                            </div>
                            <div>
                                <div className="u-meta">Device</div>
                                <strong>{latestInference.device ?? 'unknown'}</strong>
                            </div>
                            <div>
                                <div className="u-meta">Avg Latency</div>
                                <strong>{latestInference.average_latency_ms != null ? `${latestInference.average_latency_ms.toFixed(2)} ms` : '—'}</strong>
                            </div>
                            <div>
                                <div className="u-meta">Token Throughput</div>
                                <strong>{latestInference.token_throughput_tps != null ? `${latestInference.token_throughput_tps.toFixed(2)} tok/s` : '—'}</strong>
                            </div>
                        </div>
                    )}
                    {scorecard && (
                        <div className="card">
                            <h3 className="u-card-title">Safety Scorecard</h3>
                            <div className="u-row-lg u-mb-md">
                                <div>
                                    <strong>Overall Risk:</strong>{' '}
                                    <span className={`badge ${riskColor(scorecard.overall_risk)}`}>{scorecard.overall_risk.toUpperCase()}</span>
                                </div>
                            </div>
                            {(scorecard.red_flags ?? []).length > 0 && (
                                <div className="u-surface u-tint-error u-mt-sm">
                                    {(scorecard.red_flags ?? []).map((flag, idx) => (
                                        <div key={idx} className="u-text-error u-text-sm">
                                            <AlertTriangle className="u-icon" aria-hidden="true" /> {flag}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                    </div>
                </AdvancedSection>
            )}

            {selectedExp && evalResults.length > 0 && wf('failures') && (
                <FailureClustersPanel
                    projectId={projectId}
                    evalResults={evalResults.map((r) => ({
                        id: r.id,
                        dataset_name: r.dataset_name,
                        eval_type: r.eval_type,
                        pass_rate: r.pass_rate,
                    }))}
                />
            )}



            {selectedExp && evalResults.length > 0 && wf('compare') && (
                <StudentTeacherComparisonPanel
                    projectId={projectId}
                    experimentId={selectedExp}
                    experiments={experiments}
                    refreshToken={evalResults.length}
                />
            )}

            {/* Epic D — honest SLM-vs-frontier report (X% as good as gpt-4o-mini */}
            {/* at Y% cost, Z× latency). Self-hides on 4xx/5xx. */}
            {selectedExp && evalResults.length > 0 && wf('compare') && (
                <FrontierComparisonPanel
                    projectId={projectId}
                    experimentId={selectedExp}
                    refreshToken={evalResults.length}
                />
            )}


            {/* Phase 9d — auto-RAG with-vs-without comparison. Renders */}
            {/* only for QA-SFT projects (the panel itself self-hides */}
            {/* when the recipe is ineligible via a 400 from the API). */}
            {wf('compare') && <AutoRagComparisonPanel projectId={projectId} />}





            {samplePredictionsResult?.metrics?.schema_mismatch && wf('failures') && (
                <div
                    className="card eval-panel__schema-banner"
                    data-testid="eval-schema-mismatch-banner"
                >
                    <h3 className="eval-panel__schema-title">
                        Schema mismatch detected
                        <span className="eval-panel__schema-detail">
                            {(samplePredictionsResult.metrics.schema_mismatch.ratio * 100).toFixed(0)}%
                            of{' '}
                            {samplePredictionsResult.metrics.schema_mismatch.sample_size}
                            {' '}sampled predictions don't share top-level keys with gold
                        </span>
                    </h3>
                    <p className="u-mt0 u-mb-sm">
                        {samplePredictionsResult.metrics.schema_mismatch.hint}
                    </p>
                    <div className="eval-panel__schema-grid">
                        <div>
                            <strong>Gold expects (top key sets):</strong>
                            <ul className="eval-panel__keylist">
                                {samplePredictionsResult.metrics.schema_mismatch.expected_top_keys.map((entry, idx) => (
                                    <li key={`exp-${idx}`}>
                                        <code>{entry.keys.length ? entry.keys.join(', ') : '∅'}</code>
                                        {' '}
                                        <span className="u-text-tertiary">
                                            ({entry.count}×)
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                        <div>
                            <strong>Model emits (top key sets):</strong>
                            <ul className="eval-panel__keylist">
                                {samplePredictionsResult.metrics.schema_mismatch.observed_top_keys.map((entry, idx) => (
                                    <li key={`obs-${idx}`}>
                                        <code>{entry.keys.length ? entry.keys.join(', ') : '∅'}</code>
                                        {' '}
                                        <span className="u-text-tertiary">
                                            ({entry.count}×)
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            )}

            {samplePredictions.length > 0 && wf('failures') && (
                <div className="card eval-sample-predictions">
                    <h3>
                        Sample predictions
                        {samplePredictionsDatasetName && (
                            <span className="eval-sample-predictions__provenance">
                                {' · '}
                                {samplePredictionsDatasetName}
                                {' · '}
                                {samplePredictionsEvalType.replace('_', ' ') || 'eval'}
                                {samplePredictionsResultId != null && (
                                    <span className="eval-sample-predictions__run">
                                        {' '}
                                        (run #{samplePredictionsResultId})
                                    </span>
                                )}
                            </span>
                        )}
                    </h3>
                    <p className="eval-sample-predictions__lede">
                        First {samplePredictions.length} rows from the
                        {samplePredictionsIsLatestRun ? ' most recent ' : ' most recent run that captured predictions (an older run — '}
                        eval{!samplePredictionsIsLatestRun ? '). ' : '. '}
                        When metrics look surprisingly low, eyeball these — a prompt/format mismatch
                        is usually visible at a glance.
                    </p>
                    <div className="table-container">
                        <table className="docs-table eval-sample-predictions-table">
                            <thead>
                                <tr>
                                    <th>Prompt</th>
                                    <th>Expected</th>
                                    <th>Model prediction</th>
                                    {samplePredictionsHasRowScores && <th>Score</th>}
                                </tr>
                            </thead>
                            <tbody>
                                {samplePredictions.map((row, idx) => {
                                    const promptText = String(row.prompt || '').trim();
                                    const referenceText = String(row.reference || '').trim();
                                    const predictionText = String(row.prediction || '').trim();
                                    const formattedPrompt = String(row.formatted_prompt || '').trim();
                                    const promptDiffers =
                                        formattedPrompt && formattedPrompt !== promptText;
                                    const answerSpan = String(row.answer_span || '').trim();
                                    const spanMarker = row.span_marker ?? null;
                                    const spanDiffers =
                                        spanMarker !== null &&
                                        answerSpan.length > 0 &&
                                        answerSpan !== predictionText;
                                    const rowEm = row.row_exact_match;
                                    const rowF1 = row.row_f1;
                                    // Phase 5.3.4: structured extraction rows
                                    // carry these enrichments. is_valid_json
                                    // being a boolean is the signal that the
                                    // run was an extraction (not just QA).
                                    const hasStructured =
                                        row.is_valid_json !== undefined &&
                                        row.is_valid_json !== null;
                                    const isValidJson = Boolean(row.is_valid_json);
                                    const missingRequired = Array.isArray(
                                        row.missing_required_fields,
                                    )
                                        ? row.missing_required_fields
                                        : [];
                                    const rowFieldResults =
                                        row.row_field_results &&
                                        typeof row.row_field_results === 'object'
                                            ? row.row_field_results
                                            : {};
                                    const fieldCount = Object.keys(rowFieldResults).length;
                                    const fieldCorrect = Object.values(rowFieldResults).filter(
                                        (entry) => (entry?.em ?? 0) >= 1,
                                    ).length;
                                    // Phase 5.3.4b: span_set scoring mode swaps the per-field
                                    // comparison disclosure for an entity-by-entity view
                                    // (matched / missed / hallucinated). Triggered when the
                                    // handler stamped scoring_mode === "span_set" on the row.
                                    const isSpanSetMode = row.scoring_mode === 'span_set';
                                    const matchedEntities = Array.isArray(row.row_matched_entities)
                                        ? row.row_matched_entities
                                        : [];
                                    const missedEntities = Array.isArray(row.row_missed_entities)
                                        ? row.row_missed_entities
                                        : [];
                                    const hallucinatedEntities = Array.isArray(
                                        row.row_hallucinated_entities,
                                    )
                                        ? row.row_hallucinated_entities
                                        : [];
                                    const rowPrecision = row.row_precision;
                                    const rowRecall = row.row_recall;
                                    // Phase 5.3.5: RAG handler enrichments.
                                    // We render the RAG surface whenever the
                                    // handler reported context (whether or
                                    // not faithfulness > threshold).
                                    const hasRagContext = Boolean(row.rag_has_context);
                                    const ragContext = String(row.rag_context || '').trim();
                                    const ragFaithfulness = row.rag_faithfulness;
                                    const ragContextRecall = row.rag_context_recall;
                                    const ragUnsupportedRate = row.rag_unsupported_rate;
                                    const ragIsFaithful = row.rag_is_faithful;
                                    // Phase 5.3.6: alignment (DPO/ORPO) per-row diagnostics.
                                    const hasAlignmentPair = Boolean(row.alignment_has_pair);
                                    const alignmentChosen = String(row.alignment_chosen || '').trim();
                                    const alignmentRejected = String(row.alignment_rejected || '').trim();
                                    const alignmentChosenSim = row.alignment_chosen_sim;
                                    const alignmentRejectedSim = row.alignment_rejected_sim;
                                    const alignmentMargin = row.alignment_margin;
                                    const alignmentPreferenceCorrect = row.alignment_preference_correct;
                                    const hasRowScores =
                                        (rowEm !== undefined && rowEm !== null) ||
                                        (rowF1 !== undefined && rowF1 !== null);
                                    // Status: EM=1 → pass, F1≥0.5 → partial, else fail.
                                    let statusClass = '';
                                    let statusLabel = '';
                                    if (hasRowScores) {
                                        if (rowEm === 1 || rowEm === 1.0) {
                                            statusClass = 'is-pass';
                                            statusLabel = 'pass';
                                        } else if (typeof rowF1 === 'number' && rowF1 >= 0.5) {
                                            statusClass = 'is-partial';
                                            statusLabel = 'partial';
                                        } else {
                                            statusClass = 'is-fail';
                                            statusLabel = 'fail';
                                        }
                                    }
                                    return (
                                        <tr key={`sample-prediction-${idx}`}>
                                            <td>
                                                <div className="eval-sample-predictions-cell">
                                                    {promptText || '—'}
                                                </div>
                                                <RowContextChip
                                                    context={row.row_context}
                                                    toolListSource={row.row_tool_list_source ?? null}
                                                    testId={`eval-row-context-${idx}`}
                                                />
                                                {typeof row.row_output_schema_valid === 'boolean' && (
                                                    <div className="u-meta" data-testid={`eval-row-output-schema-${idx}`}>
                                                        {row.row_output_schema_valid
                                                            ? 'Answer matches the row\'s output schema'
                                                            : `Answer does not match the row's output schema: ${(row.row_output_schema_errors ?? []).join('; ')}`}
                                                    </div>
                                                )}
                                                {promptDiffers && (
                                                    <details className="eval-sample-predictions-templated">
                                                        <summary>
                                                            Show templated prompt (what the model
                                                            actually saw)
                                                        </summary>
                                                        <pre>{formattedPrompt}</pre>
                                                    </details>
                                                )}
                                            </td>
                                            <td>
                                                <div className="eval-sample-predictions-cell eval-sample-predictions-cell--reference">
                                                    {referenceText || '—'}
                                                </div>
                                            </td>
                                            <td>
                                                {predictionText ? (
                                                    <div className="eval-sample-predictions-cell eval-sample-predictions-cell--prediction">
                                                        {predictionText}
                                                    </div>
                                                ) : (
                                                    <div className="eval-sample-predictions-cell eval-sample-predictions-cell--prediction-empty">
                                                        (empty output)
                                                    </div>
                                                )}
                                                {spanDiffers && (
                                                    <details className="eval-sample-predictions-span">
                                                        <summary>
                                                            Show extracted answer span (what got
                                                            scored)
                                                        </summary>
                                                        <div className="eval-sample-predictions-span__body">
                                                            <span className="eval-sample-predictions-span__marker">
                                                                marker matched
                                                            </span>
                                                            <code>{answerSpan}</code>
                                                        </div>
                                                    </details>
                                                )}
                                                {!hasStructured && isToolCallRow(row) && (
                                                    <ToolCallRowView row={row} />
                                                )}
                                                {!hasStructured && isSqlExecRow(row) && (
                                                    <SqlExecutionRowView row={row} />
                                                )}
                                                {hasStructured && (
                                                    <div className="eval-sample-predictions-structured">
                                                        <span
                                                            className={`eval-sample-predictions-json-badge ${
                                                                isValidJson
                                                                    ? 'is-valid'
                                                                    : 'is-invalid'
                                                            }`}
                                                        >
                                                            {isValidJson
                                                                ? 'JSON: valid'
                                                                : 'JSON: malformed'}
                                                        </span>
                                                        {!isSpanSetMode && isValidJson && fieldCount > 0 && (
                                                            <span className="eval-sample-predictions-structured__fields">
                                                                {fieldCorrect}/{fieldCount} fields
                                                            </span>
                                                        )}
                                                        {missingRequired.length > 0 && (
                                                            <span className="eval-sample-predictions-structured__missing">
                                                                missing: {missingRequired.join(', ')}
                                                            </span>
                                                        )}
                                                        {isSpanSetMode && (
                                                            <span className="eval-sample-predictions-structured__fields">
                                                                {matchedEntities.length} matched ·{' '}
                                                                {missedEntities.length > 0 ? (
                                                                    <span className="eval-sample-predictions-structured__missing">
                                                                        {missedEntities.length} missed
                                                                    </span>
                                                                ) : (
                                                                    `${missedEntities.length} missed`
                                                                )}
                                                                {' · '}
                                                                {hallucinatedEntities.length > 0 ? (
                                                                    <span className="eval-sample-predictions-structured__missing">
                                                                        {hallucinatedEntities.length} hallucinated
                                                                    </span>
                                                                ) : (
                                                                    `${hallucinatedEntities.length} hallucinated`
                                                                )}
                                                                {typeof rowPrecision === 'number' &&
                                                                    typeof rowRecall === 'number' && (
                                                                        <span className="eval-sample-predictions-status__metrics">
                                                                            P {rowPrecision.toFixed(2)} · R{' '}
                                                                            {rowRecall.toFixed(2)}
                                                                        </span>
                                                                    )}
                                                            </span>
                                                        )}
                                                        {row.scoring_mode === 'finding_set' && (
                                                            <FindingSetRowView row={row} />
                                                        )}
                                                        {isRemediationRow(row) && (
                                                            <RemediationRowView row={row} />
                                                        )}
                                                        {isStructuredPlanRow(row) && (
                                                            <StructuredPlanRowView row={row} />
                                                        )}
                                                        {isMultiOptionRow(row) && (
                                                            <MultiOptionRowView row={row} />
                                                        )}
                                                        {isToolCallRow(row) && (
                                                            <ToolCallRowView row={row} />
                                                        )}
                                                        {isSqlExecRow(row) && (
                                                            <SqlExecutionRowView row={row} />
                                                        )}
                                                        {isSpanSetMode &&
                                                            (matchedEntities.length > 0 ||
                                                                missedEntities.length > 0 ||
                                                                hallucinatedEntities.length > 0) && (
                                                                <details className="eval-sample-predictions-fields">
                                                                    <summary>
                                                                        Show entity-by-entity breakdown
                                                                    </summary>
                                                                    <table className="eval-sample-predictions-fields__table">
                                                                        <thead>
                                                                            <tr>
                                                                                <th>Status</th>
                                                                                <th>Type</th>
                                                                                <th>Text</th>
                                                                                <th>Offset</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody>
                                                                            {matchedEntities.map((e, i) => (
                                                                                <tr key={`m-${i}`}>
                                                                                    <td>
                                                                                        <span className="eval-sample-predictions-fields__em is-pass">
                                                                                            <Check className="u-icon" aria-hidden="true" /> matched
                                                                                        </span>
                                                                                    </td>
                                                                                    <td>
                                                                                        <code>{e.type}</code>
                                                                                    </td>
                                                                                    <td>{e.text || '—'}</td>
                                                                                    <td>
                                                                                        {e.start}–{e.end}
                                                                                    </td>
                                                                                </tr>
                                                                            ))}
                                                                            {missedEntities.map((e, i) => (
                                                                                <tr key={`miss-${i}`}>
                                                                                    <td>
                                                                                        <span className="eval-sample-predictions-fields__em is-fail">
                                                                                            <X className="u-icon" aria-hidden="true" /> missed
                                                                                        </span>
                                                                                    </td>
                                                                                    <td>
                                                                                        <code>{e.type}</code>
                                                                                    </td>
                                                                                    <td>{e.text || '—'}</td>
                                                                                    <td>
                                                                                        {e.start}–{e.end}
                                                                                    </td>
                                                                                </tr>
                                                                            ))}
                                                                            {hallucinatedEntities.map((e, i) => (
                                                                                <tr key={`fp-${i}`}>
                                                                                    <td>
                                                                                        <span className="eval-sample-predictions-fields__em is-fail">
                                                                                            <X className="u-icon" aria-hidden="true" /> hallucinated
                                                                                        </span>
                                                                                    </td>
                                                                                    <td>
                                                                                        <code>{e.type}</code>
                                                                                    </td>
                                                                                    <td>{e.text || '—'}</td>
                                                                                    <td>
                                                                                        {e.start}–{e.end}
                                                                                    </td>
                                                                                </tr>
                                                                            ))}
                                                                        </tbody>
                                                                    </table>
                                                                </details>
                                                            )}
                                                        {!isSpanSetMode && isValidJson && fieldCount > 0 && (
                                                            <details className="eval-sample-predictions-fields">
                                                                <summary>
                                                                    Show field-by-field comparison
                                                                </summary>
                                                                <table className="eval-sample-predictions-fields__table">
                                                                    <thead>
                                                                        <tr>
                                                                            <th>Field</th>
                                                                            <th>Expected</th>
                                                                            <th>Got</th>
                                                                            <th>EM</th>
                                                                        </tr>
                                                                    </thead>
                                                                    <tbody>
                                                                        {Object.entries(rowFieldResults).map(
                                                                            ([fieldName, scores]) => {
                                                                                const expected =
                                                                                    row.parsed_reference &&
                                                                                    typeof row.parsed_reference ===
                                                                                        'object'
                                                                                        ? String(
                                                                                              (
                                                                                                  row.parsed_reference as Record<
                                                                                                      string,
                                                                                                      unknown
                                                                                                  >
                                                                                              )[fieldName] ??
                                                                                                  '',
                                                                                          )
                                                                                        : '';
                                                                                const got =
                                                                                    row.parsed_prediction &&
                                                                                    typeof row.parsed_prediction ===
                                                                                        'object'
                                                                                        ? String(
                                                                                              (
                                                                                                  row.parsed_prediction as Record<
                                                                                                      string,
                                                                                                      unknown
                                                                                                  >
                                                                                              )[fieldName] ??
                                                                                                  '',
                                                                                          )
                                                                                        : '';
                                                                                const em =
                                                                                    typeof scores?.em ===
                                                                                    'number'
                                                                                        ? scores.em
                                                                                        : 0;
                                                                                return (
                                                                                    <tr
                                                                                        key={`field-${fieldName}`}
                                                                                    >
                                                                                        <td>
                                                                                            <code>
                                                                                                {fieldName}
                                                                                            </code>
                                                                                        </td>
                                                                                        <td>{expected || '—'}</td>
                                                                                        <td>{got || '—'}</td>
                                                                                        <td>
                                                                                            <span
                                                                                                className={`eval-sample-predictions-fields__em ${
                                                                                                    em >= 1
                                                                                                        ? 'is-pass'
                                                                                                        : 'is-fail'
                                                                                                }`}
                                                                                            >
                                                                                                {em >= 1 ? <Check className="u-icon" aria-label="match" /> : <X className="u-icon" aria-label="no match" />}
                                                                                            </span>
                                                                                        </td>
                                                                                    </tr>
                                                                                );
                                                                            },
                                                                        )}
                                                                    </tbody>
                                                                </table>
                                                            </details>
                                                        )}
                                                    </div>
                                                )}
                                                {hasRagContext && (
                                                    <div className="eval-sample-predictions-rag">
                                                        {typeof ragFaithfulness === 'number' && (
                                                            <span
                                                                className={`eval-sample-predictions-rag-badge ${
                                                                    ragIsFaithful
                                                                        ? 'is-faithful'
                                                                        : 'is-hallucinated'
                                                                }`}
                                                            >
                                                                {ragIsFaithful
                                                                    ? 'Faithful'
                                                                    : 'Hallucinated'}{' '}
                                                                ({ragFaithfulness.toFixed(2)})
                                                            </span>
                                                        )}
                                                        {typeof ragContextRecall === 'number' && (
                                                            <span className="eval-sample-predictions-rag__metric">
                                                                context covers gold:{' '}
                                                                <strong>
                                                                    {Math.round(
                                                                        ragContextRecall * 100,
                                                                    )}
                                                                    %
                                                                </strong>
                                                            </span>
                                                        )}
                                                        {typeof ragUnsupportedRate === 'number' &&
                                                            ragUnsupportedRate > 0 && (
                                                                <span className="eval-sample-predictions-rag__unsupported">
                                                                    unsupported tokens:{' '}
                                                                    {Math.round(
                                                                        ragUnsupportedRate * 100,
                                                                    )}
                                                                    %
                                                                </span>
                                                            )}
                                                        {ragContext && (
                                                            <details className="eval-sample-predictions-rag-context">
                                                                <summary>
                                                                    Show retrieved context (what the
                                                                    model was given)
                                                                </summary>
                                                                <pre>{ragContext}</pre>
                                                            </details>
                                                        )}
                                                    </div>
                                                )}
                                                {hasAlignmentPair && (
                                                    <div className="eval-sample-predictions-alignment">
                                                        {alignmentPreferenceCorrect !== null &&
                                                            alignmentPreferenceCorrect !== undefined && (
                                                                <span
                                                                    className={`eval-sample-predictions-alignment-badge ${
                                                                        alignmentPreferenceCorrect
                                                                            ? 'is-correct'
                                                                            : 'is-wrong'
                                                                    }`}
                                                                >
                                                                    {alignmentPreferenceCorrect
                                                                        ? 'Preferred chosen'
                                                                        : 'Preferred rejected'}
                                                                </span>
                                                            )}
                                                        {typeof alignmentChosenSim === 'number' &&
                                                            typeof alignmentRejectedSim === 'number' && (
                                                                <span className="eval-sample-predictions-alignment__sims">
                                                                    chosen sim {alignmentChosenSim.toFixed(2)} ·
                                                                    rejected sim{' '}
                                                                    {alignmentRejectedSim.toFixed(2)}
                                                                    {typeof alignmentMargin === 'number' && (
                                                                        <>
                                                                            {' '}· margin{' '}
                                                                            <strong>
                                                                                {alignmentMargin >= 0 ? '+' : ''}
                                                                                {alignmentMargin.toFixed(2)}
                                                                            </strong>
                                                                        </>
                                                                    )}
                                                                </span>
                                                            )}
                                                        {(alignmentChosen || alignmentRejected) && (
                                                            <details className="eval-sample-predictions-alignment-pair">
                                                                <summary>
                                                                    Show chosen vs rejected completions
                                                                </summary>
                                                                <div className="eval-sample-predictions-alignment-pair__body">
                                                                    <div>
                                                                        <span className="eval-sample-predictions-alignment-pair__label is-chosen">
                                                                            chosen
                                                                        </span>
                                                                        <pre>{alignmentChosen || '—'}</pre>
                                                                    </div>
                                                                    <div>
                                                                        <span className="eval-sample-predictions-alignment-pair__label is-rejected">
                                                                            rejected
                                                                        </span>
                                                                        <pre>{alignmentRejected || '—'}</pre>
                                                                    </div>
                                                                </div>
                                                            </details>
                                                        )}
                                                    </div>
                                                )}
                                            </td>
                                            {samplePredictionsHasRowScores && (
                                                <td>
                                                    {hasRowScores ? (
                                                        <div className="eval-sample-predictions-status">
                                                            <span
                                                                className={`eval-sample-predictions-badge ${statusClass}`}
                                                            >
                                                                {statusLabel}
                                                            </span>
                                                            <div className="eval-sample-predictions-status__metrics">
                                                                {typeof rowEm === 'number' && (
                                                                    <span>
                                                                        EM {rowEm.toFixed(0)}
                                                                    </span>
                                                                )}
                                                                {typeof rowF1 === 'number' && (
                                                                    <span>
                                                                        F1 {rowF1.toFixed(2)}
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    ) : (
                                                        <span className="eval-sample-predictions-cell--prediction-empty">
                                                            —
                                                        </span>
                                                    )}
                                                </td>
                                            )}
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {scoredPredictions.length > 0 && wf('failures') && (
                <div className="card">
                    <h3 className="u-card-title">LLM Judge: Side-by-Side Comparison</h3>
                    <div className="u-stack-lg">
                        {scoredPredictions.slice(0, 3).map((prediction, idx) => (
                            <div
                                key={idx}
                                className={`eval-panel__judge-row ${prediction.judge_score >= 4 ? 'eval-panel__judge-row--ok' : 'eval-panel__judge-row--bad'}`}
                            >
                                <div>
                                    <div className="eval-panel__label">Prompt</div>
                                    <div className="eval-panel__box">{prediction.prompt}</div>

                                    <div className="eval-panel__label">Ground Truth (Reference)</div>
                                    <div className="eval-panel__box eval-panel__box--ref">{prediction.reference}</div>
                                </div>

                                <div>
                                    <div className="eval-panel__label">Model Prediction</div>
                                    <div className="eval-panel__box eval-panel__box--pred">{prediction.prediction}</div>
                                </div>

                                <div className="eval-panel__judge-side">
                                    <div className="u-row u-between u-mb-sm">
                                        <span className="eval-panel__label eval-panel__label--inline">Judge Score</span>
                                        <span className={`badge ${prediction.judge_score >= 4 ? 'badge-success' : 'badge-error'} u-text-md`}>
                                            {prediction.judge_score} / 5
                                        </span>
                                    </div>
                                    <div className="u-hint">
                                        <strong>Rationale:</strong> {prediction.judge_rationale}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    {scoredPredictions.length > 3 && (
                        <div className="u-text-center u-mt-lg">
                            <button className="btn btn-secondary btn-sm" disabled>
                                Showing first 3 of {scoredPredictions.length}
                            </button>
                        </div>
                    )}
                </div>
            )}


            {wf('failures') && <EvalGapsPanel projectId={projectId} />}
            {/* Failures › Advanced: tools most users never need on the way to a verdict. keepMounted: the
                rule-set and planner forms keep their state. ⌘K: Results › Failure tools. */}
            {wf('failures') && (
                <AdvancedSection
                    pageKey="results-failures"
                    id="failures-tools"
                    count={10}
                    hint="probe pack, pass/fail rules and slices, remediation planner, behavioral tests, active learning, reroute, drift queue, skill radar, answer-key review and workbench"
                    keepMounted
                    {...sectionRequest('failures-tools')}
                >
                    <div className="u-stack-xl">
                    {wf('failures') && (
                    <ProbePackPanel
                        projectId={projectId}
                        onOpenRun={(expId) => {
                            userPickedRunRef.current = true;
                            void loadResults(expId);
                            // The run's scorecard lives in Verdict › Details: switch tab and open that section
                            // (the panel keeps the picked run across tabs; the section scrolls into view).
                            navigate(resultsPath(projectId, 'verdict', 'open=verdict-details'));
                            setLocalOpen((prev) => ({ id: 'verdict-details', nonce: (prev?.nonce ?? 0) + 1 }));
                        }}
                    />
                    )}
                    {selectedExp && wf('failures') && (
                        <div className="card eval-pack-card" id="eval-pack-editor">
                            <div className="eval-pack-header">
                                <div>
                                    <h3 className="eval-panel__h3 eval-panel__h3--xs">
                                        Pass/fail rules
                                    </h3>
                                    <p className="eval-pack-subtitle">
                                        Choose the project's rule set and review each pass/fail rule for the selected run.
                                    </p>
                                </div>
                                <span className={`badge ${gateStatusBadge.className}`}>{gateStatusBadge.label}</span>
                            </div>

                            <div className="eval-pack-controls">
                                <div className="form-group">
                                    <label className="form-label">Project rule set</label>
                                    <select
                                        className="input"
                                        value={preferredPackId ?? AUTO_PACK_VALUE}
                                        onChange={(e) => onPackPreferenceSelect(e.target.value)}
                                        disabled={loadingPackState || isSavingPackPreference}
                                    >
                                        <option value={AUTO_PACK_VALUE}>Auto (domain profile fallback)</option>
                                        {evaluationPacks.map((pack, index) => (
                                            <option key={`${pack.pack_id}:${index}`} value={pack.pack_id}>
                                                {pack.display_name} ({pack.gate_count} rules)
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div className="eval-pack-meta">
                                    <div>
                                        <span>Active Pack</span>
                                        <strong>{activePackDisplayName}</strong>
                                    </div>
                                    <div>
                                        <span>Source</span>
                                        <strong>{activePackSource.replace(/_/g, ' ')}</strong>
                                    </div>
                                </div>
                            </div>

                            {(loadingPackState || isSavingPackPreference) && (
                                <div className="eval-pack-note">
                                    {loadingPackState ? 'Loading evaluation pack settings...' : 'Saving pack preference...'}
                                </div>
                            )}

                            {/* E5: when the project has no preferred pack set,
                                offer a recipe-derived draft the user can edit
                                inline + save with one click. Stays hidden once
                                a preference (any pack id) is set so the regular
                                pack picker remains the path for users who
                                already chose. */}
                            {!loadingPackState && !preferredPackId && (
                                <EvalPackScaffoldPanel
                                    projectId={projectId}
                                    onSaved={() => {
                                        // Re-fetch the pack preference + gate
                                        // report after save so the surrounding
                                        // card reflects the scaffolded pack
                                        // immediately.
                                        void api
                                            .get(`/projects/${projectId}/evaluation/pack-preference`)
                                            .then((res) => {
                                                setPreferredPackId(
                                                    res.data?.preferred_pack_id ?? null,
                                                );
                                                setActivePackId(res.data?.active_pack_id ?? null);
                                                setActivePackSource(res.data?.active_pack_source ?? 'default');
                                                setActivePackLabel(res.data?.active_pack?.display_name ?? '');
                                            })
                                            .catch(() => {
                                                // Best-effort refresh — the page
                                                // still works if pack-preference
                                                // hiccups.
                                            });
                                    }}
                                />
                            )}

                            {isLoadingGateReport ? (
                                <div className="eval-pack-note">Checking pass/fail rules...</div>
                            ) : gateErrorMessage ? (
                                <div className="eval-pack-error">{gateErrorMessage}</div>
                            ) : gateReport ? (
                                <>
                                    <div className="eval-gate-summary">
                                        <span>
                                            Required rules failed: <strong>{gateReport.failed_gate_ids.length}</strong>
                                        </span>
                                        <span>
                                            Missing required metrics: <strong>{gateReport.missing_required_metrics.length}</strong>
                                        </span>
                                        <span>
                                            Task profile: <strong>{gateReport.task_profile || gateReport.task_profile_selected || 'auto'}</strong>
                                            {gateReport.task_profile_selected && gateReport.task_profile && gateReport.task_profile_selected !== gateReport.task_profile
                                                ? ` (rules written for ${gateReport.task_profile_selected})`
                                                : ''}
                                        </span>
                                        <span>
                                            Checked at: <strong>{new Date(gateReport.captured_at).toLocaleString()}</strong>
                                        </span>
                                    </div>
                                    {!!gateReport.task_profile_source && (
                                        <div className="eval-pack-note">
                                            Resolved from <strong>{gateReport.task_profile_source}</strong>
                                            {gateReport.task_profile_fallback_used ? ' (fallback task spec used)' : ''}
                                        </div>
                                    )}
                                    <div className="table-container eval-gate-table">
                                        <table className="docs-table">
                                            <thead>
                                                <tr>
                                                    <th><Term id="gate" /></th>
                                                    <th>Metric</th>
                                                    <th>Resolved</th>
                                                    <th>Threshold</th>
                                                    <th>Actual</th>
                                                    <th>Status</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {gateReport.checks.map((check) => (
                                                    <tr key={check.gate_id}>
                                                        <td>{check.gate_id}</td>
                                                        <td>{check.metric_id}</td>
                                                        <td>{check.resolved_metric_key || '—'}</td>
                                                        <td>
                                                            {check.threshold == null
                                                                ? '—'
                                                                : `${check.operator === 'lte' ? '≤' : '≥'} ${check.threshold}`}
                                                        </td>
                                                        <td>
                                                            {check.actual == null
                                                                ? '—'
                                                                : typeof check.actual_std === 'number' && (check.actual_n ?? 1) > 1
                                                                ? (
                                                                    <span
                                                                        title={
                                                                            check.variance_policy === 'lower_bound'
                                                                                ? `lower-bound gate value = mean ${check.operator === 'gte' ? '−' : '+'} std`
                                                                                : undefined
                                                                        }
                                                                    >
                                                                        {check.actual.toFixed(3)} ± {check.actual_std.toFixed(3)}
                                                                        {' '}<span className="eval-gate-metric-n">(n={check.actual_n})</span>
                                                                    </span>
                                                                )
                                                                : check.actual}
                                                        </td>
                                                        <td>
                                                            <span className={`badge ${check.passed ? 'badge-success' : 'badge-error'}`}>
                                                                {check.passed
                                                                    ? 'pass'
                                                                    : check.reason === 'variance_below_threshold' || check.reason === 'variance_above_threshold'
                                                                    ? 'fail (variance)'
                                                                    : check.reason}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </>
                            ) : (
                                <div className="eval-pack-note">Select a run to compute gate checks.</div>
                            )}
                        </div>
                    )}
                    {selectedExp && wf('failures') && (
                        <div className="card eval-pack-card">
                            <div className="eval-pack-header">
                                <div>
                                    <h3 className="eval-panel__h3 eval-panel__h3--xs">
                                        Closed-Loop Remediation Planner
                                    </h3>
                                    <p className="eval-pack-subtitle">
                                        Turn evaluation failures into concrete data and training fixes.
                                    </p>
                                </div>
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => {
                                        if (selectedExp) {
                                            void loadRemediationPlans(selectedExp);
                                        }
                                    }}
                                    disabled={isLoadingRemediationPlans || isLoadingRemediationPlan}
                                >
                                    {isLoadingRemediationPlans ? 'Refreshing...' : 'Refresh Plans'}
                                </button>
                            </div>

                            <div className="eval-remediation-controls">
                                <div className="form-group">
                                    <label className="form-label">Evaluation Result Scope</label>
                                    <select
                                        className="input"
                                        value={selectedRemediationEvalResultId}
                                        onChange={(e) => setSelectedRemediationEvalResultId(e.target.value)}
                                    >
                                        <option value="">Auto (latest available result)</option>
                                        {evalResults.map((result) => (
                                            <option key={result.id} value={result.id}>
                                                #{result.id} • {result.dataset_name} • {result.eval_type}
                                                {' '}
                                                {displayPassRate(result) == null ? '' : `(${((displayPassRate(result) ?? 0) * 100).toFixed(1)}%)`}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Failures to Analyze</label>
                                    <input
                                        className="input"
                                        type="number"
                                        min={1}
                                        max={1000}
                                        value={remediationMaxFailures}
                                        onChange={(e) => setRemediationMaxFailures(Number(e.target.value))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Generate Plan</label>
                                    <button
                                        className="btn btn-primary"
                                        onClick={() => void generateRemediationPlan()}
                                        disabled={isGeneratingRemediationPlan || !selectedExp}
                                    >
                                        {isGeneratingRemediationPlan ? 'Generating...' : 'Generate Remediation Plan'}
                                    </button>
                                </div>
                            </div>

                            {remediationErrorMessage && (
                                <div className="eval-pack-error">{remediationErrorMessage}</div>
                            )}

                            {isLoadingRemediationPlans ? (
                                <div className="eval-pack-note">Loading remediation plans...</div>
                            ) : remediationPlans.length === 0 ? (
                                <div className="eval-pack-note">
                                    No remediation plans generated yet for this run.
                                </div>
                            ) : (
                                <div className="eval-remediation-plan-list">
                                    {remediationPlans.map((plan) => (
                                        <button
                                            key={plan.plan_id}
                                            className={`btn btn-sm ${selectedRemediationPlanId === plan.plan_id ? 'btn-primary' : 'btn-secondary'}`}
                                            onClick={() => void loadRemediationPlan(plan.plan_id)}
                                        >
                                            {plan.eval_type || 'evaluation'} • {plan.dataset_name || 'dataset'}
                                            {typeof plan.summary?.recommendation_count === 'number'
                                                ? ` • ${plan.summary.recommendation_count} actions`
                                                : ''}
                                        </button>
                                    ))}
                                </div>
                            )}

                            {isLoadingRemediationPlan && (
                                <div className="eval-pack-note">Loading plan details...</div>
                            )}

                            {!isLoadingRemediationPlan && remediationPlan && (
                                <div className="eval-remediation-details">
                                    <div className="eval-remediation-summary">
                                        <div>
                                            <span>Plan</span>
                                            <strong>{remediationPlan.plan_id}</strong>
                                        </div>
                                        <div>
                                            <span>Failures Analyzed</span>
                                            <strong>{Number(remediationPlan.summary?.total_failures_analyzed || remediationPlan.analysis_evidence?.failures_analyzed || 0)}</strong>
                                        </div>
                                        <div>
                                            <span>Dominant Root Cause</span>
                                            <strong>{String(remediationPlan.summary?.dominant_root_cause || 'unknown').replace(/_/g, ' ')}</strong>
                                        </div>
                                        <div>
                                            <span>Created</span>
                                            <strong>
                                                {remediationPlan.created_at
                                                    ? new Date(remediationPlan.created_at).toLocaleString()
                                                    : '—'}
                                            </strong>
                                        </div>
                                    </div>

                                    {Array.isArray(remediationPlan.clusters) && remediationPlan.clusters.length > 0 && (
                                        <div className="table-container eval-gate-table">
                                            <table className="docs-table">
                                                <thead>
                                                    <tr>
                                                        <th>Root Cause</th>
                                                        <th>Slice</th>
                                                        <th>Failures</th>
                                                        <th>Confidence</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {remediationPlan.clusters.map((cluster, idx) => (
                                                        <tr key={`${cluster.cluster_id || cluster.root_cause || 'cluster'}-${idx}`}>
                                                            <td>{String(cluster.root_cause || cluster.error_type || 'unknown').replace(/_/g, ' ')}</td>
                                                            <td>{cluster.slice || 'general'}</td>
                                                            <td>{Number(cluster.failure_count || 0)}</td>
                                                            <td>
                                                                {typeof cluster.confidence === 'number'
                                                                    ? `${(cluster.confidence * 100).toFixed(0)}%`
                                                                    : '—'}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    )}

                                    {Array.isArray(remediationPlan.recommendations) && remediationPlan.recommendations.length > 0 && (
                                        <div className="eval-remediation-recommendations">
                                            {remediationPlan.recommendations.map((recommendation, idx) => {
                                                const dataOps = remediationDataOps(recommendation);
                                                const trainingChanges = remediationTrainingChanges(recommendation);
                                                return (
                                                    <div
                                                        key={`${recommendation.recommendation_id || recommendation.title || 'recommendation'}-${idx}`}
                                                        className="eval-remediation-recommendation-card"
                                                    >
                                                        <div className="eval-remediation-recommendation-header">
                                                            <strong>{recommendation.title || 'Remediation action'}</strong>
                                                            <span className="badge badge-info">
                                                                {String(recommendation.root_cause || 'coverage_gap').replace(/_/g, ' ')}
                                                            </span>
                                                        </div>
                                                        {typeof recommendation.confidence === 'number' && (
                                                            <div className="eval-pack-note">
                                                                Confidence {(recommendation.confidence * 100).toFixed(0)}%
                                                            </div>
                                                        )}
                                                        {recommendation.rationale && (
                                                            <div className="eval-remediation-rationale">{recommendation.rationale}</div>
                                                        )}
                                                        {dataOps.length > 0 && (
                                                            <div>
                                                                <div className="eval-remediation-list-title">Data Operations</div>
                                                                <ul className="eval-remediation-list">
                                                                    {dataOps.map((item) => <li key={item}>{item}</li>)}
                                                                </ul>
                                                            </div>
                                                        )}
                                                        {trainingChanges.length > 0 && (
                                                            <div>
                                                                <div className="eval-remediation-list-title">Training config changes</div>
                                                                <ul className="eval-remediation-list">
                                                                    {trainingChanges.map((item) => <li key={item}>{item}</li>)}
                                                                </ul>
                                                            </div>
                                                        )}
                                                        {recommendation.expected_impact && (
                                                            <div className="eval-pack-note">
                                                                Expected impact:
                                                                {' '}
                                                                {recommendation.expected_impact.metric || 'score'}
                                                                {typeof recommendation.expected_impact.estimated_delta === 'number'
                                                                    ? ` +${recommendation.expected_impact.estimated_delta.toFixed(3)}`
                                                                    : ''}
                                                                {recommendation.expected_impact.horizon
                                                                    ? ` (${recommendation.expected_impact.horizon})`
                                                                    : ''}
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                    {/* Quality-Lift phase 8 slice 3 — behavioral-results table.
                        Reads ``metrics["behavioral"]`` off each EvalResult so
                        every authored INV/DIR/MFT test gets a pass-rate row
                        even when no gate references it. The ScorecardPanel
                        still owns the *gate* view (failed-row drill-down); the
                        table here is the *test catalog* view (every test,
                        gated or not). One table per EvalResult that carries a
                        behavioral block; silent on rows without it. */}
                    {wf('failures') && evalResults
                        .filter((r) => {
                            const b = (r.metrics as Record<string, unknown>)?.['behavioral'];
                            return typeof b === 'object' && b !== null;
                        })
                        .map((r) => (
                            <BehavioralResultsTable
                                key={`bt-${r.id}`}
                                metrics={r.metrics as Record<string, unknown>}
                            />
                        ))}
                    {selectedExp && evalResults.length > 0 && wf('failures') && (
                        <ActiveLearningPanel
                            projectId={projectId}
                            experimentId={selectedExp}
                            refreshToken={evalResults.length}
                        />
                    )}
                    {/* Phase 7c — post-eval reroute recommendation. Self-hides */}
                    {/* on stay_the_course / 404 / 400. Sits above the */}
                    {/* AutoRagComparisonPanel so the approach-pivot signal is */}
                    {/* the first thing the user reads on a failing eval. */}
                    {wf('failures') && (
                    <RerouteRecommendationPanel
                        projectId={projectId}
                        evalResultId={latestBuiltInResult(evalResults)?.id ?? null}
                    />
                    )}
                    {/* E4 — drift-trap review queue. Mounted under the runs */}
                    {/* sub-tab; self-fetches its own state + recipe so it doesn't */}
                    {/* depend on selectedExp. Opt-in banner explains how to */}
                    {/* enable auto-refresh; manual Generate-now always works. */}
                    {wf('failures') && <DriftReviewQueuePanel projectId={projectId} />}
                        {selectedExp && evalResults.length > 0 && (
                            <div className="card eval-panel__card--col">
                                <h3 className="u-card-title">Skill Radar (LLM Judge)</h3>
                                <div className="eval-panel__radar">
                                    {radarData.length > 0 ? (
                                        <ResponsiveContainer width="100%" height="100%">
                                            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                                                <PolarGrid stroke="var(--chart-grid)" />
                                                <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--chart-axis)', fontSize: 12 }} />
                                                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: 'var(--chart-axis)' }} />
                                                <Radar name="Pass Rate %" dataKey="A" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.5} />
                                            </RadarChart>
                                        </ResponsiveContainer>
                                    ) : (
                                        <div className="eval-panel__radar-empty">
                                            No LLM-judge benchmarks run yet.<br />Score this run with the metric set to LLM Judge (Score this run › Advanced).
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    {wf('failures') && (
                        <details className="card eval-collapsible" data-testid="eval-answer-key-review">
                            <summary>Answer-key review — check the key row by row</summary>
                            <AnswerKeyReviewPanel projectId={projectId} />
                        </details>
                    )}
                    {wf('failures') && (
                        <details className="card eval-collapsible" data-testid="eval-workbench">
                            <summary>Answer-key workbench — suggested rows, coverage, authoring</summary>
                            <GoldSetWorkbenchPanel projectId={projectId} />
                        </details>
                    )}
                    </div>
                </AdvancedSection>
            )}


            </>)}
            </>)}

            {onNextStep && (
                <StepFooter
                    currentStep="Evaluation"
                    nextStep="Ship"
                    nextStepIcon=""
                    isComplete={evalResults.length > 0}
                    hint="Run at least one evaluation to continue"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
