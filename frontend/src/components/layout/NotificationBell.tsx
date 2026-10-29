/**
 * NotificationBell — Hardening Phase H1.
 *
 * Top-bar bell icon + dropdown showing in-flight + recently-
 * completed background jobs (synth playbook runs, RAG clones,
 * future training/eval kicks). Azure-portal pattern: user starts a
 * long-running operation, leaves the page, and gets notified via
 * the bell when it completes.
 *
 * Polling is driven by ``useJobsStore`` — only ticks when there's
 * something to watch or the dropdown is open. Closing/clicking
 * outside the dropdown stops the poll if no job is in-flight.
 */

import { useEffect, useRef, useState } from 'react';

import { benchmarkHeadlineLabel } from '../../api/benchmarks';
import { cancelJob, dismissJob, type Job } from '../../api/jobs';
import { percent, pipelinePath, topFirstWrongStage } from '../../api/pipelines';
import { useJobsStore } from '../../stores/jobsStore';
import { dataStepPath, resultsPath, stagePath, trainSectionPath } from '../../utils/workspaceRoutes';
import { metricDisplayName, shortEvidenceText } from '../evaluation/liftEvidence';
import { toast } from '../../stores/toastStore';
import TrainingKillSwitch from './TrainingKillSwitch';
import TrainingLossSparkline from './TrainingLossSparkline';
import './NotificationBell.css';


/**
 * Navigate via window.location.assign — matches the pattern in
 * RerouteRecommendationPanel and keeps the bell mountable from
 * any context (no Router requirement).
 */
function navigateTo(url: string): void {
    window.location.assign(url);
}


function jobDeepLink(job: Job): string | null {
    // Map each job kind to where clicking "Open" should land.
    if (job.kind === 'reroute_to_rag' && job.status === 'succeeded') {
        const newId = job.result?.new_project_id;
        if (typeof newId === 'number') return `/project/${newId}`;
    }
    if (job.kind === 'synth_playbook' && job.project_id) {
        return dataStepPath(job.project_id, 'synthetic', 'review-queue') + '#synth-review-queue';
    }
    if (job.kind === 'synth_augment_from_cluster' && job.project_id) {
        return dataStepPath(job.project_id, 'synthetic', 'review-queue') + '#synth-review-queue';
    }
    if (job.kind === 'synth_review_assist' && job.project_id) {
        return dataStepPath(job.project_id, 'synthetic', 'review-queue') + '#synth-review-queue';
    }
    // Legacy synth (QA / spans / conversations) all land on the
    // Synthetic tab where the inline review UI lives — the user
    // reviews the generated rows + saves from there.
    if (job.kind.startsWith('synth_legacy') && job.project_id) {
        return stagePath(job.project_id, 'synthetic');
    }
    if (job.kind === 'training_start' && job.project_id) {
        return trainSectionPath(job.project_id, 'plan');
    }
    if (job.kind === 'post_training_lift_eval' && job.project_id) {
        return stagePath(job.project_id, 'eval');
    }
    if (job.kind === 'documents_qa_flow' && job.project_id) {
        return stagePath(job.project_id, 'training') + '#documents-qa-flow';
    }
    if (job.kind === 'seed_documents' && job.project_id) {
        return stagePath(job.project_id, 'data') + '#seed-documents';
    }
    if ((job.kind === 'document_source_web_crawl' || job.kind === 'document_source_git_repo') && job.project_id) {
        return stagePath(job.project_id, 'data') + '#document-sources';
    }
    if (job.kind === 'tool_catalog_import' && job.project_id) {
        return stagePath(job.project_id, 'goldset') + '#tool-catalog';
    }
    if (job.kind === 'span_relabel' && job.project_id) {
        return stagePath(job.project_id, 'eval');
    }
    if (job.kind === 'benchmark_eval' && job.project_id) {
        return resultsPath(job.project_id, 'compare') + '#external-benchmarks';
    }
    if (job.kind === 'pipeline_eval') {
        // Project-less: the run lives under /pipelines, not a project.
        const pid = job.params?.pipeline_id ?? job.result?.pipeline_id;
        const rid = job.params?.run_id ?? job.result?.run_id;
        if (typeof pid === 'string' && pid) return pipelinePath(pid, typeof rid === 'string' ? rid : undefined);
        return null;
    }
    if (job.project_id) {
        return `/project/${job.project_id}`;
    }
    return null;
}


/** A running Job the command backend has parked until its usage window reopens. */
export function isPausedJob(job: Job): boolean {
    return job.status === 'running' && (job.progress_message ?? '').toLowerCase().startsWith('waiting for usage window');
}

function jobStatusLabel(job: Job): string {
    switch (job.status) {
        case 'queued': return 'Queued';
        case 'running': return job.progress_message || 'Running…';
        case 'succeeded': return 'Done';
        case 'failed': return 'Failed';
        case 'cancelled': return 'Cancelled';
        default: return job.status;
    }
}


/**
 * Per-kind one-line outcome summary for a terminal job. Pulls
 * fields out of ``job.result`` and formats them so the bell row
 * tells the user WHAT happened, not just THAT it finished.
 * Returns null when the job has no useful summary (caller hides
 * the line entirely).
 */
export function jobOutcomeSummary(job: Job): string | null {
    const core = jobOutcomeSummaryCore(job);
    // Any kind: the jobs framework meters every runner's cloud-LLM calls
    // into result.tokens / cost_usd — also on a failed or cancelled Job
    // (tokens spent are spent).
    const spend = cloudSpendText((job.result || {}) as Record<string, unknown>);
    if (!spend) return core;
    if (job.status === 'succeeded') return core ? `${core} · ${spend}` : spend;
    if (job.status === 'failed' || job.status === 'cancelled') return `spent ${spend}`;
    return core;
}

/** "12k tokens (~$0.04 at list price)" from a Job result that carries
 *  cloud-LLM usage, else null. A result whose calls went through a Message
 *  Batch (``batch`` set) is priced at the batch rate: "… with Message Batch
 *  pricing" (any calls the batch didn't answer are still at list price). */
export function cloudSpendText(r: Record<string, unknown>): string | null {
    if (typeof r.tokens !== 'number' || r.tokens <= 0) return null;
    const k = r.tokens >= 1000 ? `${Math.round(r.tokens / 1000)}k` : String(r.tokens);
    const priced = r.batch && typeof r.batch === 'object' ? 'with Message Batch pricing' : 'at list price';
    return typeof r.cost_usd === 'number' ? `${k} tokens (~$${r.cost_usd.toFixed(2)} ${priced})` : `${k} tokens`;
}

/** "2000 of 17798 chunks (sampled) · sources #3 1200, #5 800 · python 900, go 1100" from a
 *  playbook Job result's `seed_pool`, else null. Source ids (not titles) are what the result
 *  carries; they match the ledger's "source #id". */
export function seedPoolText(raw: unknown): string | null {
    if (!raw || typeof raw !== 'object') return null;
    const p = raw as Record<string, unknown>;
    if (typeof p.chunks_used !== 'number' || typeof p.chunks_eligible !== 'number') return null;
    const counts = (m: unknown, prefix: string) =>
        m && typeof m === 'object'
            ? Object.entries(m as Record<string, number>).map(([k, n]) => (k === 'unknown' ? `${k} ${n}` : `${prefix}${k} ${n}`)).join(', ')
            : '';
    const parts = [`${p.chunks_used} of ${p.chunks_eligible} chunks${p.sampled ? ' (sampled)' : ''}`];
    const bySource = counts(p.by_source, '#');
    if (bySource) parts.push(`sources ${bySource}`);
    const byLang = counts(p.by_language, '');
    if (byLang) parts.push(byLang);
    return parts.join(' \u00b7 ');
}

function jobOutcomeSummaryCore(job: Job): string | null {
    if (job.status !== 'succeeded') return null;
    const r = (job.result || {}) as Record<string, unknown>;

    if (job.kind === 'synth_playbook' || job.kind === 'synth_augment_from_cluster') {
        const rows = typeof r.rows_generated === 'number' ? r.rows_generated : null;
        const backend = typeof r.backend_used === 'string' ? r.backend_used : null;
        const elapsed = typeof r.elapsed_sec === 'number' ? r.elapsed_sec : null;
        const clusterId =
            typeof r.cluster_id === 'string' && r.cluster_id.length > 0
                ? r.cluster_id
                : null;
        const parts: string[] = [];
        if (rows !== null) parts.push(`${rows} row${rows === 1 ? '' : 's'} generated`);
        if (clusterId) parts.push(`cluster ${clusterId.slice(0, 16)}`);
        if (backend) parts.push(`via ${backend}`);
        if (elapsed !== null) parts.push(`${elapsed.toFixed(1)}s`);
        const rejected = typeof r.rejected_count === 'number' ? r.rejected_count : 0;
        if (rejected > 0) parts.push(`${rejected} rejected`);
        const pool = seedPoolText(r.seed_pool);
        if (pool) parts.push(`seeds: ${pool}`);
        const batch = r.batch as { fell_back?: number } | null | undefined;
        if (batch && typeof batch === 'object') {
            const redone = typeof batch.fell_back === 'number' && batch.fell_back > 0 ? batch.fell_back : 0;
            parts.push(`via batch (half price)${redone ? `, ${redone} request${redone === 1 ? '' : 's'} redone directly` : ''}`);
        }
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind === 'synth_review_assist') {
        const v = (r.verdicts || {}) as Record<string, unknown>;
        const n = (k: string) => (typeof v[k] === 'number' ? (v[k] as number) : 0);
        const parts = [`pass ${n('pass')}`, `fail ${n('fail')}`, `unsure ${n('unsure')}`];
        const unparseable = typeof r.rows_unparseable === 'number' ? r.rows_unparseable : 0;
        if (unparseable > 0) parts.push(`${unparseable} unreadable`);
        // Two or more judges on the same rows: where they agree or disagree.
        const c = (r.consensus || {}) as Record<string, unknown>;
        const cn = (k: string) => (typeof c[k] === 'number' ? (c[k] as number) : 0);
        const agree = cn('agree_pass') + cn('agree_fail');
        if (agree + cn('disagree') > 0) parts.push(`agree ${agree} \u00b7 disagree ${cn('disagree')}`);
        const batch = r.batch as { fell_back?: number } | null | undefined;
        if (batch && typeof batch === 'object') {
            const redone = typeof batch.fell_back === 'number' && batch.fell_back > 0 ? batch.fell_back : 0;
            parts.push(`via batch (half price)${redone ? `, ${redone} request${redone === 1 ? '' : 's'} redone directly` : ''}`);
        }
        return parts.join(' · ');
    }

    if (job.kind.startsWith('synth_legacy')) {
        const rowsSaved = typeof r.rows_saved === 'number' ? r.rows_saved : null;
        const rowsGenerated = typeof r.rows_generated === 'number' ? r.rows_generated : null;
        const batches = typeof r.batches_done === 'number' ? r.batches_done : null;
        const total = typeof r.batches_total === 'number' ? r.batches_total : null;
        const parts: string[] = [];
        // Prefer rows_saved (the persisted count) — that's the
        // number the user actually cares about. Fall back to
        // rows_generated for older jobs from before auto-save shipped.
        if (rowsSaved !== null) {
            parts.push(`${rowsSaved} row${rowsSaved === 1 ? '' : 's'} saved to dataset`);
        } else if (rowsGenerated !== null) {
            parts.push(`${rowsGenerated} row${rowsGenerated === 1 ? '' : 's'} generated`);
        }
        if (batches !== null && total !== null) parts.push(`${batches}/${total} batches`);
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind === 'reroute_to_rag') {
        const newId = typeof r.new_project_id === 'number' ? r.new_project_id : null;
        const newName = typeof r.new_project_name === 'string' ? r.new_project_name : null;
        if (newId !== null) {
            return newName
                ? `Created "${newName}" (project #${newId})`
                : `Created project #${newId}`;
        }
        return null;
    }

    if (job.kind === 'pipeline_eval') {
        const rows = typeof r.rows === 'number' ? r.rows : null;
        if (rows === null) return null;
        const share = typeof r.share_fully_correct === 'number' ? r.share_fully_correct : null;
        const attribution = r.attribution as { first_wrong_stage?: Record<string, number> } | undefined;
        const top = topFirstWrongStage(attribution?.first_wrong_stage);
        const parts = [`${rows} row${rows === 1 ? '' : 's'}`];
        if (share !== null) parts.push(`${percent(share)} fully correct`);
        if (top) parts.push(`first-wrong: ${top}`);
        return parts.join(' · ');
    }

    if (job.kind === 'tool_catalog_import') {
        // The runner returns the catalog status ({catalog: {tools, version, source}}); flat fields are accepted too.
        const cat = (r.catalog && typeof r.catalog === 'object' ? r.catalog : r) as Record<string, unknown>;
        const n = Array.isArray(cat.tools) ? cat.tools.length : typeof r.tool_count === 'number' ? r.tool_count : null;
        if (n === null) return null;
        const version = typeof cat.version === 'string' && cat.version ? ` (catalog version ${cat.version})` : '';
        const srcKind = (cat.source as { kind?: unknown } | undefined)?.kind ?? r.source_kind;
        const source = typeof srcKind === 'string' && srcKind ? ` from ${srcKind}` : '';
        return `${n} tool${n === 1 ? '' : 's'} imported${source}${version}`;
    }

    if (job.kind === 'benchmark_eval') {
        const score = typeof r.score === 'number' ? r.score : null;
        if (score === null) return null;
        const metric = benchmarkHeadlineLabel(typeof r.headline === 'string' ? r.headline : null);
        const rows = typeof r.rows === 'number' ? ` on ${r.rows} rows` : '';
        return `${String(r.benchmark ?? 'benchmark')}: ${metric} ${score.toFixed(2)}${rows}`;
    }

    if (job.kind === 'span_relabel') {
        const rows = typeof r.labelled_rows === 'number' ? r.labelled_rows : null;
        const entities = typeof r.entities === 'number' ? r.entities : null;
        const agreement = r.agreement && typeof (r.agreement as { f1?: unknown }).f1 === 'number' ? (r.agreement as { f1: number }).f1 : null;
        const parts: string[] = [];
        const isTraining = r.split === 'synthetic';
        const batch = r.batch as { fell_back?: number } | null | undefined;
        const viaBatch = r.via_batch === true
            ? `via batch (half price)${typeof batch?.fell_back === 'number' && batch.fell_back > 0 ? `, ${batch.fell_back} request${batch.fell_back === 1 ? '' : 's'} redone directly` : ''}`
            : null;
        if (job.title?.includes('split relabelled') && rows !== null) {
            return `${String(r.split)} split relabelled in place: ${rows} rows`
                + (agreement !== null ? ` · agreement with the old labels F1 ${agreement.toFixed(2)}` : '')
                + (viaBatch ? ` · ${viaBatch}` : '');
        }
        const isPassages = r.split === 'passages' || r.split === 'documents';
        if (rows !== null) {
            parts.push(
                isPassages ? `${rows} ${r.split === 'documents' ? 'documents' : 'passages'} labelled — accept them in the review queue`
                    : isTraining ? `training labels: ${rows} rows relabelled` : `answer key: ${rows} rows`,
            );
        }
        if (entities !== null) parts.push(`${entities} entities`);
        if (agreement !== null) parts.push(`agreement with the old labels F1 ${agreement.toFixed(2)}`);
        const review = r.review as { reviewed?: number; disagreed?: number; rows?: number; review_backend?: string; skipped?: string } | null | undefined;
        if (review?.skipped) parts.push(`no review pass (${review.skipped})`);
        else if (review && typeof review.disagreed === 'number') {
            parts.push(`${review.review_backend ?? 'review model'} checked ${review.disagreed} of ${review.rows} rows (disagreements)`);
        }
        const targeting = r.targeting as { affected_rows?: number; carried_forward?: number } | null | undefined;
        if (targeting && typeof targeting.carried_forward === 'number' && targeting.carried_forward > 0) {
            parts.push(`${targeting.carried_forward} rows untouched by the change carried forward`);
        }
        if (viaBatch) parts.push(viaBatch);
        // Spend is appended by jobOutcomeSummary (every kind).
        return parts.length ? parts.join(' · ') : null;
    }
    if (job.kind === 'seed_documents') {
        const written = typeof r.written === 'number' ? r.written : null;
        const passages = typeof r.passages === 'number' ? r.passages : null;
        const parts: string[] = [];
        if (written !== null) parts.push(`${written} sample documents written`);
        if (passages !== null) parts.push(`${passages} passages`);
        return parts.length ? parts.join(' · ') : null;
    }
    if (job.kind === 'document_source_web_crawl' || job.kind === 'document_source_git_repo') {
        const made = typeof r.documents_created === 'number' ? r.documents_created : null;
        // ``rejected`` is capped at 500 rows; ``rejection_counts`` is the full tally.
        const counts = r.rejection_counts && typeof r.rejection_counts === 'object'
            ? Object.values(r.rejection_counts as Record<string, unknown>)
                .reduce<number>((sum, n) => sum + (typeof n === 'number' ? n : 0), 0)
            : null;
        const rej = counts !== null ? counts : Array.isArray(r.rejected) ? r.rejected.length : null;
        const parts: string[] = [];
        if (made !== null) parts.push(`${made} documents imported`);
        if (rej !== null) parts.push(`${rej} rejected`);
        return parts.length ? parts.join(', ') : null;
    }
    if (job.kind === 'documents_qa_flow') {
        const pairs = typeof r.training_pairs === 'number' ? r.training_pairs : null;
        const keyRows = typeof r.answer_key_rows === 'number' ? r.answer_key_rows : null;
        const exp = typeof r.experiment_id === 'number' ? r.experiment_id : null;
        const stopped = typeof r.stopped_reason === 'string' ? r.stopped_reason : null;
        const parts: string[] = [];
        if (pairs !== null) parts.push(`${pairs} Q&A pairs`);
        if (keyRows !== null) parts.push(`${keyRows} answer-key rows`);
        if (exp !== null) parts.push(`training run #${exp} started`);
        else if (stopped) parts.push(stopped);
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind === 'post_training_lift_eval') {
        if (r.kind === 'span_eval') {
            // Token-classification run: its own span scores, the answer key first.
            const blocks = [['answer key', r.answer_key], ['test examples', r.test]] as const;
            const parts = blocks
                .filter(([, b]) => b && typeof (b as { f1?: unknown }).f1 === 'number')
                .map(([label, b]) => `span F1 ${(b as { f1: number }).f1.toFixed(2)} on the ${label}`);
            return parts.length ? parts.join(' · ') : 'span evaluation done';
        }
        if (r.kind === 'passage_reader_eval') {
            // Trained to answer from the passages: both arms WITH the same passages.
            const base = (r.base_with_passages || {}) as { score?: number };
            const ft = (r.finetune_with_passages || {}) as { score?: number };
            const ev = (r.evidence || {}) as { verdict?: string; better?: number; worse?: number };
            if (typeof base.score !== 'number' || typeof ft.score !== 'number') return 'passages check done';
            const verb = ev.verdict === 'better' ? 'better' : ev.verdict === 'worse' ? 'worse' : ev.verdict === 'too_few_rows' ? 'too few rows to tell' : 'within noise';
            const counts = typeof ev.better === 'number' && typeof ev.worse === 'number' ? ` · ${ev.better} better, ${ev.worse} worse` : '';
            return `judge score with passages: base ${base.score.toFixed(3)} → fine-tuned ${ft.score.toFixed(3)} (${verb}${counts})`;
        }
        if (r.kind === 'head_eval') {
            // Classifier head on an encoder: scored alone (no zero-shot base model to compare with).
            const acc = typeof r.accuracy === 'number' ? `accuracy ${r.accuracy.toFixed(2)}` : null;
            const f1 = typeof r.macro_f1 === 'number' ? `macro-F1 ${r.macro_f1.toFixed(2)}` : null;
            const parts = [acc, f1].filter(Boolean);
            return parts.length ? `${parts.join(' · ')} on the test examples (no base-model baseline for an encoder)` : 'evaluation done';
        }
        const h = (r.headline || null) as {
            metric_id?: string;
            baseline_value?: number;
            trained_value?: number;
            direction?: string;
        } | null;
        if (h && typeof h.baseline_value === 'number' && typeof h.trained_value === 'number') {
            // Row-level evidence, when the job recorded it: a change that is
            // within noise is not announced as better / worse than base.
            const ev = (r.evidence || null) as {
                verdict?: string;
                better?: number;
                worse?: number;
            } | null;
            const counts = ev && typeof ev.better === 'number' && typeof ev.worse === 'number'
                ? ` · ${ev.better} row${ev.better === 1 ? '' : 's'} better, ${ev.worse} worse`
                : '';
            let verb = h.direction === 'improved'
                ? 'better than base'
                : h.direction === 'regressed' ? 'worse than base' : 'same as base';
            if (ev?.verdict === 'within_noise') verb = 'within noise';
            else if (ev?.verdict === 'too_few_rows') verb = 'too few rows to tell';
            // Multi-seed run: the headline is a mean; say whether the seeds agree.
            const seeds = (r.seed_evidence || null) as { verdict?: string; n?: number; std?: number | null } | null;
            if (seeds && typeof seeds.n === 'number') {
                const spread = typeof seeds.std === 'number' ? ` ± ${seeds.std.toFixed(3)}` : '';
                const agree = seeds.verdict === 'within_noise'
                    ? 'seeds disagree'
                    : seeds.verdict === 'too_few_seeds'
                        ? 'one seed'
                        : seeds.verdict === 'worse' ? 'drop holds across seeds' : 'holds across seeds';
                return `${metricDisplayName(h.metric_id)}: ${h.baseline_value.toFixed(3)} → ${h.trained_value.toFixed(3)}${spread} across ${seeds.n} seeds (${agree})`;
            }
            return `${metricDisplayName(h.metric_id)}: ${h.baseline_value.toFixed(3)} → ${h.trained_value.toFixed(3)} (${verb}${counts})`;
        }
        return typeof r.lift_status === 'string' ? `lift: ${r.lift_status}` : null;
    }

    if (job.kind === 'training_start') {
        const loss = typeof r.final_train_loss === 'number' ? r.final_train_loss : null;
        const steps = typeof r.total_steps === 'number' ? r.total_steps : null;
        const terminal = typeof r.terminal_status === 'string' ? r.terminal_status : null;
        const parts: string[] = [];
        if (terminal && terminal !== 'completed') {
            parts.push(terminal);
        }
        if (loss !== null) parts.push(`final loss ${loss.toFixed(4)}`);
        if (steps !== null) parts.push(`${steps} steps`);
        return parts.length ? parts.join(' · ') : null;
    }

    if (job.kind === 'auto_rag_comparison') {
        const offF1 = typeof r.off_mean_f1 === 'number' ? r.off_mean_f1 : null;
        const onF1 = typeof r.on_mean_f1 === 'number' ? r.on_mean_f1 : null;
        const lift = typeof r.relative_lift_pct === 'number' ? r.relative_lift_pct : null;
        const nVal = typeof r.n_val_rows === 'number' ? r.n_val_rows : null;
        const parts: string[] = [];
        // Which model was scored: the base, or the fine-tuned run (a request
        // can pick an older run than the latest, so name it).
        if (r.model === 'base') parts.push('base model');
        else if (typeof r.experiment_id === 'number') parts.push(`run #${r.experiment_id}`);
        if (offF1 !== null && onF1 !== null) {
            parts.push(`off F1 ${offF1.toFixed(2)} → on F1 ${onF1.toFixed(2)}`);
        } else if (onF1 !== null) {
            parts.push(`on F1 ${onF1.toFixed(2)}`);
        }
        if (lift !== null) {
            const sign = lift >= 0 ? '+' : '';
            // Row-level evidence, when the job recorded it: a lift on a few
            // rows is qualified right next to the number.
            const ev = (r.evidence || null) as { verdict?: string; better?: number; worse?: number } | null;
            const note = ev && typeof ev.better === 'number' && typeof ev.worse === 'number'
                && (ev.verdict === 'within_noise' || ev.verdict === 'too_few_rows' || ev.verdict === 'better' || ev.verdict === 'worse')
                ? ` (${shortEvidenceText({ verdict: ev.verdict, better: ev.better, worse: ev.worse })})`
                : '';
            parts.push(`${sign}${lift.toFixed(2)}% lift${note}`);
        }
        if (nVal !== null) {
            const split = r.split === 'test' ? 'test' : 'val';
            parts.push(`${nVal} ${split} row${nVal === 1 ? '' : 's'}`);
        }
        return parts.length ? parts.join(' · ') : null;
    }

    return null;
}


export default function NotificationBell() {
    const { jobs, isPolling, bellOpen, setBellOpen, refreshAfterLocalChange } =
        useJobsStore();
    const [, setTick] = useState(0);  // forces re-render on dismissals
    const wrapperRef = useRef<HTMLDivElement>(null);

    // Mount → start polling so the bell starts fresh when the app
    // boots and any jobs already in-flight show up.
    useEffect(() => {
        void useJobsStore.getState().fetchOnce();
        return () => {
            // Component unmount: leave polling alone — store decides.
        };
    }, []);

    // Close dropdown on outside-click.
    useEffect(() => {
        if (!bellOpen) return;
        const onDocClick = (ev: MouseEvent) => {
            const target = ev.target as Node | null;
            if (wrapperRef.current && target && !wrapperRef.current.contains(target)) {
                setBellOpen(false);
            }
        };
        document.addEventListener('mousedown', onDocClick);
        return () => document.removeEventListener('mousedown', onDocClick);
    }, [bellOpen, setBellOpen]);

    const inFlight = jobs.filter(
        (j) => j.status === 'queued' || j.status === 'running',
    );
    const recentlyDone = jobs.filter(
        (j) => j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled',
    );

    const badgeCount = inFlight.length;
    // A failed job that hasn't been dismissed shows as a red badge
    // even when no jobs are currently running — same urgency
    // signal as an Azure-portal alert.
    const hasUnacknowledgedFailure = recentlyDone.some((j) => j.status === 'failed');

    const handleDismiss = async (job: Job) => {
        try {
            await dismissJob(job.id);
            await refreshAfterLocalChange();
            setTick((t) => t + 1);
        } catch (err) {
            const msg =
                (err as { response?: { data?: { detail?: string } } })?.response
                    ?.data?.detail || 'Dismiss failed';
            toast.error(msg);
        }
    };

    const handleCancel = async (job: Job) => {
        try {
            await cancelJob(job.id);
            toast.info(`Cancelled "${job.title}"`, 3000);
            await refreshAfterLocalChange();
            setTick((t) => t + 1);
        } catch (err) {
            const msg =
                (err as { response?: { data?: { detail?: string } } })?.response
                    ?.data?.detail || 'Cancel failed';
            toast.error(msg);
        }
    };

    const handleOpen = (job: Job) => {
        const url = jobDeepLink(job);
        if (url) {
            setBellOpen(false);
            navigateTo(url);
        }
    };

    return (
        <div
            ref={wrapperRef}
            className="notif-bell"
            data-testid="notification-bell"
        >
            <button
                type="button"
                className={
                    'notif-bell__btn'
                    + (badgeCount > 0 ? ' has-active' : '')
                    + (hasUnacknowledgedFailure ? ' has-failure' : '')
                }
                onClick={() => setBellOpen(!bellOpen)}
                aria-label={
                    badgeCount > 0
                        ? `${badgeCount} background jobs running`
                        : 'Notifications'
                }
                data-testid="notification-bell-button"
            >
                {/* Inline SVG bell so we don't need an icon library. */}
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                    <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                </svg>
                {badgeCount > 0 && (
                    <span
                        className="notif-bell__badge"
                        data-testid="notification-bell-badge"
                    >
                        {badgeCount}
                    </span>
                )}
                {badgeCount === 0 && hasUnacknowledgedFailure && (
                    <span
                        className="notif-bell__badge notif-bell__badge--failure"
                        data-testid="notification-bell-badge-failure"
                    >
                        !
                    </span>
                )}
            </button>

            {bellOpen && (
                <div
                    className="notif-bell__panel"
                    data-testid="notification-bell-panel"
                    role="dialog"
                    aria-label="Background jobs"
                >
                    <div className="notif-bell__head">
                        <h3>Background jobs</h3>
                        <span className="notif-bell__head-meta">
                            {isPolling ? 'live' : 'idle'}
                        </span>
                    </div>

                    {jobs.length === 0 && (
                        <p
                            className="notif-bell__empty"
                            data-testid="notification-bell-empty"
                        >
                            No background jobs right now. Long-running
                            operations like synth runs and RAG clones will
                            show up here.
                        </p>
                    )}

                    {inFlight.length > 0 && (
                        <section className="notif-bell__group">
                            <div className="notif-bell__group-head">
                                In progress ({inFlight.length})
                            </div>
                            {inFlight.map((job) => (
                                <JobRow
                                    key={job.id}
                                    job={job}
                                    onOpen={handleOpen}
                                    onDismiss={handleDismiss}
                                    onCancel={handleCancel}
                                />
                            ))}
                        </section>
                    )}

                    {recentlyDone.length > 0 && (
                        <section className="notif-bell__group">
                            <div className="notif-bell__group-head">
                                Recently completed
                            </div>
                            {recentlyDone.map((job) => (
                                <JobRow
                                    key={job.id}
                                    job={job}
                                    onOpen={handleOpen}
                                    onDismiss={handleDismiss}
                                />
                            ))}
                        </section>
                    )}
                </div>
            )}
        </div>
    );
}


interface JobRowProps {
    job: Job;
    onOpen: (job: Job) => void;
    onDismiss: (job: Job) => void;
    /** Optional — only passed for in-flight jobs by the parent. */
    onCancel?: (job: Job) => void;
}


function JobRow({ job, onOpen, onDismiss, onCancel }: JobRowProps) {
    const isInFlight = job.status === 'queued' || job.status === 'running';
    const link = jobDeepLink(job);
    return (
        <div
            className={`notif-bell__row notif-bell__row--${job.status}`}
            data-testid={`notification-bell-row-${job.id}`}
        >
            <div className="notif-bell__row-main">
                <div className="notif-bell__row-title">{job.title}</div>
                <div className="notif-bell__row-status">
                    {isPausedJob(job) && (
                        <span className="notif-bell__chip notif-bell__chip--paused" data-testid={`notification-bell-row-${job.id}-paused`}>
                            paused
                        </span>
                    )}
                    {jobStatusLabel(job)}
                </div>
                {(() => {
                    const summary = jobOutcomeSummary(job);
                    return summary ? (
                        <div
                            className="notif-bell__row-summary"
                            data-testid={`notification-bell-row-${job.id}-summary`}
                        >
                            {summary}
                        </div>
                    ) : null;
                })()}
                {job.status === 'failed' && job.error && (
                    <div
                        className="notif-bell__row-error"
                        data-testid={`notification-bell-row-${job.id}-error`}
                    >
                        {job.error}
                    </div>
                )}
                {isInFlight && typeof job.progress === 'number' && (() => {
                    // role="progressbar" with dynamic aria-valuenow trips
                    // a strict-mode ARIA validator that wants literal
                    // string values; aria-label conveys the same info
                    // for screen readers and keeps the lint clean.
                    // We snap the fill width to nearest-10% (defined as
                    // CSS classes) so we don't need inline styles —
                    // strict CSS lint also wants those off.
                    const pct = Math.max(0, Math.min(100, Math.round(job.progress * 100)));
                    const bucket = Math.round(pct / 10) * 10;
                    return (
                        <div
                            className="notif-bell__progress"
                            aria-label={`${pct}% complete`}
                        >
                            <div
                                className={`notif-bell__progress-fill notif-bell__progress-fill--w-${bucket}`}
                            />
                        </div>
                    );
                })()}
                {isInFlight
                    && job.kind === 'training_start'
                    && Array.isArray(job.metrics_recent) && (
                    <div
                        className="notif-bell__loss-sparkline"
                        data-testid={`notification-bell-row-${job.id}-sparkline`}
                    >
                        <TrainingLossSparkline points={job.metrics_recent} />
                    </div>
                )}
                {isInFlight && job.kind === 'training_start' && (
                    <TrainingKillSwitch job={job} />
                )}
            </div>
            <div className="notif-bell__row-actions">
                {link && (
                    <button
                        type="button"
                        className="notif-bell__row-btn"
                        onClick={() => onOpen(job)}
                        data-testid={`notification-bell-row-${job.id}-open`}
                    >
                        Open
                    </button>
                )}
                {!isInFlight && (
                    <button
                        type="button"
                        className="notif-bell__row-btn notif-bell__row-btn--ghost"
                        onClick={() => onDismiss(job)}
                        data-testid={`notification-bell-row-${job.id}-dismiss`}
                    >
                        Dismiss
                    </button>
                )}
                {isInFlight && onCancel && (
                    <button
                        type="button"
                        className="notif-bell__row-btn notif-bell__row-btn--danger"
                        onClick={() => onCancel(job)}
                        data-testid={`notification-bell-row-${job.id}-cancel`}
                        title="Stop showing this job. The underlying work may still finish server-side until the runner honors the cancel flag."
                    >
                        Cancel
                    </button>
                )}
            </div>
        </div>
    );
}
