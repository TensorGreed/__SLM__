/**
 * Guided flows — several pipeline steps chained as one background Job.
 * Today: documents → generated Q&A → answer key → split → train → lift check.
 */

import api from './client';
import type { Job } from './jobs';

export interface DocumentsQaFlowPreview {
    project_id: number;
    eligible: boolean;
    blockers: string[];
    passages: number;
    backend: string | null;
    current_recipe_id: string | null;
    already_generated: number;
    plan: {
        max_passages: number;
        pairs_per_passage: number;
        estimated_training_pairs: number;
        estimated_answer_key_rows: number;
        llm_calls: number;
        /** Why that many passages: the test split must let a judge-score
         *  gap of `judge_gap` clear noise (`target_test_rows` paired rows). */
        sized_for?: {
            target_test_rows: number;
            judge_gap: number;
            passages: number;
            passages_needed: number;
            test_ratio: number;
            expected_test_rows: number;
            capped_by_available: boolean;
            reaches_target: boolean;
        } | null;
    };
}

export interface DocumentsQaFlowResult {
    project_id: number;
    passages_total: number;
    passages_used: number;
    passages_failed: number;
    training_pairs: number;
    answer_key_rows: number;
    /** Evaluation questions the quotable gate dropped from the answer key,
     *  by reason: "deictic" (pointed at the passage) / "unsupported" (the
     *  passage doesn't state the answer). */
    eval_rejected?: Record<string, number> | null;
    /** Questions that pointed at the passage and were repaired: "stripped"
     *  (a pure "according to the passage" cut by code) / "rewritten" (by the
     *  generation model). */
    questions_repaired?: Record<string, number> | null;
    /** Training pairs dropped because their question still pointed at the
     *  passage after the rewrite, by reason ("deictic"). */
    pairs_rejected?: Record<string, number> | null;
    backend: string;
    split: { train: number | null; val: number | null; test: number | null; dedup_dropped: number | null } | null;
    experiment_id: number | null;
    stopped_reason: string | null;
    warnings: string[];
    /** The run was trained to answer from the retrieved passages. */
    train_with_passages?: boolean;
    /** The teacher step (answers_from_teacher): per-split counts, the
     *  quality filter and the cloud usage. */
    teacher?: {
        teacher: string;
        usable: number;
        counts: Record<string, {
            rows: number; reused: number; asked: number; answered: number; failed: number; abstained: number; filter_judged: number; filter_dropped: number;
            /** The critique-and-revise loop over the answers the filter dropped (present when rounds > 0). */
            revise?: { attempted: number; fixed: number; still_dropped: number; calls: number };
        }>;
        filter: { applied: boolean; judge?: string; reason?: string; keep?: string };
        /** rounds 0 = off; `critic` = who wrote the critiques (the judge itself when `critic_is_judge`). */
        revise?: { rounds: number; critic?: string; judge?: string; critic_is_judge?: boolean };
    } | null;
    /** The pre-training gate: what the base model + document passages got
     *  right on the test examples, judged, before any training. */
    passages_gate?: {
        status: 'retrieval_ready' | 'retrieval_promising' | 'retrieval_weak' | 'not_judged' | 'not_run' | 'error';
        /** The retrieval the gate judged (the sweep's winner, now the project's setting). */
        retrieval?: { k: number; reranker: string | null; pool?: number | null; chunk_size?: number | null } | null;
        retrieval_sweep?: Array<{ label: string; judge_score: number | null; chosen: boolean }> | null;
        reason?: string | null;
        score?: number | null;
        correct?: number;
        partial?: number;
        wrong?: number;
        judged?: number;
        judge?: string | null;
        split?: string | null;
        /** "val": the sweep chose on validation rows; the gate's numbers are
         *  the chosen retrieval scored once on the test examples. */
        sweep_split?: 'val' | 'test';
        sweep?: { split: string; rows: number | null; chosen_score: number | null } | null;
    } | null;
}

/** `backend`: the generation model picked (`cloud:<provider>:<model>` counts
 *  when the project has that provider's key saved); omit for the local one. */
export async function fetchDocumentsQaFlowPreview(projectId: number, backend?: string | null): Promise<DocumentsQaFlowPreview> {
    const res = await api.get<DocumentsQaFlowPreview>(
        `/projects/${projectId}/flows/documents-to-qa/preview`,
        backend ? { params: { backend } } : undefined,
    );
    return res.data;
}

export async function startDocumentsQaFlow(
    projectId: number,
    options: {
        maxPassages?: number;
        pairsPerPassage?: number;
        train?: boolean;
        reuseExisting?: boolean;
        passagesCheck?: boolean;
        trainIfRetrievalReady?: boolean;
        /** Train the model to answer FROM the retrieved passages (the served
         *  retrieval prompt); its lift check scores it with them against the
         *  base model + the same passages. */
        trainWithPassages?: boolean;
        /** `cloud:<provider>:<model>` (the project's saved key) or a local backend name. */
        backend?: string | null;
        /** Judge-score gap the test examples must resolve (0.15 → 45 rows, 0.08 → 157). */
        decisionGap?: number;
        /** Keep the current test examples verbatim and grow them to the sized target. */
        keepTest?: boolean;
        /** Targets = a teacher model's answers to exactly the served passages. */
        answersFromTeacher?: boolean;
        teacherBackend?: string | null;
        /** Critique-and-revise rounds (0–2) for teacher answers the filter
         *  drops; `teacherCriticBackend` writes the critique (null = the
         *  filter judge's own reason). */
        teacherReviseRounds?: number;
        teacherCriticBackend?: string | null;
        /** Anthropic Message Batches (half price, slower). */
        useBatch?: boolean;
        concurrency?: number;
    } = {},
): Promise<Job> {
    const body: Record<string, unknown> = {};
    if (options.backend) body.backend = options.backend;
    if (options.maxPassages != null) body.max_passages = options.maxPassages;
    if (options.pairsPerPassage != null) body.pairs_per_passage = options.pairsPerPassage;
    if (options.train != null) body.train = options.train;
    if (options.reuseExisting) body.reuse_existing = true;
    if (options.passagesCheck != null) body.passages_check = options.passagesCheck;
    if (options.trainIfRetrievalReady) body.train_if_retrieval_ready = true;
    if (options.trainWithPassages) body.train_with_passages = true;
    if (options.decisionGap != null) body.decision_gap = options.decisionGap;
    if (options.keepTest) body.keep_test = true;
    if (options.answersFromTeacher) {
        body.answers_from_teacher = true;
        if (options.teacherBackend) body.teacher_backend = options.teacherBackend;
        if (options.teacherReviseRounds && options.teacherReviseRounds > 0) {
            body.teacher_revise_rounds = options.teacherReviseRounds;
            if (options.teacherCriticBackend) body.teacher_critic_backend = options.teacherCriticBackend;
        }
    }
    if (options.useBatch) body.use_batch = true;
    if (options.concurrency && options.concurrency > 1) body.concurrency = options.concurrency;
    const res = await api.post<Job>(`/projects/${projectId}/flows/documents-to-qa`, body);
    return res.data;
}


// ── Seed documents: write sample documents for a domain ──────────────────

export interface SeedDocumentsDefaults {
    genres: string[];
    count: number;
    words: number;
}

export async function fetchSeedDocumentsDefaults(projectId: number): Promise<SeedDocumentsDefaults> {
    const res = await api.get<SeedDocumentsDefaults>(`/projects/${projectId}/flows/seed-documents/defaults`);
    return res.data;
}

/** Starts the `seed_documents` Job: `count` invented documents over
 *  `genres`, each containing `kinds`, ingested as one text document and
 *  cleaned into passages. */
export async function startSeedDocuments(
    projectId: number,
    options: {
        kinds?: string; genres?: string[]; count?: number; words?: number; backend?: string | null; domainHint?: string;
        /** "paged": pages with header / footer lines (letterheads, "Page 2 of 5"…). */
        layout?: 'plain' | 'paged' | 'scanned';
        /** Run even though the prompt carries benchmark gold values (the user saw the list). */
        ignoreBenchmarkValues?: boolean;
    } = {},
): Promise<Job> {
    const body: Record<string, unknown> = {};
    if (options.ignoreBenchmarkValues) body.ignore_benchmark_values = true;
    if (options.kinds != null) body.kinds = options.kinds;
    if (options.genres && options.genres.length) body.genres = options.genres;
    if (options.count != null) body.count = options.count;
    if (options.words != null) body.words = options.words;
    if (options.backend) body.backend = options.backend;
    if (options.domainHint) body.domain_hint = options.domainHint;
    if (options.layout && options.layout !== 'plain') body.layout = options.layout;
    const res = await api.post<Job>(`/projects/${projectId}/flows/seed-documents`, body);
    return res.data;
}
