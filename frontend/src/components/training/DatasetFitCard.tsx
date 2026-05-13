/**
 * DatasetFitCard — "Why this dataset isn't ready for SFT" explainer
 * (newbie UX Phase 5.1).
 *
 * Replaces the raw `Dataset contract mismatch for task_type=causal_lm:
 * compatible coverage 0.0% is below required 90.0%` error with a
 * plain-language card that names the rows we actually saw next to the
 * shape the chosen task needs, and offers concrete "click here to
 * unblock yourself" branches.
 *
 * All data already comes back inside
 * ``preflight.capability_summary.dataset.contract`` — this component
 * is purely a renderer.
 */

import { Link } from 'react-router-dom';
import { CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';

import './DatasetFitCard.css';

interface DatasetContract {
    task_type?: string;
    required_shapes?: string[];
    shape_counts?: Record<string, number>;
    sampled_rows?: number;
    compatible_rows?: number;
    coverage?: number;
    errors?: string[];
    warnings?: string[];
    hints?: string[];
    manifest_adapter_id?: string | null;
    manifest_field_mapping?: Record<string, string>;
}

interface DatasetFitCardProps {
    contract: DatasetContract;
    projectId: number | string;
}

const TASK_LABELS: Record<string, string> = {
    causal_lm: 'SFT (causal language modeling)',
    classification: 'Classification',
    seq2seq: 'Seq-to-seq',
};

const SHAPE_LABELS: Record<string, { label: string; hint: string }> = {
    text: {
        label: 'Plain text',
        hint: 'rows have a `text` field with the full passage',
    },
    qa_pair: {
        label: 'Question + answer',
        hint: 'rows have `question` and `answer` fields',
    },
    chat_messages: {
        label: 'Chat transcripts',
        hint: 'rows have a `messages` array of {role, content}',
    },
    seq2seq_pair: {
        label: 'Source → target',
        hint: 'rows have `source_text` and `target_text` fields',
    },
    classification_label: {
        label: 'Text + label',
        hint: 'rows have `text` (or `question`) and `label` fields',
    },
};

function humanTask(task: string | undefined): string {
    if (!task) return 'training';
    return TASK_LABELS[task] || task;
}

function bestAlternativeTask(
    shapeCounts: Record<string, number>,
    requiredShapes: string[],
    sampledRows: number,
): { taskType: string; coverage: number; shape: string } | null {
    if (sampledRows <= 0) return null;
    const required = new Set(requiredShapes);
    const candidates: Array<{ taskType: string; shape: string }> = [
        { taskType: 'classification', shape: 'classification_label' },
        { taskType: 'seq2seq', shape: 'seq2seq_pair' },
        { taskType: 'causal_lm', shape: 'qa_pair' },
        { taskType: 'causal_lm', shape: 'text' },
    ];
    for (const candidate of candidates) {
        if (required.has(candidate.shape)) continue;
        const count = shapeCounts[candidate.shape] || 0;
        const coverage = count / sampledRows;
        if (coverage >= 0.5) {
            return { ...candidate, coverage };
        }
    }
    return null;
}

export default function DatasetFitCard({ contract, projectId }: DatasetFitCardProps) {
    const taskType = String(contract.task_type || 'causal_lm');
    const requiredShapes = Array.isArray(contract.required_shapes)
        ? contract.required_shapes
        : [];
    const shapeCounts = contract.shape_counts || {};
    const sampledRows = Math.max(0, Number(contract.sampled_rows) || 0);
    const compatibleRows = Math.max(0, Number(contract.compatible_rows) || 0);
    const coverage = Math.max(0, Math.min(1, Number(contract.coverage) || 0));
    const hasErrors = Array.isArray(contract.errors) && contract.errors.length > 0;
    const isReady = !hasErrors && coverage >= 0.9;

    const alt = bestAlternativeTask(shapeCounts, requiredShapes, sampledRows);

    if (sampledRows === 0 && !hasErrors) {
        // Contract didn't run (no prepared data yet) — bail out and let the
        // existing missing-train-file message handle it; nothing useful to
        // show here.
        return null;
    }

    return (
        <section
            className={`dataset-fit-card ${isReady ? 'is-ready' : 'is-blocked'}`}
            role="region"
            aria-labelledby="dataset-fit-card-title"
        >
            <div className="dataset-fit-card__head">
                {isReady ? (
                    <CheckCircle2 size={16} aria-hidden="true" />
                ) : (
                    <AlertTriangle size={16} aria-hidden="true" />
                )}
                <h4 id="dataset-fit-card-title">
                    {isReady
                        ? `Your dataset looks ready for ${humanTask(taskType)}.`
                        : `Why your dataset isn't ready for ${humanTask(taskType)}`}
                </h4>
            </div>

            {!isReady && (
                <p className="dataset-fit-card__lede">
                    You picked <strong>{humanTask(taskType)}</strong>, which needs rows shaped like
                    one of:
                </p>
            )}

            {!isReady && requiredShapes.length > 0 && (
                <ul className="dataset-fit-card__shape-list">
                    {requiredShapes.map((shape) => {
                        const meta = SHAPE_LABELS[shape] || { label: shape, hint: '' };
                        return (
                            <li key={shape}>
                                <strong>{meta.label}</strong>
                                {meta.hint && <span> — {meta.hint}</span>}
                            </li>
                        );
                    })}
                </ul>
            )}

            <div className="dataset-fit-card__stats">
                <div className="dataset-fit-card__stats-title">
                    What we saw in {sampledRows} sampled row{sampledRows === 1 ? '' : 's'}:
                </div>
                <ul className="dataset-fit-card__shape-counts">
                    {Object.entries(SHAPE_LABELS).map(([shape, meta]) => {
                        const count = Number(shapeCounts[shape] || 0);
                        const isRequired = requiredShapes.includes(shape);
                        const pct =
                            sampledRows > 0 ? Math.round((count / sampledRows) * 100) : 0;
                        return (
                            <li
                                key={shape}
                                className={`dataset-fit-card__shape-row ${
                                    isRequired ? 'is-required' : ''
                                }`}
                            >
                                <span className="dataset-fit-card__shape-name">{meta.label}</span>
                                <span className="dataset-fit-card__shape-bar">
                                    <span
                                        className="dataset-fit-card__shape-bar-fill"
                                        style={{ width: `${pct}%` }}
                                    />
                                </span>
                                <span className="dataset-fit-card__shape-count">
                                    {count} {isRequired && count > 0 ? '✓' : isRequired ? '✗' : ''}
                                </span>
                            </li>
                        );
                    })}
                </ul>
                <div className="dataset-fit-card__coverage">
                    Coverage: <strong>{Math.round(coverage * 100)}%</strong>
                    {!isReady && (
                        <span className="dim">
                            {' '}
                            — we need ≥ 90% rows in a compatible shape to start training.
                        </span>
                    )}
                    {compatibleRows > 0 && sampledRows > 0 && (
                        <span className="dim">
                            {' '}
                            ({compatibleRows} of {sampledRows} rows fit)
                        </span>
                    )}
                </div>
            </div>

            {!isReady && (
                <div className="dataset-fit-card__branches">
                    <div className="dataset-fit-card__branches-title">
                        Three ways to unblock yourself:
                    </div>
                    <div className="dataset-fit-card__branches-list">
                        <Link
                            className="dataset-fit-card__branch"
                            to={`/project/${projectId}/adapter-studio`}
                        >
                            <span>
                                <strong>Map your columns</strong>
                                <span className="dim">
                                    {' '}
                                    in Adapter Studio — point your existing fields at the ones the
                                    task needs.
                                </span>
                            </span>
                            <ArrowRight size={14} aria-hidden="true" />
                        </Link>
                        {alt && (
                            <Link
                                className="dataset-fit-card__branch"
                                to={`/project/${projectId}/training-config`}
                            >
                                <span>
                                    <strong>
                                        Switch task to {humanTask(alt.taskType)}
                                    </strong>
                                    <span className="dim">
                                        {' '}
                                        — your data already looks like{' '}
                                        {SHAPE_LABELS[alt.shape]?.label.toLowerCase() || alt.shape}{' '}
                                        ({Math.round(alt.coverage * 100)}% match).
                                    </span>
                                </span>
                                <ArrowRight size={14} aria-hidden="true" />
                            </Link>
                        )}
                        <Link className="dataset-fit-card__branch" to="/">
                            <span>
                                <strong>Start from a demo</strong>
                                <span className="dim">
                                    {' '}
                                    — pick a demo on the project list; it comes pre-shaped for
                                    its task.
                                </span>
                            </span>
                            <ArrowRight size={14} aria-hidden="true" />
                        </Link>
                    </div>
                </div>
            )}
        </section>
    );
}
