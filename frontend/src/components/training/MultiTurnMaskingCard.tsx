/**
 * MultiTurnMaskingCard — how a run's chat rows were masked for training.
 *
 * Rows with a system message, or more than one assistant turn, train with the
 * loss on assistant tokens only. This card shows the share of tokens that
 * carried loss, how many assistant turns that is, what truncation dropped and
 * how many rows fell back to masking only the last turn. Simulate runs
 * estimate the same block from character counts and carry `approximate`.
 *
 * Reads `config._runtime.multi_turn`, then `training_report.multi_turn`, then
 * `runtime_environment.multi_turn`; renders nothing when none is present.
 */

import type { Experiment } from './trainingPanelTypes';
import './MultiTurnMaskingCard.css';

export interface MultiTurnBlock {
    rows?: number;
    assistant_token_share?: number | null;
    assistant_turns?: number;
    truncated_rows?: number;
    dropped_turns?: number;
    dropped_rows?: number;
    mask_fallback_rows?: number;
    approximate?: boolean;
}

function asBlock(value: unknown): MultiTurnBlock | null {
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as MultiTurnBlock) : null;
}

/** Find the multi_turn block on an experiment payload (first candidate that has one). */
export function findMultiTurnBlock(experiment: Experiment | null | undefined): MultiTurnBlock | null {
    if (!experiment) return null;
    const rec = experiment as unknown as Record<string, unknown>;
    const runtime = experiment.config?.['_runtime'] as Record<string, unknown> | undefined;
    const report = rec['training_report'] as Record<string, unknown> | undefined;
    const env = rec['runtime_environment'] as Record<string, unknown> | undefined;
    return asBlock(runtime?.['multi_turn']) ?? asBlock(report?.['multi_turn']) ?? asBlock(env?.['multi_turn']);
}

function num(value: number | undefined): string {
    return typeof value === 'number' ? String(value) : '—';
}

export default function MultiTurnMaskingCard({ experiment }: { experiment: Experiment }) {
    const block = findMultiTurnBlock(experiment);
    if (!block) return null;
    const share = typeof block.assistant_token_share === 'number'
        ? `${(block.assistant_token_share * 100).toFixed(1)}%`
        : '—';
    return (
        <div className="card multi-turn-card" data-testid="multi-turn-card">
            <h4 className="multi-turn-card__title">
                Multi-turn masking{' '}
                <span className="u-hint">Run #{experiment.id}</span>
                {block.approximate && (
                    <span className="badge badge-warning" data-testid="multi-turn-approximate"
                        title="Estimated from character counts; a real run counts tokens.">
                        approximate
                    </span>
                )}
            </h4>
            <dl className="multi-turn-card__grid">
                <div><dt>Rows</dt><dd data-testid="multi-turn-rows">{num(block.rows)}</dd></div>
                <div><dt>Assistant token share</dt><dd data-testid="multi-turn-share">{share}</dd></div>
                <div><dt>Assistant turns</dt><dd data-testid="multi-turn-turns">{num(block.assistant_turns)}</dd></div>
                <div><dt>Truncated rows</dt><dd data-testid="multi-turn-truncated">{num(block.truncated_rows)}</dd></div>
                <div><dt>Dropped turns</dt><dd data-testid="multi-turn-dropped-turns">{num(block.dropped_turns)}</dd></div>
                <div><dt>Dropped rows</dt><dd data-testid="multi-turn-dropped-rows">{num(block.dropped_rows)}</dd></div>
                <div><dt>Mask fallbacks</dt><dd data-testid="multi-turn-fallbacks">{num(block.mask_fallback_rows)}</dd></div>
            </dl>
            <p className="u-hint u-mt0">
                Loss is computed on assistant tokens only; system, user and tool tokens are masked.
                A mask fallback trained on the last assistant turn only.
            </p>
        </div>
    );
}
