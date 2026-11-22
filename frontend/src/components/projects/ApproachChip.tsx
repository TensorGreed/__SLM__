/**
 * Decision-engine chip (Theme 7).
 *
 * Renders the recommended approach (prompt_only / rag_first / sft /
 * dpo / distillation) as a one-line callout under the brief textarea
 * in the create modal. Tone is green when the recommendation is
 * `sft` (the platform's default), warning-amber when the engine is
 * steering the user toward a different approach.
 *
 * UI simplification phase 7b (gap N6): the chip is advice only. Create sets
 * up a fine-tuning project either way, so it no longer offers an "or stay
 * with SFT anyway" click that changed nothing; a non-SFT tip says so in
 * words. One confidence number (the headline's, when it states one; else the
 * recommendation's, with the detected signals on hover). Lucide icons, no
 * emoji, no inline styles.
 */

import { Check, Lightbulb } from 'lucide-react';

import type { ApproachRecommendation } from '../../api/blueprintAnalyze';
import './ApproachChip.css';

interface ApproachChipProps {
    /** Recommendation payload from the backend. `null` while
     * the debounced analyze call hasn't returned yet. */
    recommendation: ApproachRecommendation | null;
    /** True while the debounced analyze request is in flight. */
    loading: boolean;
}

export default function ApproachChip({ recommendation, loading }: ApproachChipProps) {
    if (loading && !recommendation) {
        return (
            <div data-testid="approach-chip-loading" role="status" className="approach-chip approach-chip--loading">
                Analyzing your brief…
            </div>
        );
    }

    if (!recommendation) {
        return null;
    }

    const isSft = recommendation.approach === 'sft';
    const Icon = isSft ? Check : Lightbulb;
    // The headline may already state a confidence (the analysis's); then no second number.
    const headlineHasNumber = /\d+%/.test(recommendation.headline);
    const signals = recommendation.signals.join(', ') || 'no signals';

    return (
        <div
            data-testid="approach-chip"
            data-approach={recommendation.approach}
            role="status"
            className={`approach-chip ${isSft ? 'approach-chip--sft' : 'approach-chip--tip'}`}
        >
            <div className="approach-chip__head">
                <Icon size={14} aria-hidden="true" className="approach-chip__icon" />
                <strong className="approach-chip__headline" data-testid="approach-chip-headline">
                    {recommendation.headline}
                </strong>
                <span title={signals} aria-label={`What signals were detected: ${signals}`} className="approach-chip__signals" data-testid="approach-chip-signals">
                    {headlineHasNumber ? 'why?' : `${Math.round(recommendation.confidence * 100)}% sure · why?`}
                </span>
            </div>
            <div className="approach-chip__rationale" data-testid="approach-chip-rationale">
                {recommendation.rationale}
            </div>
            {!isSft && (
                <div className="approach-chip__note" data-testid="approach-chip-advice-only">
                    Optional advice: Create still sets up a fine-tuning project. You can add retrieval later.
                </div>
            )}
        </div>
    );
}
