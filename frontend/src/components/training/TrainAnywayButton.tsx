/**
 * TrainAnywayButton — USER-SUCCESS Epic 1 supplement.
 *
 * Verdict-aware CTA that anchors the user to the existing "Create
 * Experiment" button in TrainingPanel. Label shifts with the
 * forecast verdict to make committing to a thin-data training run a
 * deliberate choice:
 *
 *   likely_pass  → "Train"          (neutral)
 *   borderline   → "Train (~XX%)"   (advisory)
 *   likely_fail  → "Train anyway"   (warning-tinted)
 *
 * Spec deviation: the roadmap said "replaces the existing Train
 * button on the config page." TrainingPanel's actual launch action
 * lives inside a 4,500-LOC component behind a multi-step Setup
 * Review flow; surgically replacing it risks far more than the
 * label-shift value justifies. Instead, this component renders an
 * anchor CTA inside the forecast panel that scrolls to / focuses
 * the existing button. The user-visible nudge — a verdict-tinted
 * label they have to actively click — is preserved.
 */

import type { ForecastVerdict } from '../../api/trainabilityForecast';
import './TrainAnywayButton.css';

interface Props {
    verdict: ForecastVerdict;
    confidencePct: number;
    /**
     * CSS selector for the actual training-submit button to scroll
     * + focus when clicked. Defaults to TrainingPanel's "Create
     * Experiment" button.
     */
    anchorSelector?: string;
}

const DEFAULT_ANCHOR = '.training-create-shell__actions button.btn-primary';

const LABEL_BY_VERDICT: Record<ForecastVerdict, (pct: number) => string> = {
    likely_pass: () => 'Train',
    borderline: (pct) => `Train (forecast ~${pct}%)`,
    likely_fail: () => 'Train anyway',
};

const TONE_BY_VERDICT: Record<ForecastVerdict, string> = {
    likely_pass: 'ok',
    borderline: 'warn',
    likely_fail: 'block',
};

const HINT_BY_VERDICT: Record<ForecastVerdict, string | null> = {
    likely_pass: null,
    borderline: 'Forecast is borderline — review signals above before committing.',
    likely_fail: 'Forecast suggests this run will fail. Address signals above or proceed deliberately.',
};

export default function TrainAnywayButton({
    verdict,
    confidencePct,
    anchorSelector = DEFAULT_ANCHOR,
}: Props) {
    const label = LABEL_BY_VERDICT[verdict](confidencePct);
    const tone = TONE_BY_VERDICT[verdict];
    const hint = HINT_BY_VERDICT[verdict];

    const handleClick = () => {
        if (typeof document === 'undefined') {
            return;
        }
        const target = document.querySelector<HTMLButtonElement>(anchorSelector);
        if (!target) {
            return;
        }
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Slight delay so the scroll completes before the focus
        // ring appears — otherwise the focus jumps the camera.
        window.setTimeout(() => target.focus({ preventScroll: true }), 300);
    };

    return (
        <div className={`train-anyway train-anyway--${tone}`}>
            <button
                type="button"
                className={`train-anyway__btn train-anyway__btn--${tone}`}
                onClick={handleClick}
                data-testid="train-anyway-button"
                data-verdict={verdict}
            >
                {label}
                <span aria-hidden="true" className="train-anyway__arrow">↓</span>
            </button>
            {hint && (
                <p className="train-anyway__hint" data-testid="train-anyway-hint">
                    {hint}
                </p>
            )}
        </div>
    );
}
