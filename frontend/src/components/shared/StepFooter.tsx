/**
 * Footer bar with completion badge and conditional next-step navigation button.
 */

import { Check, Lightbulb } from 'lucide-react';
import './StepFooter.css';

interface StepFooterProps {
    currentStep: string;
    nextStep: string;
    /** Unused since U7 (icons are gone from the footer); kept optional for older callers. */
    nextStepIcon?: string;
    isComplete: boolean;
    hint?: string;
    onNext: () => void | Promise<void>;
}

export default function StepFooter({ currentStep, nextStep, isComplete, hint, onNext }: StepFooterProps) {
    return (
        <div className={`step-footer ${isComplete ? 'step-footer--complete' : ''}`}>
            <div className="step-footer__status">
                {isComplete ? (
                    <span className="step-footer__check"><Check className="u-icon" aria-hidden="true" /> {currentStep} complete</span>
                ) : (
                    <span className="step-footer__hint"><Lightbulb className="u-icon" aria-hidden="true" /> {hint || `Complete ${currentStep} to continue`}</span>
                )}
            </div>
            <button
                className="btn btn-primary step-footer__next"
                onClick={onNext}
                disabled={!isComplete}
                title={!isComplete ? `Complete ${currentStep} first` : `Go to ${nextStep}`}
            >
                Continue to {nextStep} →
            </button>
        </div>
    );
}
