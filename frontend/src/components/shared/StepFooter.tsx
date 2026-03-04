import './StepFooter.css';

interface StepFooterProps {
    currentStep: string;
    nextStep: string;
    nextStepIcon: string;
    isComplete: boolean;
    hint?: string;
    onNext: () => void | Promise<void>;
}

export default function StepFooter({ currentStep, nextStep, nextStepIcon, isComplete, hint, onNext }: StepFooterProps) {
    return (
        <div className={`step-footer ${isComplete ? 'step-footer--complete' : ''}`}>
            <div className="step-footer__status">
                {isComplete ? (
                    <span className="step-footer__check">✅ {currentStep} complete</span>
                ) : (
                    <span className="step-footer__hint">💡 {hint || `Complete ${currentStep} to continue`}</span>
                )}
            </div>
            <button
                className="btn btn-primary step-footer__next"
                onClick={onNext}
                disabled={!isComplete}
                title={!isComplete ? `Complete ${currentStep} first` : `Go to ${nextStep}`}
            >
                Continue to {nextStepIcon} {nextStep} →
            </button>
        </div>
    );
}
