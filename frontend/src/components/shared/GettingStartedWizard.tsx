import './GettingStartedWizard.css';

interface GettingStartedWizardProps {
    onStart: () => void;
}

const STEPS = [
    { num: 1, icon: '📂', title: 'Ingest Data', desc: 'Upload files or import from HuggingFace / Kaggle' },
    { num: 2, icon: '🧹', title: 'Clean & Chunk', desc: 'Deduplicate, redact PII, score quality, chunk text' },
    { num: 3, icon: '🏆', title: 'Gold Dataset', desc: 'Write expert Q&A pairs for evaluation (test set)' },
    { num: 4, icon: '🧪', title: 'Synthetic Gen', desc: 'Teacher model converts your text into training pairs' },
    { num: 5, icon: '🔬', title: 'Train Model', desc: 'Fine-tune with LoRA / SFT on your GPU' },
    { num: 6, icon: '📊', title: 'Evaluate', desc: 'Grade against Gold Set + run LLM-as-a-Judge' },
    { num: 7, icon: '📦', title: 'Compress', desc: 'Quantize to 4-bit / 8-bit for deployment' },
    { num: 8, icon: '🚀', title: 'Export', desc: 'Package as GGUF, ONNX, or HuggingFace format' },
];

export default function GettingStartedWizard({ onStart }: GettingStartedWizardProps) {
    return (
        <div className="wizard-overlay animate-fade-in">
            <div className="wizard-card">
                <div className="wizard-header">
                    <h2 className="wizard-title">🚀 Welcome to your BrewSLM pipeline</h2>
                    <p className="wizard-subtitle">Follow these 8 steps to go from raw data to a deployed small language model.</p>
                </div>
                <div className="wizard-steps">
                    {STEPS.map((step, idx) => (
                        <div key={step.num} className={`wizard-step ${idx === 0 ? 'wizard-step--active' : ''}`}>
                            <div className="wizard-step__num">{step.num}</div>
                            <div className="wizard-step__icon">{step.icon}</div>
                            <div className="wizard-step__info">
                                <div className="wizard-step__title">{step.title}</div>
                                <div className="wizard-step__desc">{step.desc}</div>
                            </div>
                            {idx < STEPS.length - 1 && <div className="wizard-step__arrow">→</div>}
                        </div>
                    ))}
                </div>
                <div className="wizard-action">
                    <button className="btn btn-primary btn-lg" onClick={onStart}>
                        📂 Start: Upload Your Data
                    </button>
                    <p className="wizard-action__hint">You can always return to any step later from the sidebar.</p>
                </div>
            </div>
        </div>
    );
}
