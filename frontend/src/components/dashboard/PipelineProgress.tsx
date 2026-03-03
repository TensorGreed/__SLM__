import type { PipelineStageInfo } from '../../types';
import './PipelineProgress.css';

interface PipelineProgressProps {
    stages: PipelineStageInfo[];
    progressPercent: number;
}

export default function PipelineProgress({ stages, progressPercent }: PipelineProgressProps) {
    return (
        <div className="pipeline-progress">
            <div className="progress-header">
                <span className="progress-label">Pipeline Progress</span>
                <span className="progress-percent">{progressPercent}%</span>
            </div>
            <div className="progress-bar-track">
                <div
                    className="progress-bar-fill"
                    style={{ width: `${progressPercent}%` }}
                />
            </div>
            <div className="progress-stages">
                {stages.map((stage) => (
                    <div key={stage.stage} className={`progress-stage ${stage.status}`}>
                        <div className="stage-dot-container">
                            <div className="stage-dot">
                                {stage.status === 'completed' ? '✓' : stage.index + 1}
                            </div>
                        </div>
                        <span className="stage-label">{stage.display_name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
