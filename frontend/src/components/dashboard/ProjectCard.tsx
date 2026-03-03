import type { Project } from '../../types';
import './ProjectCard.css';

interface ProjectCardProps {
    project: Project;
    onClick: (id: number) => void;
    onDelete: (id: number) => void;
}

const STATUS_BADGES: Record<string, { label: string; className: string }> = {
    draft: { label: 'Draft', className: 'badge-info' },
    active: { label: 'Active', className: 'badge-success' },
    paused: { label: 'Paused', className: 'badge-warning' },
    completed: { label: 'Completed', className: 'badge-success' },
    failed: { label: 'Failed', className: 'badge-error' },
};

const STAGE_LABELS: Record<string, string> = {
    ingestion: 'Data Ingestion',
    cleaning: 'Data Cleaning',
    gold_set: 'Gold Dataset',
    synthetic: 'Synthetic Gen',
    dataset_prep: 'Dataset Prep',
    tokenization: 'Tokenization',
    training: 'Training',
    evaluation: 'Evaluation',
    compression: 'Compression',
    export: 'Export',
    completed: 'Completed',
};

export default function ProjectCard({ project, onClick, onDelete }: ProjectCardProps) {
    const statusBadge = STATUS_BADGES[project.status] || STATUS_BADGES.draft;

    return (
        <div className="project-card glass-card" onClick={() => onClick(project.id)}>
            <div className="project-card-header">
                <h3 className="project-card-name">{project.name}</h3>
                <span className={`badge ${statusBadge.className}`}>{statusBadge.label}</span>
            </div>
            {project.description && (
                <p className="project-card-desc">{project.description}</p>
            )}
            <div className="project-card-meta">
                <div className="meta-item">
                    <span className="meta-label">Stage</span>
                    <span className="meta-value badge badge-accent">
                        {STAGE_LABELS[project.pipeline_stage] || project.pipeline_stage}
                    </span>
                </div>
                {project.base_model_name && (
                    <div className="meta-item">
                        <span className="meta-label">Base Model</span>
                        <span className="meta-value">{project.base_model_name}</span>
                    </div>
                )}
            </div>
            <div className="project-card-footer">
                <span className="project-date">
                    {new Date(project.updated_at).toLocaleDateString()}
                </span>
                <button
                    className="btn btn-ghost btn-sm"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDelete(project.id);
                    }}
                >
                    🗑️
                </button>
            </div>
        </div>
    );
}
