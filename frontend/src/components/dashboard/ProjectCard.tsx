import { useState, useEffect } from 'react';
import type { Project, ProjectStats } from '../../types';
import api from '../../api/client';
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

const STAGE_ORDER = ['ingestion', 'cleaning', 'gold_set', 'synthetic', 'dataset_prep', 'tokenization', 'training', 'evaluation', 'compression', 'export', 'completed'];

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
    const stageIndex = STAGE_ORDER.indexOf(project.pipeline_stage);
    const progressPercent = Math.max(5, Math.min(100, Math.round(((stageIndex >= 0 ? stageIndex : 0) / (STAGE_ORDER.length - 1)) * 100)));

    const [stats, setStats] = useState<ProjectStats | null>(null);

    useEffect(() => {
        let isMounted = true;
        api.get(`/projects/${project.id}/stats`).then(res => {
            if (isMounted) setStats(res.data);
        }).catch(() => { });
        return () => { isMounted = false; };
    }, [project.id]);

    return (
        <div className={`project-card glass-card status-${project.status}`} onClick={() => onClick(project.id)}>
            <div className="project-card-header">
                <h3 className="project-card-name">{project.name}</h3>
                <span className={`badge ${statusBadge.className}`}>{statusBadge.label}</span>
            </div>

            {project.description && (
                <p className="project-card-desc" title={project.description}>
                    {project.description}
                </p>
            )}

            <div className="project-card-progress" title={`Pipeline Progress: ${progressPercent}%`}>
                <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
                </div>
                <div className="progress-text">
                    <span className="stage-name">{STAGE_LABELS[project.pipeline_stage] || project.pipeline_stage}</span>
                    <span className="stage-percent">{progressPercent}%</span>
                </div>
            </div>

            <div className="project-card-meta">
                <div className="meta-item">
                    <span className="meta-label">Documents</span>
                    <span className="meta-value">{stats ? stats.total_documents.toLocaleString() : '-'}</span>
                </div>
                {project.base_model_name && (
                    <div className="meta-item" style={{ flex: 1, minWidth: 0 }}>
                        <span className="meta-label">Base Model</span>
                        <span className="meta-value truncate" title={project.base_model_name}>
                            {project.base_model_name.length > 20
                                ? project.base_model_name.substring(0, 18) + '...'
                                : project.base_model_name}
                        </span>
                    </div>
                )}
                <div className="meta-item">
                    <span className="meta-label">Experiments</span>
                    <span className="meta-value">{stats ? stats.experiment_count : '-'}</span>
                </div>
            </div>

            <div className="project-card-footer">
                <span className="project-date">
                    Updated {new Date(project.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
                <button
                    className="btn btn-ghost btn-sm btn-delete"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDelete(project.id);
                    }}
                    title="Delete Project"
                >
                    🗑️
                </button>
            </div>
        </div>
    );
}
