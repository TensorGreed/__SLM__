import { useEffect, useMemo, useState } from 'react';
import { Navigate, useNavigate, useOutletContext, useParams } from 'react-router-dom';

import PipelineProgress from '../components/dashboard/PipelineProgress';
import IngestionPanel from '../components/data/IngestionPanel';
import CleaningPanel from '../components/data/CleaningPanel';
import GoldSetPanel from '../components/data/GoldSetPanel';
import SyntheticPanel from '../components/data/SyntheticPanel';
import DatasetPrepPanel from '../components/data/DatasetPrepPanel';
import TokenizationPanel from '../components/training/TokenizationPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import EvalPanel from '../components/evaluation/EvalPanel';
import CompressionPanel from '../components/compression/CompressionPanel';
import ExportPanel from '../components/export/ExportPanel';
import GettingStartedWizard from '../components/shared/GettingStartedWizard';
import api from '../api/client';
import { PIPELINE_TABS } from '../types';
import type { TabKey } from '../types';
import { useProjectStore } from '../stores/projectStore';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectPipelinePage.css';

const TAB_ORDER: TabKey[] = ['data', 'cleaning', 'goldset', 'synthetic', 'dataprep', 'tokenization', 'training', 'eval', 'compression', 'export'];

const TAB_PREREQ_INDEX: Record<TabKey, number> = {
    data: 0,
    cleaning: 1,
    goldset: 1,
    synthetic: 2,
    dataprep: 2,
    tokenization: 2,
    training: 3,
    eval: 4,
    compression: 4,
    export: 4,
};

const STAGE_ORDER = ['ingestion', 'cleaning', 'gold_set', 'synthetic', 'dataset_prep', 'data_adapter_preview', 'tokenization', 'training', 'evaluation', 'compression', 'export', 'completed'];
const TAB_TARGET_STAGE: Record<TabKey, string> = {
    data: 'ingestion',
    cleaning: 'cleaning',
    goldset: 'gold_set',
    synthetic: 'synthetic',
    dataprep: 'dataset_prep',
    tokenization: 'tokenization',
    training: 'training',
    eval: 'evaluation',
    compression: 'compression',
    export: 'export',
};

function getStageIndex(stage: string): number {
    const idx = STAGE_ORDER.indexOf(stage);
    return idx >= 0 ? idx : 0;
}

function isTabKey(value: string | undefined): value is TabKey {
    return !!value && TAB_ORDER.includes(value as TabKey);
}

export default function ProjectPipelinePage() {
    const { tabKey } = useParams<{ tabKey: string }>();
    const navigate = useNavigate();
    const { projectId, pipelineStatus, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const { activeTab, setActiveTab } = useProjectStore();

    const [wizardDismissed, setWizardDismissed] = useState(false);
    const showWizard = !wizardDismissed && pipelineStatus?.progress_percent === 0;

    const resolvedTab = useMemo<TabKey | null>(() => (isTabKey(tabKey) ? tabKey : null), [tabKey]);

    useEffect(() => {
        if (!resolvedTab) {
            return;
        }
        if (activeTab !== resolvedTab) {
            setActiveTab(resolvedTab);
        }
    }, [resolvedTab, activeTab, setActiveTab]);

    useEffect(() => {
        if (!resolvedTab) {
            navigate(`/project/${projectId}/pipeline/data`, { replace: true });
        }
    }, [resolvedTab, navigate, projectId]);

    const currentStageIndex = pipelineStatus ? getStageIndex(pipelineStatus.current_stage) : 0;
    const autoGate = pipelineStatus?.auto_gate ?? null;

    const isTabUnlocked = (key: TabKey): boolean => {
        const requiredIndex = TAB_PREREQ_INDEX[key];
        return currentStageIndex >= requiredIndex;
    };

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (
                e.target instanceof HTMLInputElement
                || e.target instanceof HTMLTextAreaElement
                || e.target instanceof HTMLSelectElement
            ) {
                return;
            }
            const current = resolvedTab || activeTab;
            if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') {
                return;
            }
            const currentIndex = TAB_ORDER.indexOf(current);
            let nextIndex = currentIndex;
            if (e.key === 'ArrowLeft' && currentIndex > 0) {
                nextIndex = currentIndex - 1;
            }
            if (e.key === 'ArrowRight' && currentIndex < TAB_ORDER.length - 1) {
                nextIndex = currentIndex + 1;
            }
            if (nextIndex === currentIndex) {
                return;
            }
            const nextTab = TAB_ORDER[nextIndex];
            if (!isTabUnlocked(nextTab)) {
                return;
            }
            setActiveTab(nextTab);
            navigate(`/project/${projectId}/pipeline/${nextTab}`);
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [resolvedTab, activeTab, currentStageIndex, navigate, projectId, setActiveTab]);

    if (!resolvedTab) {
        return <Navigate to={`/project/${projectId}/pipeline/data`} replace />;
    }

    const goToNextTab = async () => {
        const currentIndex = TAB_ORDER.indexOf(resolvedTab);
        if (currentIndex >= TAB_ORDER.length - 1) {
            return;
        }
        const nextTab = TAB_ORDER[currentIndex + 1];
        const targetStage = TAB_TARGET_STAGE[nextTab];

        let currentStage = pipelineStatus?.current_stage;
        let guard = 0;
        while (currentStage && currentStage !== targetStage && guard < 12) {
            try {
                const res = await api.post(`/projects/${projectId}/pipeline/advance`);
                currentStage = res.data.current_stage;
                guard += 1;
            } catch {
                break;
            }
        }

        await refreshPipelineStatus();
        setActiveTab(nextTab);
        navigate(`/project/${projectId}/pipeline/${nextTab}`);
    };

    const renderTabContent = () => {
        switch (resolvedTab) {
            case 'data': return <IngestionPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'cleaning': return <CleaningPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'goldset': return <GoldSetPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'synthetic': return <SyntheticPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'dataprep': return <DatasetPrepPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'tokenization': return <TokenizationPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'training':
                return (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
                        <div className="card pipeline-training-config-card">
                            <div>
                                <h3>Training Config moved to dedicated page</h3>
                                <p>
                                    Use the new Training Config menu item for model selection, hyperparameters,
                                    recipes, and preflight planning. This stage now focuses on runs and monitoring.
                                </p>
                            </div>
                            <button
                                className="btn btn-secondary"
                                onClick={() => navigate(`/project/${projectId}/training-config`)}
                            >
                                Open Training Config
                            </button>
                        </div>
                        <TrainingPanel
                            projectId={projectId}
                            onNextStep={goToNextTab}
                            title="Training Runs"
                            hideCreateControls
                        />
                    </div>
                );
            case 'eval': return <EvalPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'compression': return <CompressionPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'export': return <ExportPanel projectId={projectId} />;
            default: return null;
        }
    };

    return (
        <div className="workspace-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Pipeline Runs</h2>
                    <p className="workspace-page-subtitle">
                        Run data preparation, training, evaluation, and export stages in sequence.
                    </p>
                </div>
                <div className="workspace-page-header-actions">
                    <button
                        className="btn btn-ghost"
                        onClick={() => navigate(`/project/${projectId}/guide`)}
                    >
                        Open Guide
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={() => navigate(`/project/${projectId}/wizard`)}
                    >
                        Open Guided Setup
                    </button>
                </div>
            </section>
            {pipelineStatus && (
                <div className="card progress-card">
                    <PipelineProgress
                        stages={pipelineStatus.stages}
                        progressPercent={pipelineStatus.progress_percent}
                    />
                </div>
            )}

            {autoGate && (
                <div className="card pipeline-auto-gate-card">
                    <div className="pipeline-auto-gate-header">
                        <h3>Auto Gate</h3>
                        <span className={`badge ${autoGate.passed ? 'badge-success' : 'badge-error'}`}>
                            {autoGate.passed ? 'PASS' : 'FAIL'}
                        </span>
                    </div>
                    <div className="pipeline-auto-gate-meta">
                        <span>
                            Experiment: <strong>#{autoGate.experiment_id}</strong>
                        </span>
                        <span>
                            Pack: <strong>{autoGate.pack_id || 'auto'}</strong>
                        </span>
                        {autoGate.captured_at && (
                            <span>
                                Checked: <strong>{new Date(autoGate.captured_at).toLocaleString()}</strong>
                            </span>
                        )}
                    </div>
                    {autoGate.failed_gate_ids.length > 0 && (
                        <div className="pipeline-auto-gate-warn">
                            Failed required gates: {autoGate.failed_gate_ids.join(', ')}
                        </div>
                    )}
                    {autoGate.missing_required_metrics.length > 0 && (
                        <div className="pipeline-auto-gate-warn">
                            Missing required metrics: {autoGate.missing_required_metrics.join(', ')}
                        </div>
                    )}
                </div>
            )}

            {showWizard ? (
                <GettingStartedWizard
                    onStart={() => {
                        setWizardDismissed(true);
                        setActiveTab('data');
                        navigate(`/project/${projectId}/pipeline/data`);
                    }}
                />
            ) : (
                <div className="project-tabs-container">
                    <div className="tabs">
                        {PIPELINE_TABS.map((tab) => {
                            const unlocked = isTabUnlocked(tab.key);
                            const active = resolvedTab === tab.key;
                            return (
                                <button
                                    key={tab.key}
                                    className={`tab ${active ? 'active' : ''} ${!unlocked ? 'tab--locked' : ''}`}
                                    onClick={() => {
                                        if (!unlocked) {
                                            return;
                                        }
                                        setActiveTab(tab.key);
                                        navigate(`/project/${projectId}/pipeline/${tab.key}`);
                                    }}
                                    title={!unlocked ? 'Complete earlier steps first' : tab.label}
                                >
                                    <span>{unlocked ? tab.icon : '🔒'}</span> {tab.label}
                                </button>
                            );
                        })}
                    </div>
                    <div className="tab-content">{renderTabContent()}</div>
                </div>
            )}
        </div>
    );
}
