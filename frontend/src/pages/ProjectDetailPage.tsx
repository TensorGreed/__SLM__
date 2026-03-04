import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';
import Sidebar from '../components/layout/Sidebar';
import TopBar from '../components/layout/TopBar';
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
import { PIPELINE_TABS } from '../types';
import type { TabKey } from '../types';
import api from '../api/client';
import './ProjectDetailPage.css';

// Define tab order for next-step navigation
const TAB_ORDER: TabKey[] = ['data', 'cleaning', 'goldset', 'synthetic', 'dataprep', 'tokenization', 'training', 'eval', 'compression', 'export'];

// Define which pipeline stage index each tab requires to be unlocked
// 0 = always available, 1 = needs ingestion done, etc.
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

// Pipeline stage order for comparison
const STAGE_ORDER = ['ingestion', 'cleaning', 'gold_set', 'synthetic', 'dataset_prep', 'tokenization', 'training', 'evaluation', 'compression', 'export', 'completed'];
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

export default function ProjectDetailPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const {
        activeProject,
        pipelineStatus,
        activeTab,
        setActiveTab,
        fetchProject,
        fetchPipelineStatus,
    } = useProjectStore();

    const [showWizard, setShowWizard] = useState(false);

    useEffect(() => {
        if (id) {
            const projectId = parseInt(id, 10);
            fetchProject(projectId);
            fetchPipelineStatus(projectId);
        }
    }, [id, fetchProject, fetchPipelineStatus]);

    // Show wizard if project is at ingestion stage and has no data
    useEffect(() => {
        if (activeProject && pipelineStatus) {
            const isNew = pipelineStatus.progress_percent === 0;
            setShowWizard(isNew);
        }
    }, [activeProject, pipelineStatus]);

    const currentStageIndex = pipelineStatus ? getStageIndex(pipelineStatus.current_stage) : 0;

    const isTabUnlocked = (tabKey: TabKey): boolean => {
        const requiredIndex = TAB_PREREQ_INDEX[tabKey];
        return currentStageIndex >= requiredIndex;
    };

    // Keyboard shortcuts for tab navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;

            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                const currentIndex = TAB_ORDER.indexOf(activeTab);
                let newIndex = currentIndex;

                if (e.key === 'ArrowLeft' && currentIndex > 0) newIndex = currentIndex - 1;
                if (e.key === 'ArrowRight' && currentIndex < TAB_ORDER.length - 1) newIndex = currentIndex + 1;

                if (newIndex !== currentIndex) {
                    const nextTab = TAB_ORDER[newIndex];
                    if (isTabUnlocked(nextTab)) {
                        setActiveTab(nextTab);
                    }
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [activeTab, currentStageIndex]);

    if (!activeProject) {
        return (
            <div className="app-layout">
                <div className="main-content">
                    <div className="page-container">
                        <div className="skeleton" style={{ height: 48, width: 300, marginBottom: 24 }} />
                        <div className="skeleton" style={{ height: 200 }} />
                    </div>
                </div>
            </div>
        );
    }

    const projectId = activeProject.id;

    const handleTabClick = (tabKey: TabKey) => {
        if (isTabUnlocked(tabKey)) {
            setActiveTab(tabKey);
        }
    };

    const goToNextTab = async () => {
        const currentIndex = TAB_ORDER.indexOf(activeTab);
        if (currentIndex < TAB_ORDER.length - 1) {
            const nextTab = TAB_ORDER[currentIndex + 1];
            const targetStage = TAB_TARGET_STAGE[nextTab];

            // Keep backend pipeline stage aligned with the visible tab flow.
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

            await fetchPipelineStatus(projectId);
            setActiveTab(nextTab);
        }
    };

    const handleWizardStart = () => {
        setShowWizard(false);
        setActiveTab('data');
    };

    const renderTabContent = () => {
        switch (activeTab) {
            case 'data': return <IngestionPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'cleaning': return <CleaningPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'goldset': return <GoldSetPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'synthetic': return <SyntheticPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'dataprep': return <DatasetPrepPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'tokenization': return <TokenizationPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'training': return <TrainingPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'eval': return <EvalPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'compression': return <CompressionPanel projectId={projectId} onNextStep={goToNextTab} />;
            case 'export': return <ExportPanel projectId={projectId} />;
            default: return null;
        }
    };

    return (
        <div className="app-layout">
            <Sidebar projectName={activeProject.name} onNavigateHome={() => navigate('/')} />
            <div className="main-content">
                <TopBar
                    title={activeProject.name}
                    subtitle={activeProject.description || undefined}
                    actions={
                        <span className={`badge ${activeProject.status === 'active' ? 'badge-success' : 'badge-info'}`}>
                            {activeProject.status}
                        </span>
                    }
                />
                <div className="page-container">
                    {pipelineStatus && (
                        <div className="card progress-card">
                            <PipelineProgress stages={pipelineStatus.stages} progressPercent={pipelineStatus.progress_percent} />
                        </div>
                    )}

                    {showWizard ? (
                        <GettingStartedWizard onStart={handleWizardStart} />
                    ) : (
                        <div className="project-tabs-container">
                            <div className="tabs">
                                {PIPELINE_TABS.map((tab) => {
                                    const unlocked = isTabUnlocked(tab.key);
                                    return (
                                        <button
                                            key={tab.key}
                                            className={`tab ${activeTab === tab.key ? 'active' : ''} ${!unlocked ? 'tab--locked' : ''}`}
                                            onClick={() => handleTabClick(tab.key)}
                                            title={!unlocked ? `Complete earlier steps first` : tab.label}
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
            </div>
        </div>
    );
}
