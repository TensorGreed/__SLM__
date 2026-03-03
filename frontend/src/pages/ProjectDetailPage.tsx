import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';
import Sidebar from '../components/layout/Sidebar';
import TopBar from '../components/layout/TopBar';
import PipelineProgress from '../components/dashboard/PipelineProgress';
import IngestionPanel from '../components/data/IngestionPanel';
import CleaningPanel from '../components/data/CleaningPanel';
import GoldSetPanel from '../components/data/GoldSetPanel';
import SyntheticPanel from '../components/data/SyntheticPanel';
import TrainingPanel from '../components/training/TrainingPanel';
import EvalPanel from '../components/evaluation/EvalPanel';
import CompressionPanel from '../components/compression/CompressionPanel';
import ExportPanel from '../components/export/ExportPanel';
import { PIPELINE_TABS } from '../types';
import './ProjectDetailPage.css';

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

    useEffect(() => {
        if (id) {
            const projectId = parseInt(id, 10);
            fetchProject(projectId);
            fetchPipelineStatus(projectId);
        }
    }, [id, fetchProject, fetchPipelineStatus]);

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

    const renderTabContent = () => {
        switch (activeTab) {
            case 'data': return <IngestionPanel projectId={projectId} />;
            case 'cleaning': return <CleaningPanel projectId={projectId} />;
            case 'goldset': return <GoldSetPanel projectId={projectId} />;
            case 'synthetic': return <SyntheticPanel projectId={projectId} />;
            case 'training': return <TrainingPanel projectId={projectId} />;
            case 'eval': return <EvalPanel projectId={projectId} />;
            case 'compression': return <CompressionPanel projectId={projectId} />;
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
                    <div className="project-tabs-container">
                        <div className="tabs">
                            {PIPELINE_TABS.map((tab) => (
                                <button key={tab.key} className={`tab ${activeTab === tab.key ? 'active' : ''}`} onClick={() => setActiveTab(tab.key)}>
                                    <span>{tab.icon}</span> {tab.label}
                                </button>
                            ))}
                        </div>
                        <div className="tab-content">{renderTabContent()}</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
