import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';
import TopBar from '../components/layout/TopBar';
import ProjectCard from '../components/dashboard/ProjectCard';
import './ProjectListPage.css';

export default function ProjectListPage() {
    const navigate = useNavigate();
    const { projects, totalProjects, isLoadingProjects, fetchProjects, createProject, deleteProject } = useProjectStore();

    const [showModal, setShowModal] = useState(false);
    const [newName, setNewName] = useState('');
    const [newDesc, setNewDesc] = useState('');
    const [newModel, setNewModel] = useState('');

    useEffect(() => {
        fetchProjects();
    }, [fetchProjects]);

    const handleCreate = async () => {
        if (!newName.trim()) return;
        const project = await createProject(newName.trim(), newDesc.trim(), newModel.trim());
        setShowModal(false);
        setNewName('');
        setNewDesc('');
        setNewModel('');
        navigate(`/project/${project.id}`);
    };

    const handleDelete = async (id: number) => {
        if (confirm('Delete this project? This cannot be undone.')) {
            await deleteProject(id);
        }
    };

    return (
        <div className="main-content" style={{ marginLeft: 0 }}>
            <TopBar
                title="SLM Platform"
                subtitle={`${totalProjects} project${totalProjects !== 1 ? 's' : ''}`}
                actions={
                    <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                        + New Project
                    </button>
                }
            />
            <div className="page-container" style={{ paddingTop: 'calc(var(--topbar-height) + var(--space-xl))' }}>
                {isLoadingProjects ? (
                    <div className="project-grid">
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="skeleton" style={{ height: 200, borderRadius: 16 }} />
                        ))}
                    </div>
                ) : projects.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">◈</div>
                        <div className="empty-state-title">No projects yet</div>
                        <div className="empty-state-text">
                            Create your first SLM project to start building, evaluating, and exporting domain-specific language models.
                        </div>
                        <button className="btn btn-primary" style={{ marginTop: 24 }} onClick={() => setShowModal(true)}>
                            + Create First Project
                        </button>
                    </div>
                ) : (
                    <div className="project-grid">
                        {projects.map((p) => (
                            <ProjectCard
                                key={p.id}
                                project={p}
                                onClick={(id) => navigate(`/project/${id}`)}
                                onDelete={handleDelete}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Create Project Modal */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2 className="modal-title">New SLM Project</h2>
                            <button className="btn btn-ghost" onClick={() => setShowModal(false)}>✕</button>
                        </div>
                        <div className="modal-body">
                            <div className="form-group">
                                <label className="form-label">Project Name *</label>
                                <input
                                    className="input"
                                    placeholder="e.g. Legal Document Copilot"
                                    value={newName}
                                    onChange={(e) => setNewName(e.target.value)}
                                    autoFocus
                                    onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Description</label>
                                <input
                                    className="input"
                                    placeholder="Brief description of the project goal"
                                    value={newDesc}
                                    onChange={(e) => setNewDesc(e.target.value)}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Base Model</label>
                                <input
                                    className="input"
                                    placeholder="e.g. microsoft/phi-2, meta-llama/Llama-3.2-1B"
                                    value={newModel}
                                    onChange={(e) => setNewModel(e.target.value)}
                                />
                                <div className="form-hint">HuggingFace model ID (1B–8B recommended)</div>
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                            <button className="btn btn-primary" onClick={handleCreate} disabled={!newName.trim()}>
                                Create Project
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
