import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';
import TopBar from '../components/layout/TopBar';
import ProjectCard from '../components/dashboard/ProjectCard';
import EmptyState from '../components/shared/EmptyState';
import Skeleton from '../components/shared/Skeleton';
import './ProjectListPage.css';

export default function ProjectListPage() {
    const navigate = useNavigate();
    const { projects, totalProjects, isLoadingProjects, fetchProjects, createProject, deleteProject } = useProjectStore();

    const [showModal, setShowModal] = useState(false);
    const [newName, setNewName] = useState('');
    const [newDesc, setNewDesc] = useState('');
    const [newModel, setNewModel] = useState('');

    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState<'all' | 'draft' | 'active' | 'completed'>('all');

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
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', alignItems: 'center', flexWrap: 'wrap' }}>
                    <input
                        className="input"
                        placeholder="🔍 Search projects..."
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        style={{ maxWidth: 300, background: 'rgba(255 255 255 / .04)' }}
                    />
                    <div style={{ display: 'flex', gap: '.5rem' }}>
                        {['all', 'draft', 'active', 'completed'].map(status => (
                            <button
                                key={status}
                                onClick={() => setStatusFilter(status as any)}
                                style={{
                                    padding: '.4rem 1rem',
                                    borderRadius: 999,
                                    background: statusFilter === status ? 'rgba(168, 85, 247, .2)' : 'rgba(255 255 255 / .05)',
                                    border: `1px solid ${statusFilter === status ? '#a855f7' : 'rgba(255 255 255 / .1)'}`,
                                    color: statusFilter === status ? '#fff' : 'rgba(255 255 255 / .6)',
                                    cursor: 'pointer',
                                    textTransform: 'capitalize',
                                    fontSize: '.85rem'
                                }}
                            >
                                {status}
                            </button>
                        ))}
                    </div>
                </div>

                {isLoadingProjects ? (
                    <div className="project-grid">
                        {[1, 2, 3].map((i) => (
                            <Skeleton key={i} height={200} borderRadius={16} />
                        ))}
                    </div>
                ) : projects.length === 0 ? (
                    <EmptyState
                        icon="◈"
                        title="No projects yet"
                        description="Create your first SLM project to start building, evaluating, and exporting domain-specific language models."
                        action={
                            <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                                + Create First Project
                            </button>
                        }
                    />
                ) : (
                    <div className="project-grid">
                        {projects
                            .filter(p => statusFilter === 'all' || p.status === statusFilter)
                            .filter(p => !searchQuery || p.name.toLowerCase().includes(searchQuery.toLowerCase()) || p.description?.toLowerCase().includes(searchQuery.toLowerCase()))
                            .map((p) => (
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
