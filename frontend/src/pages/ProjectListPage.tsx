import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProjectStore } from '../stores/projectStore';
import TopBar from '../components/layout/TopBar';
import ProjectCard from '../components/dashboard/ProjectCard';
import EmptyState from '../components/shared/EmptyState';
import Skeleton from '../components/shared/Skeleton';
import api from '../api/client';
import type { DomainPackSummary, DomainProfileSummary } from '../types';
import './ProjectListPage.css';

const STATUS_FILTERS = ['all', 'draft', 'active', 'completed'] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number];

export default function ProjectListPage() {
    const navigate = useNavigate();
    const { projects, totalProjects, isLoadingProjects, fetchProjects, createProject, deleteProject } = useProjectStore();

    const [showModal, setShowModal] = useState(false);
    const [showMagicModal, setShowMagicModal] = useState(false);
    const [magicPrompt, setMagicPrompt] = useState('');
    const [isMagicCreating, setIsMagicCreating] = useState(false);

    const [newName, setNewName] = useState('');
    const [newDesc, setNewDesc] = useState('');
    const [newModel, setNewModel] = useState('');
    const [newDomainPackId, setNewDomainPackId] = useState('');
    const [newDomainProfileId, setNewDomainProfileId] = useState('');
    const [domainPacks, setDomainPacks] = useState<DomainPackSummary[]>([]);
    const [domainProfiles, setDomainProfiles] = useState<DomainProfileSummary[]>([]);

    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

    useEffect(() => {
        fetchProjects();
    }, [fetchProjects]);

    useEffect(() => {
        api.get<{ packs: DomainPackSummary[] }>('/domain-packs')
            .then((res) => setDomainPacks(res.data.packs || []))
            .catch(() => setDomainPacks([]));
    }, []);

    useEffect(() => {
        api.get<{ profiles: DomainProfileSummary[] }>('/domain-profiles')
            .then((res) => setDomainProfiles(res.data.profiles || []))
            .catch(() => setDomainProfiles([]));
    }, []);

    const handleCreate = async () => {
        if (!newName.trim()) return;
        const domainPackId = newDomainPackId ? Number(newDomainPackId) : null;
        const domainProfileId = newDomainProfileId ? Number(newDomainProfileId) : null;
        const project = await createProject(
            newName.trim(),
            newDesc.trim(),
            newModel.trim(),
            domainPackId,
            domainProfileId,
        );
        setShowModal(false);
        setNewName('');
        setNewDesc('');
        setNewModel('');
        setNewDomainPackId('');
        setNewDomainProfileId('');
        navigate(`/project/${project.id}`);
    };

    const handleMagicCreate = async () => {
        if (!magicPrompt.trim()) return;
        setIsMagicCreating(true);
        try {
            const res = await api.post('/projects/magic-create', { prompt: magicPrompt.trim() });
            setShowMagicModal(false);
            setMagicPrompt('');
            navigate(`/project/${res.data.id}`);
        } catch (error: any) {
            alert(error.response?.data?.detail || 'Magic create failed');
        } finally {
            setIsMagicCreating(false);
        }
    };

    const handleDelete = async (id: number) => {
        if (confirm('Delete this project? This cannot be undone.')) {
            await deleteProject(id);
        }
    };

    const filteredProjects = projects
        .filter((project) => statusFilter === 'all' || project.status === statusFilter)
        .filter((project) => {
            if (!searchQuery) return true;
            const query = searchQuery.toLowerCase();
            return project.name.toLowerCase().includes(query) || project.description?.toLowerCase().includes(query);
        });

    return (
        <div className="main-content project-list-main">
            <TopBar
                title="BrewSLM"
                subtitle={`${totalProjects} project${totalProjects !== 1 ? 's' : ''}`}
                actions={
                    <div className="project-list-top-actions">
                        <button className="btn btn-secondary" onClick={() => setShowMagicModal(true)}>
                            ✨ Magic Create
                        </button>
                        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                            + New Project
                        </button>
                    </div>
                }
            />

            <div className="page-container project-list-page">
                <section className="card project-list-toolbar">
                    <input
                        className="input project-list-search"
                        placeholder="Search projects..."
                        value={searchQuery}
                        onChange={(event) => setSearchQuery(event.target.value)}
                    />
                    <div className="project-list-filters">
                        {STATUS_FILTERS.map((status) => (
                            <button
                                key={status}
                                className={`project-list-filter ${statusFilter === status ? 'active' : ''}`}
                                onClick={() => setStatusFilter(status)}
                            >
                                {status}
                            </button>
                        ))}
                    </div>
                </section>

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
                        description="Create your first BrewSLM project to start building, evaluating, and exporting domain-specific small language models."
                        action={
                            <div className="project-list-empty-actions">
                                <button className="btn btn-secondary" onClick={() => setShowMagicModal(true)}>
                                    ✨ Magic Create
                                </button>
                                <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                                    + Create First Project
                                </button>
                            </div>
                        }
                    />
                ) : (
                    <div className="project-grid">
                        {filteredProjects.map((project) => (
                                <ProjectCard
                                    key={project.id}
                                    project={project}
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
                            <h2 className="modal-title">New BrewSLM Project</h2>
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
                            <div className="form-group">
                                <label className="form-label">Domain Pack</label>
                                <select
                                    className="input"
                                    value={newDomainPackId}
                                    onChange={(e) => setNewDomainPackId(e.target.value)}
                                >
                                    <option value="">Auto-assign default</option>
                                    {domainPacks.map((pack) => (
                                        <option key={pack.id} value={pack.id}>
                                            {pack.display_name} ({pack.pack_id})
                                        </option>
                                    ))}
                                </select>
                                <div className="form-hint">Optional pack-level defaults and overlays.</div>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Domain Profile</label>
                                <select
                                    className="input"
                                    value={newDomainProfileId}
                                    onChange={(e) => setNewDomainProfileId(e.target.value)}
                                >
                                    <option value="">Auto-assign default</option>
                                    {domainProfiles.map((profile) => (
                                        <option key={profile.id} value={profile.id}>
                                            {profile.display_name} ({profile.profile_id})
                                        </option>
                                    ))}
                                </select>
                                <div className="form-hint">Can be reassigned later in project settings.</div>
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

            {/* Magic Create Modal */}
            {showMagicModal && (
                <div className="modal-overlay" onClick={() => !isMagicCreating && setShowMagicModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2 className="modal-title">Magic Create</h2>
                            <button className="btn btn-ghost" onClick={() => !isMagicCreating && setShowMagicModal(false)}>✕</button>
                        </div>
                        <div className="modal-body">
                            <p className="project-list-magic-copy">
                                Describe the dataset or model you want to build. BrewSLM AI Architect will configure the pipeline, pick a base model, and assign the right domain packs for you.
                            </p>
                            <div className="form-group">
                                <label className="form-label">What do you want to build?</label>
                                <textarea
                                    className="input"
                                    placeholder="e.g. I have 500 PDFs of legal contracts and I want a model that extracts the liabilities."
                                    value={magicPrompt}
                                    onChange={(e) => setMagicPrompt(e.target.value)}
                                    rows={4}
                                    disabled={isMagicCreating}
                                    autoFocus
                                />
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={() => setShowMagicModal(false)} disabled={isMagicCreating}>Cancel</button>
                            <button className="btn btn-primary" onClick={handleMagicCreate} disabled={!magicPrompt.trim() || isMagicCreating}>
                                {isMagicCreating ? '✨ Architecting...' : 'Magic Create'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
