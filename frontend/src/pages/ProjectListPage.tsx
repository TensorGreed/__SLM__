import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import api from '../api/client';
import DemoProjectTiles from '../components/dashboard/DemoProjectTiles';
import FirstRunCheatSheet from '../components/dashboard/FirstRunCheatSheet';
import ProjectCard from '../components/dashboard/ProjectCard';
import TopBar from '../components/layout/TopBar';
import EmptyState from '../components/shared/EmptyState';
import Skeleton from '../components/shared/Skeleton';
import Term from '../components/shared/Term';
import { useProjectStore } from '../stores/projectStore';
import type {
    DomainPackSummary,
    DomainProfileSummary,
    StarterPackCatalogResponse,
    StarterPackSummary,
} from '../types';
import './ProjectListPage.css';

const STATUS_FILTERS = ['all', 'draft', 'active', 'completed'] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number];

function parseMultiline(value: string): string[] {
    return value
        .split('\n')
        .map((item) => item.trim())
        .filter(Boolean);
}

export default function ProjectListPage() {
    const navigate = useNavigate();
    const { projects, totalProjects, isLoadingProjects, fetchProjects, createProject, deleteProject } = useProjectStore();

    const [showModal, setShowModal] = useState(false);
    const [showMagicModal, setShowMagicModal] = useState(false);
    const [magicPrompt, setMagicPrompt] = useState('');
    const [isMagicCreating, setIsMagicCreating] = useState(false);
    const [isCreating, setIsCreating] = useState(false);

    const [newName, setNewName] = useState('');
    const [newDesc, setNewDesc] = useState('');
    const [newModel, setNewModel] = useState('');
    const [newStarterPackId, setNewStarterPackId] = useState('');
    const [newDomainPackId, setNewDomainPackId] = useState('');
    const [newDomainProfileId, setNewDomainProfileId] = useState('');

    // Brief-driven create (Theme 1 Epic 1). The default modal is a single
    // "What problem do you want this model to solve?" textarea — the
    // backend POST /projects handler runs analyze_domain_brief inline
    // when brief_text is supplied. Power users open the Advanced
    // disclosure for the dense config surface.
    const [briefText, setBriefText] = useState('');
    const [sampleInputsText, setSampleInputsText] = useState('');
    const [sampleOutputsText, setSampleOutputsText] = useState('');
    const [riskNotesText, setRiskNotesText] = useState('');
    const [deploymentTarget, setDeploymentTarget] = useState('vllm_server');
    const [analyzeError, setAnalyzeError] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);

    const [starterPacks, setStarterPacks] = useState<StarterPackSummary[]>([]);
    const [domainPacks, setDomainPacks] = useState<DomainPackSummary[]>([]);
    const [domainProfiles, setDomainProfiles] = useState<DomainProfileSummary[]>([]);

    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

    useEffect(() => {
        fetchProjects();
    }, [fetchProjects]);

    useEffect(() => {
        api.get<StarterPackCatalogResponse>('/starter-packs/catalog')
            .then((res) => setStarterPacks(res.data.starter_packs || []))
            .catch(() => setStarterPacks([]));
    }, []);

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

    const selectedStarterPack = starterPacks.find((pack) => pack.id === newStarterPackId) || null;

    const filteredProjects = projects
        .filter((project) => statusFilter === 'all' || project.status === statusFilter)
        .filter((project) => {
            if (!searchQuery) return true;
            const query = searchQuery.toLowerCase();
            return project.name.toLowerCase().includes(query) || project.description?.toLowerCase().includes(query);
        });

    const parsedSampleInputs = useMemo(() => parseMultiline(sampleInputsText), [sampleInputsText]);
    const parsedSampleOutputs = useMemo(() => parseMultiline(sampleOutputsText), [sampleOutputsText]);
    const parsedRiskNotes = useMemo(() => parseMultiline(riskNotesText), [riskNotesText]);

    const resetCreateModal = () => {
        setNewName('');
        setNewDesc('');
        setNewModel('');
        setNewStarterPackId('');
        setNewDomainPackId('');
        setNewDomainProfileId('');

        setBriefText('');
        setSampleInputsText('');
        setSampleOutputsText('');
        setRiskNotesText('');
        setDeploymentTarget('vllm_server');
        setAnalyzeError('');
        setShowAdvanced(false);
    };

    const openCreateModal = () => {
        resetCreateModal();
        setShowModal(true);
    };

    const closeCreateModal = () => {
        setShowModal(false);
        resetCreateModal();
    };

    /**
     * Single create handler — figures out brief-driven vs. plain
     * create based on whether the user filled in the brief textarea.
     * The backend POST /projects handler runs analyze_domain_brief
     * inline when `brief_text` is supplied, so we don't need to
     * pre-analyze on the client.
     */
    const handleCreate = async () => {
        if (!newName.trim()) return;
        const trimmedBrief = briefText.trim();
        setIsCreating(true);
        setAnalyzeError('');
        try {
            const starterPackId = newStarterPackId.trim() ? newStarterPackId.trim() : null;
            const domainPackId = newDomainPackId ? Number(newDomainPackId) : null;
            const domainProfileId = newDomainProfileId ? Number(newDomainProfileId) : null;

            const project = await createProject(
                newName.trim(),
                newDesc.trim(),
                newModel.trim(),
                starterPackId,
                domainPackId,
                domainProfileId,
                trimmedBrief
                    ? {
                        beginnerMode: true,
                        briefText: trimmedBrief,
                        sampleInputs: parsedSampleInputs,
                        sampleOutputs: parsedSampleOutputs,
                        targetProfileId: deploymentTarget.trim() || null,
                    }
                    : undefined,
            );
            closeCreateModal();
            // Land on the project guide page so the Quickstart card is
            // the first surface — frictionless first-success path.
            navigate(`/project/${project.id}/guide`);
        } catch (error) {
            const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
            if (typeof detail === 'string') {
                setAnalyzeError(detail);
            } else if (
                detail
                && typeof detail === 'object'
                && 'message' in detail
                && typeof (detail as { message?: unknown }).message === 'string'
            ) {
                setAnalyzeError((detail as { message: string }).message);
            } else {
                setAnalyzeError('Project creation failed. Please review your inputs and try again.');
            }
        } finally {
            setIsCreating(false);
        }
    };

    const handleMagicCreate = async () => {
        if (!magicPrompt.trim()) return;
        setIsMagicCreating(true);
        try {
            const res = await api.post('/projects/magic-create', { prompt: magicPrompt.trim() });
            setShowMagicModal(false);
            setMagicPrompt('');
            navigate(`/project/${res.data.id}`);
        } catch (error) {
            const detail = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            alert(typeof detail === 'string' ? detail : 'Magic create failed');
        } finally {
            setIsMagicCreating(false);
        }
    };

    const handleDelete = async (id: number) => {
        if (confirm('Delete this project? This cannot be undone.')) {
            await deleteProject(id);
        }
    };

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
                        <button className="btn btn-primary" onClick={openCreateModal}>
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

                <FirstRunCheatSheet />

                <DemoProjectTiles />

                {isLoadingProjects ? (
                    <div className="project-grid">
                        {[1, 2, 3].map((i) => (
                            <Skeleton key={i} height={200} borderRadius={16} />
                        ))}
                    </div>
                ) : projects.length === 0 ? (
                    <EmptyState
                        title="No projects yet"
                        description="Create your first BrewSLM project to start building, evaluating, and exporting domain-specific Small Language Models. Try Magic Create to describe what you want in plain English, or pick a starter template."
                        primary={{ label: '+ Create First Project', onClick: openCreateModal }}
                        secondary={{ label: '✨ Magic Create', onClick: () => setShowMagicModal(true) }}
                        docsHref="http://localhost:3001/docs/getting-started/quickstart"
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

            {showModal && (
                <div className="modal-overlay" onClick={closeCreateModal}>
                    <div
                        className="modal project-list-create-modal"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="modal-header">
                            <h2 className="modal-title">New BrewSLM Project</h2>
                            <button className="btn btn-ghost" onClick={closeCreateModal}>✕</button>
                        </div>

                        <div className="modal-body">
                            <div className="form-group">
                                <label className="form-label">Project Name *</label>
                                <input
                                    className="input"
                                    placeholder="e.g. Support FAQ Assistant"
                                    value={newName}
                                    onChange={(e) => setNewName(e.target.value)}
                                    autoFocus
                                    data-testid="create-project-name"
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">
                                    What problem do you want this model to solve? *
                                </label>
                                <textarea
                                    className="input"
                                    rows={5}
                                    placeholder="e.g. Answer customer support FAQs from our resolved tickets. Friendly tone. Never hallucinate beyond the dataset."
                                    value={briefText}
                                    onChange={(e) => setBriefText(e.target.value)}
                                    data-testid="create-project-brief"
                                />
                                <div className="form-hint">
                                    We'll analyze this and set sensible defaults — base model,
                                    task profile, output schema. You can change anything later.
                                </div>
                            </div>

                            <button
                                type="button"
                                className="btn btn-ghost"
                                onClick={() => setShowAdvanced((v) => !v)}
                                data-testid="create-project-advanced-toggle"
                                style={{ alignSelf: 'flex-start', padding: 0, fontSize: '0.85rem' }}
                            >
                                {showAdvanced ? '▼' : '▶'} Advanced options
                            </button>

                            {showAdvanced && (
                                <div
                                    data-testid="create-project-advanced"
                                    style={{
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: 'var(--space-md)',
                                        marginTop: 'var(--space-sm)',
                                        paddingTop: 'var(--space-md)',
                                        borderTop: '1px solid var(--border-color)',
                                    }}
                                >
                                    <div className="form-group">
                                        <label className="form-label">Description override</label>
                                        <input
                                            className="input"
                                            placeholder="Optional. If empty, the inferred problem statement is used."
                                            value={newDesc}
                                            onChange={(e) => setNewDesc(e.target.value)}
                                        />
                                    </div>
                                    <div className="project-list-grid-2">
                                        <div className="form-group">
                                            <label className="form-label">Sample inputs</label>
                                            <textarea
                                                className="input"
                                                rows={4}
                                                placeholder="One example per line. Helps the brief analyzer infer output shape."
                                                value={sampleInputsText}
                                                onChange={(e) => setSampleInputsText(e.target.value)}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label">Sample outputs</label>
                                            <textarea
                                                className="input"
                                                rows={4}
                                                placeholder='One example per line (plain text or JSON, e.g. {"label":"urgent"})'
                                                value={sampleOutputsText}
                                                onChange={(e) => setSampleOutputsText(e.target.value)}
                                            />
                                        </div>
                                    </div>
                                    <div className="project-list-grid-2">
                                        <div className="form-group">
                                            <label className="form-label">Base model</label>
                                            <input
                                                className="input"
                                                placeholder="e.g. HuggingFaceTB/SmolLM2-135M-Instruct"
                                                value={newModel}
                                                onChange={(e) => setNewModel(e.target.value)}
                                            />
                                            <div className="form-hint">
                                                HuggingFace model ID (135M–8B). Leave blank to
                                                inherit from your recipe.
                                            </div>
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label">Deployment target</label>
                                            <select
                                                className="input"
                                                value={deploymentTarget}
                                                onChange={(e) => setDeploymentTarget(e.target.value)}
                                            >
                                                <option value="vllm_server">vLLM Server</option>
                                                <option value="edge_gpu">Edge GPU</option>
                                                <option value="mobile_cpu">Mobile CPU</option>
                                                <option value="browser_webgpu">Browser WebGPU</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Starter pack</label>
                                        <select
                                            className="input"
                                            value={newStarterPackId}
                                            onChange={(e) => setNewStarterPackId(e.target.value)}
                                        >
                                            <option value="">No starter pack</option>
                                            {starterPacks.map((pack) => (
                                                <option key={pack.id} value={pack.id}>
                                                    {pack.display_name} ({pack.id})
                                                </option>
                                            ))}
                                        </select>
                                        <div className="form-hint">
                                            Optional domain defaults for model family, adapter
                                            profile, evaluation gates, and safety reminders.
                                        </div>
                                    </div>
                                    {selectedStarterPack && (
                                        <div className="project-list-starter-summary">
                                            <div className="project-list-starter-title">
                                                {selectedStarterPack.display_name}
                                            </div>
                                            <div className="project-list-starter-copy">
                                                {selectedStarterPack.description}
                                            </div>
                                            <div className="project-list-starter-meta">
                                                <span>Target default: {selectedStarterPack.target_profile_default}</span>
                                                {selectedStarterPack.default_base_model_name && (
                                                    <span>Base model default: {selectedStarterPack.default_base_model_name}</span>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                    <div className="project-list-grid-2">
                                        <div className="form-group">
                                            <label className="form-label"><Term id="domain_pack" advanced /></label>
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
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label"><Term id="domain_profile" advanced /></label>
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
                                        </div>
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Safety / compliance notes</label>
                                        <textarea
                                            className="input"
                                            rows={3}
                                            placeholder="One note per line (e.g. no PHI leakage, no legal advice)"
                                            value={riskNotesText}
                                            onChange={(e) => setRiskNotesText(e.target.value)}
                                        />
                                    </div>
                                </div>
                            )}

                            {analyzeError && (
                                <div className="project-list-analyze-error" data-testid="create-project-error">
                                    {analyzeError}
                                </div>
                            )}
                        </div>

                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={closeCreateModal}>Cancel</button>
                            <button
                                className="btn btn-primary"
                                onClick={handleCreate}
                                disabled={isCreating || !newName.trim() || !briefText.trim()}
                                data-testid="create-project-submit"
                            >
                                {isCreating ? 'Creating…' : 'Create project'}
                            </button>
                        </div>
                    </div>
                </div>
            )}


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
