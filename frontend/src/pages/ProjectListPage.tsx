/**
 * Dashboard page for creating, browsing, filtering, and deleting projects with approach recommendation.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import api from '../api/client';
import {
    analyzeBrief,
    type ApproachRecommendation,
} from '../api/blueprintAnalyze';
import DemoProjectTiles from '../components/dashboard/DemoProjectTiles';
import ProjectCard from '../components/dashboard/ProjectCard';
import TopBar from '../components/layout/TopBar';
import ApproachChip from '../components/projects/ApproachChip';
import ProjectTemplateGallery from '../components/projects/ProjectTemplateGallery';
import EmptyState from '../components/shared/EmptyState';
import Skeleton from '../components/shared/Skeleton';
import Term from '../components/shared/Term';
import { useProjectStore } from '../stores/projectStore';
import { projectPath } from '../utils/workspaceRoutes';
import { isProjectNameTakenError, MAX_DERIVED_NAME_RETRIES, uniqueNameFromBrief } from './projectNameFromBrief';
import type {
    DomainPackSummary,
    DomainProfileSummary,
    StarterPackCatalogResponse,
    StarterPackSummary,
} from '../types';
import './ProjectListPage.css';

/** Minimum brief length before the debounced decision-engine call
 * fires. Below this threshold the brief is almost certainly too
 * short to recommend anything useful — and we'd flicker the chip on
 * every keystroke for no signal. */
const APPROACH_CHIP_MIN_CHARS = 40;
/** Debounce window before firing the analyze call. The decision
 * engine itself is pure-Python and returns in <10 ms; this delay
 * is purely about not hammering the endpoint per keystroke. */
const APPROACH_CHIP_DEBOUNCE_MS = 600;

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
    const [deploymentTarget, setDeploymentTarget] = useState('vllm_server');
    const [analyzeError, setAnalyzeError] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Decision-engine chip state (Theme 7). Debounced analyze call
    // fires whenever the brief crosses the minimum length threshold.
    // Advice only (phase 7b): Create sets up a fine-tune either way.
    const [approachRec, setApproachRec] = useState<ApproachRecommendation | null>(null);
    const [approachLoading, setApproachLoading] = useState(false);
    // The base model the brief's task recipe suggests (the analyze call): the create applies it when the field is empty.
    const [suggestedBaseModel, setSuggestedBaseModel] = useState<string | null>(null);

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
    // What an empty base-model field resolves to on create (review S3): a picked domain preset's (starter pack) default wins over the
    // brief's task recipe, and the create re-analyzes the brief with the sample inputs / outputs.
    const packBaseModel = (selectedStarterPack?.default_base_model_name ?? '').trim() || null;

    const filteredProjects = projects
        .filter((project) => statusFilter === 'all' || project.status === statusFilter)
        .filter((project) => {
            if (!searchQuery) return true;
            const query = searchQuery.toLowerCase();
            return project.name.toLowerCase().includes(query) || project.description?.toLowerCase().includes(query);
        });

    // Phase 6: the name is optional. Empty field = the name derived from the
    // brief (shown under the field, "Edit" copies it in); the payload sends it
    // exactly like a typed name.
    const derivedName = useMemo(
        () => uniqueNameFromBrief(briefText, projects.map((project) => project.name)),
        [briefText, projects],
    );
    const effectiveName = newName.trim() || derivedName;
    const nameInputRef = useRef<HTMLInputElement>(null);

    const parsedSampleInputs = useMemo(() => parseMultiline(sampleInputsText), [sampleInputsText]);
    const parsedSampleOutputs = useMemo(() => parseMultiline(sampleOutputsText), [sampleOutputsText]);
    const hasSamples = parsedSampleInputs.length > 0 || parsedSampleOutputs.length > 0;

    // Theme 7 decision engine — debounced analyze call. Runs only
    // while the create modal is open and the brief is long enough to
    // produce meaningful signal. Cancels on every brief edit so the
    // user doesn't see a stale chip mid-typing.
    useEffect(() => {
        if (!showModal) return;
        const trimmed = briefText.trim();
        if (trimmed.length < APPROACH_CHIP_MIN_CHARS) {
            setApproachRec(null);
            setSuggestedBaseModel(null);
            setApproachLoading(false);
            return;
        }
        setApproachLoading(true);
        let cancelled = false;
        const timer = window.setTimeout(() => {
            analyzeBrief({
                brief_text: trimmed,
                deployment_target: deploymentTarget,
                llm_enrich: false,
            })
                .then((res) => {
                    if (cancelled) return;
                    setApproachRec(res.recommended_approach ?? null);
                    setSuggestedBaseModel(res.suggested_base_model ?? null);
                })
                .catch(() => {
                    if (cancelled) return;
                    // Best-effort: chip is advisory, not blocking.
                    // Silently swallow errors so the create button
                    // stays unblocked.
                    setApproachRec(null);
                    setSuggestedBaseModel(null);
                })
                .finally(() => {
                    if (cancelled) return;
                    setApproachLoading(false);
                });
        }, APPROACH_CHIP_DEBOUNCE_MS);
        return () => {
            cancelled = true;
            window.clearTimeout(timer);
        };
    }, [briefText, deploymentTarget, showModal]);

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
        setDeploymentTarget('vllm_server');
        setAnalyzeError('');
        setShowAdvanced(false);
        setApproachRec(null);
        setApproachLoading(false);
        setSuggestedBaseModel(null);
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
        const trimmedBrief = briefText.trim();
        if (!effectiveName || !trimmedBrief) return;
        setIsCreating(true);
        setAnalyzeError('');
        try {
            const starterPackId = newStarterPackId.trim() ? newStarterPackId.trim() : null;
            const domainPackId = newDomainPackId ? Number(newDomainPackId) : null;
            const domainProfileId = newDomainProfileId ? Number(newDomainProfileId) : null;

            const submit = (name: string) => createProject(
                name,
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
            // A derived name was checked only against the loaded projects (capped
            // and filtered by membership), so the server may still hold it: retry
            // with the next suffix a few times. A typed name never auto-retries.
            const nameWasDerived = !newName.trim();
            const taken = projects.map((item) => item.name);
            const createWithRetry = async () => {
                let name = effectiveName;
                for (let attempt = 0; ; attempt += 1) {
                    try {
                        return await submit(name);
                    } catch (error) {
                        if (!nameWasDerived || attempt >= MAX_DERIVED_NAME_RETRIES || !isProjectNameTakenError(error)) throw error;
                        taken.push(name);
                        name = uniqueNameFromBrief(trimmedBrief, taken);
                    }
                }
            };
            const project = await createWithRetry();
            closeCreateModal();
            // A new project has no data yet, so it lands on its first step,
            // Data › Sources (UI simplification phase 6; the Overview is one
            // click away in the sidebar). Samples and templates, which arrive
            // with data, still land on the Overview.
            navigate(projectPath(project.id, 'data'));
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

                {/* One intake: "+ New Project" (describe the goal → we set it
                    up, beginner mode on), or start from a ready-made sample. */}
                <section className="project-list-samples" data-testid="project-list-samples">
                    <h2 className="project-list-samples__title">Or begin with a starter project</h2>
                    <DemoProjectTiles />
                    <ProjectTemplateGallery hideWhenEmpty />
                </section>

                {isLoadingProjects ? (
                    <div className="project-grid">
                        {[1, 2, 3].map((i) => (
                            <Skeleton key={i} variant="card" />
                        ))}
                    </div>
                ) : projects.length === 0 ? (
                    <EmptyState
                        title="No projects yet"
                        description="Describe what you want your model to do and BrewSLM sets up the project for you — or start from one of the starter projects above."
                        primary={{ label: '+ Create First Project', onClick: openCreateModal }}
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
                            <button className="btn btn-ghost" onClick={closeCreateModal} aria-label="Close"><X size={16} aria-hidden="true" /></button>
                        </div>

                        <div className="modal-body">
                            <div className="form-group">
                                <label className="form-label" htmlFor="create-project-name">Project name (optional)</label>
                                <input
                                    id="create-project-name"
                                    ref={nameInputRef}
                                    className="input"
                                    placeholder={derivedName || 'e.g. Support FAQ Assistant'}
                                    value={newName}
                                    onChange={(e) => setNewName(e.target.value)}
                                    data-testid="create-project-name"
                                />
                                {!newName.trim() && (
                                    derivedName ? (
                                        <div className="form-hint" data-testid="create-project-derived-name">
                                            Will be named <strong>{derivedName}</strong> (from the problem below).{' '}
                                            <button
                                                type="button"
                                                className="btn btn-ghost btn-sm"
                                                onClick={() => {
                                                    setNewName(derivedName);
                                                    nameInputRef.current?.focus();
                                                }}
                                                data-testid="create-project-derived-name-edit"
                                            >
                                                Edit
                                            </button>
                                        </div>
                                    ) : (
                                        <div className="form-hint" data-testid="create-project-name-hint">
                                            {briefText.trim()
                                                ? 'Give the project a name: none can be made from the problem below.'
                                                : 'Leave it empty to name the project from the problem below.'}
                                        </div>
                                    )
                                )}
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
                                    autoFocus
                                    data-testid="create-project-brief"
                                />
                                <div className="form-hint">
                                    We'll analyze this and set sensible defaults — base model,
                                    task profile, output schema. You can change anything later.
                                </div>
                                <ApproachChip
                                    recommendation={approachRec}
                                    loading={approachLoading}
                                />
                            </div>

                            {/* Phase 7b (gap N7): the base model on the main step, not only under Advanced options. */}
                            <div className="form-group">
                                <label className="form-label" htmlFor="create-project-base-model">Base model (optional)</label>
                                <input
                                    id="create-project-base-model"
                                    className="input"
                                    placeholder={packBaseModel
                                        ? `${packBaseModel} (the domain preset's default)`
                                        : suggestedBaseModel ? `${suggestedBaseModel} (suggested from the brief)` : 'e.g. HuggingFaceTB/SmolLM2-135M-Instruct'}
                                    value={newModel}
                                    onChange={(e) => setNewModel(e.target.value)}
                                    data-testid="create-project-base-model"
                                />
                                <div className="form-hint" data-testid="create-project-base-model-hint">
                                    {newModel.trim()
                                        ? 'HuggingFace model ID (135M–8B); this one is used instead of the suggestion.'
                                        : packBaseModel
                                            ? <>Leave it empty to use <strong>{packBaseModel}</strong>, the default of the domain preset {selectedStarterPack?.display_name}. Any HuggingFace model ID (135M–8B) works.</>
                                            : suggestedBaseModel
                                                ? (hasSamples
                                                    ? <>Leave it empty to use the model suggested for the task. From the brief alone that is <strong>{suggestedBaseModel}</strong>; Create reads your samples too, which can change it.</>
                                                    : <>Leave it empty to use <strong>{suggestedBaseModel}</strong>, suggested from the brief alone. Any HuggingFace model ID (135M–8B) works.</>)
                                                : 'HuggingFace model ID (135M–8B). Leave it empty to use the one suggested for your task.'}
                                </div>
                            </div>

                            <button
                                type="button"
                                onClick={() => setShowAdvanced((v) => !v)}
                                data-testid="create-project-advanced-toggle"
                                className="btn btn-ghost project-list-advanced-toggle"
                            >
                                {showAdvanced ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />} Advanced options
                            </button>

                            {showAdvanced && (
                                <div
                                    data-testid="create-project-advanced"
                                    className="project-list-advanced"
                                >
                                    <p className="form-hint" data-testid="create-project-manifest-link">
                                        Have a <code>brewslm.yaml</code>?{' '}
                                        <button
                                            type="button"
                                            className="btn btn-ghost btn-sm"
                                            onClick={() => { closeCreateModal(); navigate('/manifest/import'); }}
                                        >
                                            Import it instead
                                        </button>
                                    </p>
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
                                            <label className="form-label">Deployment target</label>
                                            <select
                                                className="input"
                                                value={deploymentTarget}
                                                onChange={(e) => setDeploymentTarget(e.target.value)}
                                            >
                                                <option value="vllm_server">vLLM Server</option>
                                                <option value="vllm_server_large">vLLM Server (large GPU)</option>
                                                <option value="edge_gpu">Edge GPU</option>
                                                <option value="mobile_cpu">Mobile CPU</option>
                                                <option value="browser_webgpu">Browser WebGPU</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Domain preset</label>
                                        <select
                                            className="input"
                                            value={newStarterPackId}
                                            onChange={(e) => setNewStarterPackId(e.target.value)}
                                        >
                                            <option value="">No domain preset</option>
                                            {starterPacks.map((pack) => (
                                                <option key={pack.id} value={pack.id}>
                                                    {pack.display_name} ({pack.id})
                                                </option>
                                            ))}
                                        </select>
                                        <div className="form-hint">
                                            Optional domain preset: defaults for model family, adapter
                                            profile, pass/fail rules, and safety reminders.
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
                                disabled={isCreating || !effectiveName || !briefText.trim()}
                                data-testid="create-project-submit"
                            >
                                {isCreating ? 'Creating…' : 'Create project'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
