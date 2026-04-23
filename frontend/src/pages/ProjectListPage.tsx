import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import api from '../api/client';
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

interface GlossaryEntry {
    term?: string;
    plain_language?: string;
    category?: string;
    example?: string;
}

interface DomainBlueprintAnalyzeResponse {
    blueprint?: {
        task_family?: string;
        input_modality?: string;
        expected_output_schema?: Record<string, unknown>;
        unresolved_assumptions?: string[];
        glossary?: GlossaryEntry[];
        confidence_score?: number;
        [key: string]: unknown;
    };
    validation?: {
        ok?: boolean;
        errors?: Array<{ message?: string }>;
        warnings?: Array<{ message?: string }>;
    };
    guidance?: {
        recommended_next_actions?: string[];
        unresolved_questions?: string[];
    };
}

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

    const [beginnerMode, setBeginnerMode] = useState(true);
    const [beginnerStep, setBeginnerStep] = useState(1);
    const [briefText, setBriefText] = useState('');
    const [domainName, setDomainName] = useState('');
    const [targetPersona, setTargetPersona] = useState('');
    const [sampleInputsText, setSampleInputsText] = useState('');
    const [sampleOutputsText, setSampleOutputsText] = useState('');
    const [riskNotesText, setRiskNotesText] = useState('');
    const [deploymentTarget, setDeploymentTarget] = useState('vllm_server');
    const [analyzeError, setAnalyzeError] = useState('');
    const [analyzeLoading, setAnalyzeLoading] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<DomainBlueprintAnalyzeResponse | null>(null);

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

        setBeginnerMode(true);
        setBeginnerStep(1);
        setBriefText('');
        setDomainName('');
        setTargetPersona('');
        setSampleInputsText('');
        setSampleOutputsText('');
        setRiskNotesText('');
        setDeploymentTarget('vllm_server');
        setAnalyzeError('');
        setAnalyzeLoading(false);
        setAnalysisResult(null);
    };

    const openCreateModal = () => {
        resetCreateModal();
        setShowModal(true);
    };

    const closeCreateModal = () => {
        setShowModal(false);
        resetCreateModal();
    };

    const runBlueprintAnalyze = async (): Promise<DomainBlueprintAnalyzeResponse | null> => {
        setAnalyzeError('');
        setAnalyzeLoading(true);
        try {
            const res = await api.post<DomainBlueprintAnalyzeResponse>('/domain-blueprints/analyze', {
                brief_text: briefText.trim(),
                domain_name: domainName.trim() || undefined,
                target_user_persona: targetPersona.trim() || undefined,
                sample_inputs: parsedSampleInputs,
                sample_outputs: parsedSampleOutputs,
                risk_constraints: parsedRiskNotes,
                safety_compliance_notes: parsedRiskNotes,
                deployment_target: deploymentTarget,
                llm_enrich: true,
            });
            setAnalysisResult(res.data);
            return res.data;
        } catch (error: any) {
            const detail = error?.response?.data?.detail;
            setAnalyzeError(
                typeof detail === 'string' ? detail : 'Could not analyze your brief. Please refine the text and try again.',
            );
            return null;
        } finally {
            setAnalyzeLoading(false);
        }
    };

    const handleCreate = async () => {
        if (!newName.trim()) return;
        setIsCreating(true);
        try {
            if (beginnerMode) {
                if (!briefText.trim()) {
                    setAnalyzeError('A plain-language brief is required in Beginner Mode.');
                    return;
                }
                let analysis = analysisResult;
                if (!analysis?.blueprint) {
                    analysis = await runBlueprintAnalyze();
                    if (!analysis?.blueprint) {
                        return;
                    }
                }

                const project = await createProject(
                    newName.trim(),
                    (newDesc.trim() || String(analysis.blueprint.problem_statement || '')).trim(),
                    newModel.trim(),
                    null,
                    null,
                    null,
                    {
                        beginnerMode: true,
                        briefText: briefText.trim(),
                        sampleInputs: parsedSampleInputs,
                        sampleOutputs: parsedSampleOutputs,
                        domainBlueprint: analysis.blueprint,
                        targetProfileId: deploymentTarget.trim() || null,
                    },
                );
                closeCreateModal();
                navigate(`/project/${project.id}`);
                return;
            }

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
            );
            closeCreateModal();
            navigate(`/project/${project.id}`);
        } catch (error: any) {
            const detail = error?.response?.data?.detail;
            if (typeof detail === 'string') {
                setAnalyzeError(detail);
            } else if (typeof detail?.message === 'string') {
                setAnalyzeError(detail.message);
            } else {
                setAnalyzeError('Project creation failed. Please review your inputs and try again.');
            }
        } finally {
            setIsCreating(false);
        }
    };

    const handleBeginnerNext = async () => {
        if (beginnerStep === 1) {
            if (!newName.trim()) return;
            if (!briefText.trim()) {
                setAnalyzeError('Please add a plain-language brief before continuing.');
                return;
            }
            setAnalyzeError('');
            setBeginnerStep(2);
            return;
        }
        if (beginnerStep === 2) {
            const analyzed = await runBlueprintAnalyze();
            if (!analyzed?.blueprint) return;
            setBeginnerStep(3);
            return;
        }
        await handleCreate();
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

    const glossaryEntries = analysisResult?.blueprint?.glossary || [];
    const unresolvedAssumptions = analysisResult?.blueprint?.unresolved_assumptions || [];
    const validationWarnings = analysisResult?.validation?.warnings || [];

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
                                <button className="btn btn-primary" onClick={openCreateModal}>
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

            {showModal && (
                <div className="modal-overlay" onClick={closeCreateModal}>
                    <div
                        className={`modal project-list-create-modal ${beginnerMode ? 'project-list-create-modal--wide' : ''}`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="modal-header">
                            <h2 className="modal-title">New BrewSLM Project</h2>
                            <button className="btn btn-ghost" onClick={closeCreateModal}>✕</button>
                        </div>

                        <div className="project-list-mode-toggle">
                            <button
                                className={`project-list-mode-pill ${beginnerMode ? 'active' : ''}`}
                                onClick={() => {
                                    setBeginnerMode(true);
                                    setBeginnerStep(1);
                                    setAnalyzeError('');
                                }}
                            >
                                Beginner Mode
                            </button>
                            <button
                                className={`project-list-mode-pill ${!beginnerMode ? 'active' : ''}`}
                                onClick={() => {
                                    setBeginnerMode(false);
                                    setAnalyzeError('');
                                }}
                            >
                                Advanced Mode
                            </button>
                        </div>

                        {beginnerMode ? (
                            <div className="modal-body project-list-beginner-body">
                                <div className="project-list-beginner-steps">
                                    <span className={`badge ${beginnerStep >= 1 ? 'badge-success' : 'badge-info'}`}>1. Brief</span>
                                    <span className={`badge ${beginnerStep >= 2 ? 'badge-success' : 'badge-info'}`}>2. Examples</span>
                                    <span className={`badge ${beginnerStep >= 3 ? 'badge-success' : 'badge-info'}`}>3. Review</span>
                                </div>

                                {beginnerStep === 1 && (
                                    <>
                                        <div className="form-group">
                                            <label className="form-label">Project Name *</label>
                                            <input
                                                className="input"
                                                placeholder="e.g. Support FAQ Assistant"
                                                value={newName}
                                                onChange={(e) => setNewName(e.target.value)}
                                                autoFocus
                                            />
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label">Plain-English Brief *</label>
                                            <textarea
                                                className="input"
                                                rows={4}
                                                placeholder="Describe what model behavior you want and what success looks like."
                                                value={briefText}
                                                onChange={(e) => setBriefText(e.target.value)}
                                            />
                                            <div className="form-hint">This is the main input used to infer task family, output contract, and assumptions.</div>
                                        </div>
                                        <div className="project-list-grid-2">
                                            <div className="form-group">
                                                <label className="form-label">Domain Name (optional)</label>
                                                <input
                                                    className="input"
                                                    placeholder="e.g. Legal, Healthcare, Customer Support"
                                                    value={domainName}
                                                    onChange={(e) => setDomainName(e.target.value)}
                                                />
                                            </div>
                                            <div className="form-group">
                                                <label className="form-label">Target User Persona (optional)</label>
                                                <input
                                                    className="input"
                                                    placeholder="e.g. Support agents, analysts, operators"
                                                    value={targetPersona}
                                                    onChange={(e) => setTargetPersona(e.target.value)}
                                                />
                                            </div>
                                        </div>
                                    </>
                                )}

                                {beginnerStep === 2 && (
                                    <>
                                        <div className="project-list-grid-2">
                                            <div className="form-group">
                                                <label className="form-label">Sample Inputs</label>
                                                <textarea
                                                    className="input"
                                                    rows={5}
                                                    placeholder="One example per line"
                                                    value={sampleInputsText}
                                                    onChange={(e) => setSampleInputsText(e.target.value)}
                                                />
                                            </div>
                                            <div className="form-group">
                                                <label className="form-label">Sample Outputs</label>
                                                <textarea
                                                    className="input"
                                                    rows={5}
                                                    placeholder='One example per line (plain text or JSON, e.g. {"label":"urgent"})'
                                                    value={sampleOutputsText}
                                                    onChange={(e) => setSampleOutputsText(e.target.value)}
                                                />
                                            </div>
                                        </div>
                                        <div className="project-list-grid-2">
                                            <div className="form-group">
                                                <label className="form-label">Safety / Compliance Notes</label>
                                                <textarea
                                                    className="input"
                                                    rows={3}
                                                    placeholder="One note per line (e.g. no PHI leakage, no legal advice)"
                                                    value={riskNotesText}
                                                    onChange={(e) => setRiskNotesText(e.target.value)}
                                                />
                                            </div>
                                            <div className="form-group">
                                                <label className="form-label">Deployment Target</label>
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
                                            <label className="form-label">Description Override (optional)</label>
                                            <input
                                                className="input"
                                                placeholder="If empty, inferred problem statement will be used."
                                                value={newDesc}
                                                onChange={(e) => setNewDesc(e.target.value)}
                                            />
                                        </div>
                                    </>
                                )}

                                {beginnerStep === 3 && (
                                    <div className="project-list-blueprint-review">
                                        <div className="project-list-blueprint-panels">
                                            <section className="project-list-blueprint-panel">
                                                <h3>What The System Understood</h3>
                                                <div className="project-list-blueprint-kv">
                                                    <span>Task Family</span>
                                                    <strong>{String(analysisResult?.blueprint?.task_family || 'n/a')}</strong>
                                                </div>
                                                <div className="project-list-blueprint-kv">
                                                    <span>Input Modality</span>
                                                    <strong>{String(analysisResult?.blueprint?.input_modality || 'n/a')}</strong>
                                                </div>
                                                <div className="project-list-blueprint-kv">
                                                    <span>Confidence</span>
                                                    <strong>
                                                        {typeof analysisResult?.blueprint?.confidence_score === 'number'
                                                            ? `${Math.round((analysisResult.blueprint.confidence_score || 0) * 100)}%`
                                                            : 'n/a'}
                                                    </strong>
                                                </div>
                                                <div className="project-list-blueprint-schema">
                                                    <h4>Output Contract</h4>
                                                    <pre>{JSON.stringify(analysisResult?.blueprint?.expected_output_schema || {}, null, 2)}</pre>
                                                </div>
                                            </section>

                                            <section className="project-list-blueprint-panel">
                                                <h3>Assumptions And Warnings</h3>
                                                {unresolvedAssumptions.length > 0 ? (
                                                    <ul>
                                                        {unresolvedAssumptions.map((item) => (
                                                            <li key={item}>{item}</li>
                                                        ))}
                                                    </ul>
                                                ) : (
                                                    <p>No unresolved assumptions detected.</p>
                                                )}
                                                {validationWarnings.length > 0 && (
                                                    <>
                                                        <h4>Validation Warnings</h4>
                                                        <ul>
                                                            {validationWarnings.map((item, idx) => (
                                                                <li key={`${idx}-${item.message || ''}`}>{item.message || 'warning'}</li>
                                                            ))}
                                                        </ul>
                                                    </>
                                                )}
                                                {(analysisResult?.guidance?.recommended_next_actions || []).length > 0 && (
                                                    <>
                                                        <h4>Recommended Next Actions</h4>
                                                        <ul>
                                                            {(analysisResult?.guidance?.recommended_next_actions || []).map((item) => (
                                                                <li key={item}>{item}</li>
                                                            ))}
                                                        </ul>
                                                    </>
                                                )}
                                            </section>
                                        </div>

                                        <section className="project-list-blueprint-panel project-list-blueprint-panel--full">
                                            <h3>Jargon Translator</h3>
                                            <div className="project-list-glossary-grid">
                                                {glossaryEntries.length > 0 ? glossaryEntries.map((entry) => (
                                                    <article key={`${entry.term || ''}-${entry.category || ''}`} className="project-list-glossary-card">
                                                        <h4>{entry.term || 'term'}</h4>
                                                        <p>{entry.plain_language || 'No explanation available.'}</p>
                                                        {entry.category && <span className="badge badge-info">{entry.category}</span>}
                                                    </article>
                                                )) : (
                                                    <p>No glossary entries available yet.</p>
                                                )}
                                            </div>
                                        </section>
                                    </div>
                                )}

                                {analyzeError && (
                                    <div className="project-list-analyze-error">{analyzeError}</div>
                                )}
                            </div>
                        ) : (
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
                                    <label className="form-label">Starter Pack</label>
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
                                        Optional domain defaults for model family, adapter profile, evaluation gates, and safety reminders.
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
                                            <span>Model families: {selectedStarterPack.recommended_model_families.join(', ') || 'n/a'}</span>
                                            <span>Target default: {selectedStarterPack.target_profile_default}</span>
                                            {selectedStarterPack.default_base_model_name && (
                                                <span>Base model default: {selectedStarterPack.default_base_model_name}</span>
                                            )}
                                        </div>
                                        {selectedStarterPack.safety_compliance_reminders.length > 0 && (
                                            <div className="project-list-starter-reminders">
                                                <strong>Safety reminders:</strong>{' '}
                                                {selectedStarterPack.safety_compliance_reminders.join(' ')}
                                            </div>
                                        )}
                                        <div className="form-hint">
                                            Starter defaults apply when you leave Base Model and target settings on auto/default.
                                        </div>
                                    </div>
                                )}
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
                                    <div className="form-hint">Optional pack-level defaults and overlays.</div>
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
                                    <div className="form-hint">Can be reassigned later in project settings.</div>
                                </div>
                            </div>
                        )}
                        {!beginnerMode && analyzeError && (
                            <div className="project-list-analyze-error">{analyzeError}</div>
                        )}

                        <div className="modal-footer">
                            <button className="btn btn-secondary" onClick={closeCreateModal}>Cancel</button>
                            {beginnerMode && beginnerStep > 1 && (
                                <button
                                    className="btn btn-secondary"
                                    onClick={() => setBeginnerStep((prev) => Math.max(1, prev - 1))}
                                >
                                    Back
                                </button>
                            )}
                            {beginnerMode ? (
                                <button
                                    className="btn btn-primary"
                                    onClick={handleBeginnerNext}
                                    disabled={
                                        isCreating
                                        || analyzeLoading
                                        || !newName.trim()
                                        || (beginnerStep === 1 && !briefText.trim())
                                    }
                                >
                                    {analyzeLoading
                                        ? 'Analyzing...'
                                        : isCreating
                                            ? 'Creating...'
                                            : beginnerStep < 3
                                                ? 'Next'
                                                : 'Create From Brief'}
                                </button>
                            ) : (
                                <button className="btn btn-primary" onClick={handleCreate} disabled={!newName.trim() || isCreating}>
                                    {isCreating ? 'Creating...' : 'Create Project'}
                                </button>
                            )}
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
