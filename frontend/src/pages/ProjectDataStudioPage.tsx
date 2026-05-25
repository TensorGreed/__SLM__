import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useLocation, useNavigate, useOutletContext } from 'react-router-dom';
import { ChevronDown, ChevronRight, ExternalLink, Search } from 'lucide-react';

import DataStudioCoachRailPanel from '../components/data/DataStudioCoachRailPanel';
import DataStudioOverviewPanel from '../components/data/DataStudioOverviewPanel';
import DataStudioSourcesSummaryPanel from '../components/data/DataStudioSourcesSummaryPanel';
import DataStudioMappingPreviewPanel from '../components/data/DataStudioMappingPreviewPanel';
import DataStudioDomainDetectionPanel from '../components/data/DataStudioDomainDetectionPanel';
import DataStudioAssistPanel from '../components/data/DataStudioAssistPanel';
import DataStudioGoldSetWorkbenchPanel from '../components/data/DataStudioGoldSetWorkbenchPanel';
import DataStudioSyntheticPlaybookCenterPanel from '../components/data/DataStudioSyntheticPlaybookCenterPanel';
import DataStudioSyntheticRecommendationsPanel from '../components/data/DataStudioSyntheticRecommendationsPanel';
import DataStudioReviewQueuePanel from '../components/data/DataStudioReviewQueuePanel';
import DataStudioPrepareDatasetPanel from '../components/data/DataStudioPrepareDatasetPanel';
import DataStudioDatasetVersionsPanel from '../components/data/DataStudioDatasetVersionsPanel';
import type { DataStudioCoachCheck, DataStudioCoachRail } from '../api/dataStudio';
import { useProjectStore } from '../stores/projectStore';
import type { TabKey } from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectDataStudioPage.css';

const PIPELINE_TAB_KEYS: TabKey[] = [
    'data',
    'cleaning',
    'goldset',
    'synthetic',
    'dataprep',
    'tokenization',
    'training',
    'eval',
    'compression',
    'export',
];

const DATA_STUDIO_SECTION_IDS = [
    'overview',
    'sources',
    'mapping',
    'domain',
    'assist',
    'gold-set',
    'synthetic-playbooks',
    'synthetic-recommendations',
    'review-queue',
    'prepare-dataset',
    'dataset-versions',
] as const;

type DataStudioSectionId = typeof DATA_STUDIO_SECTION_IDS[number];

type DataStudioSectionGroup = 'start' | 'shape' | 'examples' | 'release';

interface DataStudioHandoffChip {
    label: string;
    target: string;
    sectionToken?: string | null;
    signalSectionId?: DataStudioSectionId;
}

type DataStudioHandoffSignalStatus = 'blocker' | 'attention' | 'ready';
type DataStudioTriageFilter = 'all' | DataStudioHandoffSignalStatus;

interface DataStudioHandoffSignal {
    status: DataStudioHandoffSignalStatus;
    label: string;
}

interface DataStudioSectionSignalSummary {
    directSignal: DataStudioHandoffSignal | null;
    statuses: Set<DataStudioHandoffSignalStatus>;
}

interface DataStudioSectionConfig {
    id: DataStudioSectionId;
    title: string;
    summary: string;
    group: DataStudioSectionGroup;
    keywords: string[];
    handoffs: DataStudioHandoffChip[];
    content: ReactNode;
}

const DEFAULT_EXPANDED_SECTIONS: DataStudioSectionId[] = ['overview', 'sources', 'mapping'];

const SECTION_GROUP_LABELS: Record<DataStudioSectionGroup, string> = {
    start: 'Start',
    shape: 'Shape',
    examples: 'Examples',
    release: 'Release',
};

const SECTION_GROUP_ORDER: DataStudioSectionGroup[] = ['start', 'shape', 'examples', 'release'];

const SECTION_TOKEN_ALIASES: Record<string, DataStudioSectionId> = {
    overview: 'overview',
    sources: 'sources',
    source: 'sources',
    mapping: 'mapping',
    'mapping-preview': 'mapping',
    domain: 'domain',
    'domain-detection': 'domain',
    assist: 'assist',
    'llm-assist': 'assist',
    gold: 'gold-set',
    goldset: 'gold-set',
    'gold-set': 'gold-set',
    'synthetic-playbooks': 'synthetic-playbooks',
    playbooks: 'synthetic-playbooks',
    synthetic: 'synthetic-playbooks',
    'synthetic-recommendations': 'synthetic-recommendations',
    recommendations: 'synthetic-recommendations',
    'review-queue': 'review-queue',
    review: 'review-queue',
    'prepare-dataset': 'prepare-dataset',
    prepare: 'prepare-dataset',
    dataprep: 'prepare-dataset',
    'dataset-versions': 'dataset-versions',
    versions: 'dataset-versions',
};

interface DataStudioSectionProps {
    id: DataStudioSectionId;
    title: string;
    summary: string;
    expanded: boolean;
    handoffs: DataStudioHandoffChip[];
    onToggle: (id: DataStudioSectionId) => void;
    onOpenHandoff: (handoff: DataStudioHandoffChip) => void;
    getHandoffSignal: (handoff: DataStudioHandoffChip, sectionId: DataStudioSectionId) => DataStudioHandoffSignal | null;
    triageFilter: DataStudioTriageFilter;
    sectionRef: (node: HTMLElement | null) => void;
    children: ReactNode;
}

const HANDOFF_SIGNAL_LABELS: Record<DataStudioHandoffSignalStatus, string> = {
    blocker: 'Blocker',
    attention: 'Attention',
    ready: 'Ready',
};

const TRIAGE_FILTERS: Array<{ id: DataStudioTriageFilter; label: string }> = [
    { id: 'all', label: 'All' },
    { id: 'blocker', label: 'Blockers' },
    { id: 'attention', label: 'Attention' },
    { id: 'ready', label: 'Ready' },
];

function isTabKey(value: string | undefined): value is TabKey {
    return !!value && PIPELINE_TAB_KEYS.includes(value as TabKey);
}

function storageKey(projectId: number): string {
    return `brewslm:data-studio:expanded-sections:${projectId}`;
}

function normalizeSectionToken(value: string | null | undefined): DataStudioSectionId | null {
    if (!value) {
        return null;
    }
    let decoded = value;
    try {
        decoded = decodeURIComponent(value);
    } catch {
        decoded = value;
    }
    const normalized = decoded.replace(/^#/, '').trim().toLowerCase().replace(/_/g, '-');
    return SECTION_TOKEN_ALIASES[normalized] ?? null;
}

function handoffSignalFromCheck(check: DataStudioCoachCheck): DataStudioHandoffSignal | null {
    const status = String(check.status || '').toLowerCase();
    let signalStatus: DataStudioHandoffSignalStatus | null = null;
    if (status === 'ready') {
        signalStatus = 'ready';
    } else if (status === 'blocked' || check.blocker_count > 0) {
        signalStatus = 'blocker';
    } else if (
        status === 'attention'
        || status === 'empty'
        || status === 'needs_work'
        || check.warning_count > 0
    ) {
        signalStatus = 'attention';
    }
    if (!signalStatus) {
        return null;
    }
    return {
        status: signalStatus,
        label: HANDOFF_SIGNAL_LABELS[signalStatus],
    };
}

function sectionIndexStatusLabel(
    summary: DataStudioSectionSignalSummary | undefined,
    expanded: boolean,
    triageFilter: DataStudioTriageFilter,
): string {
    const openState = expanded ? 'Open' : 'Closed';
    if (
        triageFilter !== 'all'
        && summary?.statuses.has(triageFilter)
        && summary.directSignal?.status !== triageFilter
    ) {
        return `${HANDOFF_SIGNAL_LABELS[triageFilter]} handoff · ${openState}`;
    }
    if (summary?.directSignal) {
        return `${summary.directSignal.label} · ${openState}`;
    }
    if (triageFilter !== 'all' && summary?.statuses.has(triageFilter)) {
        return `${HANDOFF_SIGNAL_LABELS[triageFilter]} handoff · ${openState}`;
    }
    return openState;
}

function readExpandedSections(projectId: number): Set<DataStudioSectionId> {
    if (typeof window === 'undefined') {
        return new Set(DEFAULT_EXPANDED_SECTIONS);
    }
    try {
        const raw = window.localStorage.getItem(storageKey(projectId));
        if (!raw) {
            return new Set(DEFAULT_EXPANDED_SECTIONS);
        }
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
            return new Set(DEFAULT_EXPANDED_SECTIONS);
        }
        const valid = parsed
            .map((item) => normalizeSectionToken(String(item)))
            .filter((item): item is DataStudioSectionId => item !== null);
        return new Set(valid);
    } catch {
        return new Set(DEFAULT_EXPANDED_SECTIONS);
    }
}

function writeExpandedSections(projectId: number, expanded: Set<DataStudioSectionId>): void {
    if (typeof window === 'undefined') {
        return;
    }
    try {
        window.localStorage.setItem(storageKey(projectId), JSON.stringify(Array.from(expanded)));
    } catch {
        // Ignore storage failures; expansion still works for this session.
    }
}

function DataStudioSection({
    id,
    title,
    summary,
    expanded,
    handoffs,
    onToggle,
    onOpenHandoff,
    getHandoffSignal,
    triageFilter,
    sectionRef,
    children,
}: DataStudioSectionProps) {
    const bodyId = `data-studio-section-${id}`;
    return (
        <section
            id={id}
            className="data-studio-page-section"
            ref={sectionRef}
            data-testid={`data-studio-section-${id}`}
        >
            <button
                type="button"
                className="data-studio-page-section__toggle"
                aria-expanded={expanded}
                aria-controls={bodyId}
                onClick={() => onToggle(id)}
            >
                <span className="data-studio-page-section__icon">
                    {expanded ? <ChevronDown size={16} aria-hidden="true" /> : <ChevronRight size={16} aria-hidden="true" />}
                </span>
                <span>
                    <strong>{title}</strong>
                    <small>{summary}</small>
                </span>
            </button>
            {handoffs.length > 0 ? (
                <div
                    className="data-studio-page-section__handoffs"
                    aria-label={`${title} workflow handoffs`}
                >
                    {handoffs.map((handoff) => {
                        const signal = getHandoffSignal(handoff, id);
                        const matchesActiveFilter = triageFilter !== 'all' && signal?.status === triageFilter;
                        const mutedByActiveFilter = triageFilter !== 'all' && signal?.status !== triageFilter;
                        return (
                            <button
                                type="button"
                                className={[
                                    'data-studio-page-section__handoff',
                                    signal ? `data-studio-page-section__handoff--${signal.status}` : '',
                                    matchesActiveFilter ? 'is-filter-match' : '',
                                    mutedByActiveFilter ? 'is-filter-muted' : '',
                                ].filter(Boolean).join(' ')}
                                data-status={signal?.status}
                                key={`${handoff.label}:${handoff.target}:${handoff.sectionToken ?? ''}`}
                                onClick={() => onOpenHandoff(handoff)}
                            >
                                <ExternalLink size={13} aria-hidden="true" />
                                <span>{handoff.label}</span>
                                {signal ? <small>{signal.label}</small> : null}
                            </button>
                        );
                    })}
                </div>
            ) : null}
            {expanded ? (
                <div id={bodyId} className="data-studio-page-section__body">
                    {children}
                </div>
            ) : null}
        </section>
    );
}

export default function ProjectDataStudioPage() {
    const navigate = useNavigate();
    const location = useLocation();
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const { setActiveTab } = useProjectStore();
    const previousProjectIdRef = useRef(projectId);
    const sectionRefs = useRef<Partial<Record<DataStudioSectionId, HTMLElement | null>>>({});
    const [expandedSections, setExpandedSections] = useState<Set<DataStudioSectionId>>(
        () => readExpandedSections(projectId),
    );
    const [sectionSearch, setSectionSearch] = useState('');
    const [triageFilter, setTriageFilter] = useState<DataStudioTriageFilter>('all');
    const [coachSignals, setCoachSignals] = useState<DataStudioCoachRail | null>(null);

    useEffect(() => {
        setExpandedSections(readExpandedSections(projectId));
        if (previousProjectIdRef.current !== projectId) {
            previousProjectIdRef.current = projectId;
            setCoachSignals(null);
        }
    }, [projectId]);

    const updateExpandedSections = useCallback(
        (updater: (current: Set<DataStudioSectionId>) => Set<DataStudioSectionId>) => {
            setExpandedSections((current) => {
                const next = updater(new Set(current));
                writeExpandedSections(projectId, next);
                return next;
            });
        },
        [projectId],
    );

    const scrollToSection = useCallback((sectionId: DataStudioSectionId) => {
        window.setTimeout(() => {
            sectionRefs.current[sectionId]?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
        }, 0);
    }, []);

    const setSectionRef = useCallback(
        (sectionId: DataStudioSectionId) => (node: HTMLElement | null) => {
            sectionRefs.current[sectionId] = node;
        },
        [],
    );

    const openDataStudioSection = useCallback(
        (sectionId: DataStudioSectionId, replace = false) => {
            updateExpandedSections((current) => {
                current.add(sectionId);
                return current;
            });
            navigate(`/project/${projectId}/data-studio#${sectionId}`, { replace });
            scrollToSection(sectionId);
        },
        [navigate, projectId, scrollToSection, updateExpandedSections],
    );

    useEffect(() => {
        const sectionId = normalizeSectionToken(location.hash);
        if (!sectionId) {
            return;
        }
        updateExpandedSections((current) => {
            current.add(sectionId);
            return current;
        });
        scrollToSection(sectionId);
    }, [location.hash, scrollToSection, updateExpandedSections]);

    const openPipelineTab = useCallback(
        (tabKey: TabKey) => {
            setActiveTab(tabKey);
            navigate(`/project/${projectId}/pipeline/${tabKey}`);
        },
        [navigate, projectId, setActiveTab],
    );

    const openDataStudioTarget = useCallback(
        (target: string, sectionToken?: string | null) => {
            const sectionId = normalizeSectionToken(sectionToken);
            if (sectionId) {
                openDataStudioSection(sectionId);
                return;
            }
            if (isTabKey(target)) {
                openPipelineTab(target);
                return;
            }
            if (target === 'annotate') {
                navigate(`/project/${projectId}/annotate`);
                return;
            }
            if (target === 'domain') {
                navigate(`/project/${projectId}/domain`);
                return;
            }
            if (target === 'domain-packs') {
                navigate(`/project/${projectId}/domain/packs`);
                return;
            }
            if (target === 'domain-profiles') {
                navigate(`/project/${projectId}/domain/profiles`);
                return;
            }
            const targetSectionId = normalizeSectionToken(target);
            if (targetSectionId) {
                openDataStudioSection(targetSectionId);
            }
        },
        [navigate, openDataStudioSection, openPipelineTab, projectId],
    );

    const toggleSection = useCallback(
        (sectionId: DataStudioSectionId) => {
            updateExpandedSections((current) => {
                if (current.has(sectionId)) {
                    current.delete(sectionId);
                } else {
                    current.add(sectionId);
                }
                return current;
            });
        },
        [updateExpandedSections],
    );

    const expandAll = useCallback(() => {
        updateExpandedSections(() => new Set(DATA_STUDIO_SECTION_IDS));
    }, [updateExpandedSections]);

    const collapseAll = useCallback(() => {
        updateExpandedSections(() => new Set());
    }, [updateExpandedSections]);

    const handoffSignalsBySection = useMemo(() => {
        const signals = new Map<DataStudioSectionId, DataStudioHandoffSignal>();
        coachSignals?.checks.forEach((check) => {
            const sectionId = normalizeSectionToken(check.id);
            const signal = handoffSignalFromCheck(check);
            if (sectionId && signal) {
                signals.set(sectionId, signal);
            }
        });
        return signals;
    }, [coachSignals]);

    const getHandoffSignal = useCallback(
        (handoff: DataStudioHandoffChip, fallbackSectionId: DataStudioSectionId) => (
            handoffSignalsBySection.get(handoff.signalSectionId ?? fallbackSectionId) ?? null
        ),
        [handoffSignalsBySection],
    );

    const sectionConfigs = useMemo<DataStudioSectionConfig[]>(
        () => [
            {
                id: 'overview' as const,
                title: 'Overview readiness',
                summary: 'One-page status for recipe, sources, domain, reviews, and the next action.',
                group: 'start' as const,
                keywords: ['status', 'readiness', 'recipe', 'next action', 'checks'],
                handoffs: [
                    { label: 'Source Ingestion', target: 'data', signalSectionId: 'sources' },
                    { label: 'Dataset Prep', target: 'dataprep', signalSectionId: 'prepare-dataset' },
                    { label: 'Training', target: 'training', signalSectionId: 'dataset-versions' },
                ],
                content: (
                    <DataStudioOverviewPanel
                        projectId={projectId}
                        onOpenTab={(targetTab) => {
                            if (isTabKey(targetTab)) {
                                openPipelineTab(targetTab);
                            }
                        }}
                    />
                ),
            },
            {
                id: 'sources' as const,
                title: 'Sources summary',
                summary: 'Read-only source coverage and ingestion health, with source upload kept in Pipeline Runs.',
                group: 'start' as const,
                keywords: ['sources', 'ingestion', 'datasets', 'documents', 'coverage'],
                handoffs: [
                    { label: 'Source Ingestion', target: 'data', signalSectionId: 'sources' },
                ],
                content: <DataStudioSourcesSummaryPanel projectId={projectId} />,
            },
            {
                id: 'mapping' as const,
                title: 'Schema mapping',
                summary: 'Recipe-aware mapping preview before Data Prep writes prepared artifacts.',
                group: 'shape' as const,
                keywords: ['schema', 'fields', 'mapping', 'contract', 'adapter'],
                handoffs: [
                    { label: 'Dataset Prep', target: 'dataprep', signalSectionId: 'mapping' },
                ],
                content: <DataStudioMappingPreviewPanel projectId={projectId} />,
            },
            {
                id: 'domain' as const,
                title: 'Domain detection',
                summary: 'Detected training domain, applied profile/pack alignment, and guided draft setup.',
                group: 'shape' as const,
                keywords: ['domain', 'profile', 'pack', 'policy', 'pii', 'pci'],
                handoffs: [
                    { label: 'Domain Managers', target: 'domain', signalSectionId: 'domain' },
                ],
                content: (
                    <DataStudioDomainDetectionPanel
                        projectId={projectId}
                        onOpenTarget={openDataStudioTarget}
                    />
                ),
            },
            {
                id: 'assist' as const,
                title: 'LLM assist',
                summary: 'Optional local Ollama or OpenAI-compatible explanations for mapping and domain choices.',
                group: 'shape' as const,
                keywords: ['llm', 'ollama', 'openai', 'assist', 'explain'],
                handoffs: [
                    { label: 'Domain Managers', target: 'domain', signalSectionId: 'domain' },
                    { label: 'Dataset Prep', target: 'dataprep', signalSectionId: 'mapping' },
                ],
                content: <DataStudioAssistPanel projectId={projectId} />,
            },
            {
                id: 'gold-set' as const,
                title: 'Gold Set workbench',
                summary: 'Trusted examples, validation status, field coverage, and review needs.',
                group: 'examples' as const,
                keywords: ['gold', 'trusted', 'examples', 'labels', 'validation'],
                handoffs: [
                    { label: 'Gold Set', target: 'goldset', signalSectionId: 'gold-set' },
                    { label: 'Review', target: 'annotate', signalSectionId: 'review-queue' },
                ],
                content: (
                    <DataStudioGoldSetWorkbenchPanel
                        projectId={projectId}
                        onOpenGoldSet={() => openPipelineTab('goldset')}
                    />
                ),
            },
            {
                id: 'synthetic-playbooks' as const,
                title: 'Synthetic Playbook Center',
                summary: 'Local-first generation readiness, playbook compatibility, and review prerequisites.',
                group: 'examples' as const,
                keywords: ['synthetic', 'playbooks', 'ollama', 'generation', 'local'],
                handoffs: [
                    { label: 'Synthetic', target: 'synthetic', signalSectionId: 'synthetic-playbooks' },
                ],
                content: (
                    <DataStudioSyntheticPlaybookCenterPanel
                        projectId={projectId}
                        onOpenSynthetic={() => openPipelineTab('synthetic')}
                    />
                ),
            },
            {
                id: 'synthetic-recommendations' as const,
                title: 'Synthetic recommendations',
                summary: 'Domain-aware strategies based on recipe, mappings, Gold Set, and review queue state.',
                group: 'examples' as const,
                keywords: ['synthetic', 'recommendations', 'strategy', 'domain', 'gold'],
                handoffs: [
                    { label: 'Synthetic', target: 'synthetic', signalSectionId: 'synthetic-recommendations' },
                    { label: 'Gold Set', target: 'goldset', signalSectionId: 'gold-set' },
                ],
                content: (
                    <DataStudioSyntheticRecommendationsPanel
                        projectId={projectId}
                        onOpenTab={(targetTab) => {
                            openDataStudioTarget(targetTab);
                        }}
                    />
                ),
            },
            {
                id: 'review-queue' as const,
                title: 'Review Queue',
                summary: 'Synthetic, Gold Set, promoted, and annotation review needs grouped for triage.',
                group: 'release' as const,
                keywords: ['review', 'queue', 'annotation', 'triage', 'promoted'],
                handoffs: [
                    { label: 'Review', target: 'annotate', signalSectionId: 'review-queue' },
                    { label: 'Synthetic', target: 'synthetic', signalSectionId: 'synthetic-playbooks' },
                    { label: 'Gold Set', target: 'goldset', signalSectionId: 'gold-set' },
                ],
                content: (
                    <DataStudioReviewQueuePanel
                        projectId={projectId}
                        onOpenTarget={openDataStudioTarget}
                    />
                ),
            },
            {
                id: 'prepare-dataset' as const,
                title: 'Prepare Dataset',
                summary: 'Readiness checks for recipe, mapping, splits, reviews, Gold Set, synthetic, and manifest outputs.',
                group: 'release' as const,
                keywords: ['prepare', 'dataset', 'splits', 'manifest', 'data prep'],
                handoffs: [
                    { label: 'Dataset Prep', target: 'dataprep', signalSectionId: 'prepare-dataset' },
                    { label: 'Review', target: 'annotate', signalSectionId: 'review-queue' },
                ],
                content: (
                    <DataStudioPrepareDatasetPanel
                        projectId={projectId}
                        onOpenTarget={openDataStudioTarget}
                    />
                ),
            },
            {
                id: 'dataset-versions' as const,
                title: 'Dataset Versions',
                summary: 'Prepared artifact versions, manifest metadata, reproducibility, and downstream reuse.',
                group: 'release' as const,
                keywords: ['versions', 'artifacts', 'manifest', 'training', 'eval'],
                handoffs: [
                    { label: 'Dataset Prep', target: 'dataprep', signalSectionId: 'dataset-versions' },
                    { label: 'Training', target: 'training', signalSectionId: 'dataset-versions' },
                    { label: 'Eval', target: 'eval', signalSectionId: 'dataset-versions' },
                ],
                content: (
                    <DataStudioDatasetVersionsPanel
                        projectId={projectId}
                        onOpenTarget={openDataStudioTarget}
                    />
                ),
            },
        ],
        [openDataStudioTarget, openPipelineTab, projectId],
    );

    const sectionSignalSummaries = useMemo(() => {
        const summaries = new Map<DataStudioSectionId, DataStudioSectionSignalSummary>();
        sectionConfigs.forEach((section) => {
            const statuses = new Set<DataStudioHandoffSignalStatus>();
            const directSignal = handoffSignalsBySection.get(section.id) ?? null;
            if (directSignal) {
                statuses.add(directSignal.status);
            }
            section.handoffs.forEach((handoff) => {
                const signal = getHandoffSignal(handoff, section.id);
                if (signal) {
                    statuses.add(signal.status);
                }
            });
            summaries.set(section.id, { directSignal, statuses });
        });
        return summaries;
    }, [getHandoffSignal, handoffSignalsBySection, sectionConfigs]);

    const triageFilterCounts = useMemo(() => {
        const counts: Record<DataStudioTriageFilter, number> = {
            all: sectionConfigs.length,
            blocker: 0,
            attention: 0,
            ready: 0,
        };
        sectionConfigs.forEach((section) => {
            const statuses = sectionSignalSummaries.get(section.id)?.statuses;
            if (!statuses) {
                return;
            }
            if (statuses.has('blocker')) counts.blocker += 1;
            if (statuses.has('attention')) counts.attention += 1;
            if (statuses.has('ready')) counts.ready += 1;
        });
        return counts;
    }, [sectionConfigs, sectionSignalSummaries]);

    const visibleSectionGroups = useMemo(() => {
        const query = sectionSearch.trim().toLowerCase();
        const matches = (section: DataStudioSectionConfig) => {
            const signalSummary = sectionSignalSummaries.get(section.id);
            if (!query) {
                return triageFilter === 'all' || !!signalSummary?.statuses.has(triageFilter);
            }
            const haystack = [
                section.title,
                section.summary,
                SECTION_GROUP_LABELS[section.group],
                ...section.keywords,
                signalSummary?.directSignal?.label ?? '',
                ...(signalSummary ? Array.from(signalSummary.statuses).map((status) => HANDOFF_SIGNAL_LABELS[status]) : []),
            ].join(' ').toLowerCase();
            const searchMatches = haystack.includes(query);
            const filterMatches = triageFilter === 'all' || !!signalSummary?.statuses.has(triageFilter);
            return searchMatches && filterMatches;
        };

        return SECTION_GROUP_ORDER.map((group) => ({
            group,
            sections: sectionConfigs.filter((section) => section.group === group && matches(section)),
        })).filter((group) => group.sections.length > 0);
    }, [sectionConfigs, sectionSearch, sectionSignalSummaries, triageFilter]);

    const openSectionCount = expandedSections.size;
    const activeSectionId = normalizeSectionToken(location.hash);

    return (
        <div className="workspace-page project-data-studio-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Data Studio</h2>
                    <p className="workspace-page-subtitle">
                        Inspect data readiness, guidance, reviews, synthetic strategy, and dataset version reuse before mutating pipeline stages.
                    </p>
                </div>
                <div className="workspace-page-header-actions">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => openPipelineTab('data')}
                    >
                        Open Source Ingestion
                    </button>
                </div>
            </section>

            <div className="data-studio-page">
                <DataStudioCoachRailPanel
                    projectId={projectId}
                    onOpenTarget={openDataStudioTarget}
                    onCoachLoaded={setCoachSignals}
                />

                <section className="data-studio-page-index" aria-label="Data Studio section index">
                    <div className="data-studio-page-index__head">
                        <div>
                            <h3>Jump to workbench</h3>
                            <p>
                                {sectionConfigs.length} sections
                                {' · '}
                                {openSectionCount} open
                            </p>
                        </div>
                        <label className="data-studio-page-index__search">
                            <Search size={15} aria-hidden="true" />
                            <input
                                type="search"
                                value={sectionSearch}
                                onChange={(event) => setSectionSearch(event.target.value)}
                                placeholder="Search sections"
                                aria-label="Search Data Studio sections"
                            />
                        </label>
                    </div>

                    <div className="data-studio-page-index__filters" role="group" aria-label="Data Studio triage filters">
                        {TRIAGE_FILTERS.map((filter) => {
                            const isActive = triageFilter === filter.id;
                            const count = triageFilterCounts[filter.id];
                            return (
                                <button
                                    type="button"
                                    key={filter.id}
                                    className={[
                                        'data-studio-page-index__filter',
                                        filter.id !== 'all' ? `data-studio-page-index__filter--${filter.id}` : '',
                                        isActive ? 'is-active' : '',
                                    ].filter(Boolean).join(' ')}
                                    aria-pressed={isActive}
                                    onClick={() => setTriageFilter(filter.id)}
                                >
                                    <span>{filter.label}</span>
                                    <small>{count}</small>
                                </button>
                            );
                        })}
                    </div>

                    {visibleSectionGroups.length > 0 ? (
                        <div className="data-studio-page-index__groups">
                            {visibleSectionGroups.map(({ group, sections }) => (
                                <div className="data-studio-page-index__group" key={group}>
                                    <span className="data-studio-page-index__group-label">
                                        {SECTION_GROUP_LABELS[group]}
                                    </span>
                                    <div className="data-studio-page-index__buttons">
                                        {sections.map((section) => {
                                            const summary = sectionSignalSummaries.get(section.id);
                                            const directStatus = summary?.directSignal?.status;
                                            const matchesActiveFilter = triageFilter !== 'all' && !!summary?.statuses.has(triageFilter);
                                            return (
                                                <button
                                                    type="button"
                                                    key={section.id}
                                                    className={[
                                                        'data-studio-page-index__button',
                                                        activeSectionId === section.id ? 'is-active' : '',
                                                        directStatus ? `data-studio-page-index__button--${directStatus}` : '',
                                                        matchesActiveFilter ? 'is-filter-match' : '',
                                                    ].filter(Boolean).join(' ')}
                                                    data-status={directStatus}
                                                    onClick={() => openDataStudioSection(section.id)}
                                                >
                                                    <span>{section.title}</span>
                                                    <small>{sectionIndexStatusLabel(summary, expandedSections.has(section.id), triageFilter)}</small>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-page-index__empty">No matching sections.</p>
                    )}
                </section>

                <section className="data-studio-page-controls" aria-label="Data Studio section controls">
                    <div>
                        <h3>Data Studio sections</h3>
                        <p>Keep high-signal checks open and expand deeper workbenches only when needed.</p>
                    </div>
                    <div className="data-studio-page-controls__actions">
                        <button type="button" className="btn btn-ghost" onClick={collapseAll}>
                            Collapse all
                        </button>
                        <button type="button" className="btn btn-secondary" onClick={expandAll}>
                            Expand all
                        </button>
                    </div>
                </section>

                {sectionConfigs.map((section) => (
                    <DataStudioSection
                        key={section.id}
                        id={section.id}
                        title={section.title}
                        summary={section.summary}
                        expanded={expandedSections.has(section.id)}
                        handoffs={section.handoffs}
                        onToggle={toggleSection}
                        onOpenHandoff={(handoff) => openDataStudioTarget(handoff.target, handoff.sectionToken)}
                        getHandoffSignal={getHandoffSignal}
                        triageFilter={triageFilter}
                        sectionRef={setSectionRef(section.id)}
                    >
                        {section.content}
                    </DataStudioSection>
                ))}
            </div>
        </div>
    );
}
