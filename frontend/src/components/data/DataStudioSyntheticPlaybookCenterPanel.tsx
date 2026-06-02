/**
 * Panel for recipe-aware synthetic generation playbooks with prerequisites and local backend readiness.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    Bot,
    CheckCircle2,
    ExternalLink,
    Library,
    ListChecks,
    RefreshCw,
    Sparkles,
} from 'lucide-react';

import {
    getDataStudioSyntheticPlaybookCenter,
} from '../../api/dataStudio';
import type {
    DataStudioSyntheticDomainLibrary,
    DataStudioSyntheticDomainPlaybook,
    DataStudioSyntheticPlaybookCenter,
    DataStudioSyntheticPrerequisite,
} from '../../api/dataStudio';
import './DataStudioSyntheticPlaybookCenterPanel.css';

interface DataStudioSyntheticPlaybookCenterPanelProps {
    projectId: number;
    onOpenSynthetic: () => void;
    /**
     * Arc C — when an unmet prerequisite has a clear setup
     * destination, the panel routes through this callback so the
     * user can act in one click instead of bouncing back to the
     * pipeline tabs and finding the right surface themselves.
     * Optional so legacy callers that only need ``onOpenSynthetic``
     * keep working without a no-op handler.
     */
    onOpenTarget?: (target: string) => void;
}

/**
 * Arc C — per-prerequisite setup affordance. The backend already
 * stamps a ``target_tab`` on each prerequisite, but the *button label*
 * needs to be specific ("Configure Ollama" beats "Open synthetic")
 * for the user to know what they'll see on the other side. This map
 * keys off ``prerequisite.id`` (stable contract from
 * data_studio_service._synthetic_library_prerequisites). Unknown ids
 * fall back to a generic "Set up" label.
 */
const PREREQUISITE_SETUP_LABEL: Record<string, string> = {
    recipe: 'Pick a recipe',
    playbook_mode: 'Pick a playbook mode',
    mapping: 'Fix mapping',
    gold_examples: 'Open Gold Set',
    local_ollama: 'Configure Ollama',
    review_gate: 'Review pending rows',
};

function prerequisiteSetupLabel(item: DataStudioSyntheticPrerequisite): string {
    return PREREQUISITE_SETUP_LABEL[item.id] || 'Set up';
}

const SYNTHETIC_VERDICT_COPY: Record<DataStudioSyntheticPlaybookCenter['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No playbooks',
        detail: 'Select a recipe to unlock recipe-aware synthetic playbooks.',
    },
    attention: {
        label: 'Needs setup',
        detail: 'Synthetic playbooks are available, but prerequisites or review queues need attention.',
    },
    ready: {
        label: 'Ready',
        detail: 'Playbooks, local backend, and review gates are ready for synthetic expansion.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function labelForStatus(status: string | undefined): string {
    if (!status) return 'Unknown';
    return status.replace(/_/g, ' ');
}

function readinessLabel(status: string | undefined): string {
    if (status === 'ready') return 'Ready';
    if (status === 'blocked') return 'Blocked';
    if (status === 'attention') return 'Needs review';
    return labelForStatus(status);
}

function librarySourceLabel(source: string | undefined): string {
    if (source === 'detected') return 'Detected domain';
    if (source === 'applied') return 'Applied domain';
    if (source === 'fallback') return 'Fallback';
    return labelForStatus(source);
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function prerequisiteIcon(item: DataStudioSyntheticPrerequisite) {
    if (item.status === 'met') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function issueIcon(severity: string) {
    if (severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function DomainPlaybookCard({
    playbook,
    onOpenSynthetic,
}: {
    playbook: DataStudioSyntheticDomainPlaybook;
    onOpenSynthetic: () => void;
}) {
    return (
        <article className={`data-studio-synth__domain-playbook data-studio-synth__domain-playbook--${playbook.readiness}`}>
            <div className="data-studio-synth__domain-playbook-head">
                <div>
                    <strong>{playbook.title}</strong>
                    <small>
                        {labelForStatus(playbook.strategy)}
                        {' · '}
                        {playbook.mode_label}
                    </small>
                </div>
                <span>{readinessLabel(playbook.readiness)}</span>
            </div>
            <p>{playbook.readiness_reason}</p>
            <div className="data-studio-synth__domain-chips">
                <span>{playbook.generation_path.available ? playbook.generation_path.describe : 'Ollama setup'}</span>
                <span>{playbook.generation_path.paid_required ? 'paid backend' : 'local default'}</span>
                <span>{playbook.mode_available ? 'mode available' : 'mode missing'}</span>
                <span>{playbook.recipe_compatible ? 'recipe fit' : 'recipe review'}</span>
            </div>
            <div className="data-studio-synth__domain-details">
                <div>
                    <b>Required fields</b>
                    <small>{playbook.required_fields.join(', ') || 'n/a'}</small>
                </div>
                <div>
                    <b>Output shape</b>
                    <small>
                        {playbook.expected_output_shape.format}
                        {' · '}
                        {playbook.expected_output_shape.payload_fields.slice(0, 5).join(', ')}
                    </small>
                </div>
            </div>
            {playbook.missing_fields.length > 0 ? (
                <p className="data-studio-synth__domain-warning">
                    Missing or ambiguous mapping fields: {playbook.missing_fields.join(', ')}
                </p>
            ) : null}
            <ul className="data-studio-synth__domain-gates">
                {playbook.review_gates.slice(0, 3).map((gate) => (
                    <li key={gate}>{gate}</li>
                ))}
            </ul>
            <button type="button" className="btn btn-secondary" onClick={onOpenSynthetic}>
                <ExternalLink size={15} aria-hidden="true" />
                {playbook.generation_action.label}
            </button>
        </article>
    );
}

function DomainLibraryCard({
    library,
    onOpenSynthetic,
}: {
    library: DataStudioSyntheticDomainLibrary;
    onOpenSynthetic: () => void;
}) {
    const primaryPlaybooks = library.playbooks.slice(0, 2);
    return (
        <article className={`data-studio-synth__domain-library data-studio-synth__domain-library--${library.status}`}>
            <div className="data-studio-synth__domain-library-head">
                <div>
                    <h5>{library.domain_label}</h5>
                    <p>{library.summary}</p>
                </div>
                <div className="data-studio-synth__domain-library-badges">
                    <span>{librarySourceLabel(library.source)}</span>
                    <span>{formatPercent(library.confidence)}</span>
                    <span>{readinessLabel(library.status)}</span>
                </div>
            </div>
            <div className="data-studio-synth__domain-chips">
                <span>{library.active_recipe_label}</span>
                <span>{library.local_first ? 'local-first' : 'remote capable'}</span>
                <span>
                    {library.compatible_modes.length}
                    {' compatible mode'}
                    {library.compatible_modes.length === 1 ? '' : 's'}
                </span>
                <span>
                    {library.recommended_recipes.length
                        ? `Recommended: ${library.recommended_recipes.join(', ')}`
                        : 'Any recipe'}
                </span>
            </div>
            <div className="data-studio-synth__domain-prereqs">
                {primaryPlaybooks[0]?.prerequisites.slice(0, 6).map((item) => (
                    <span
                        className={`data-studio-synth__domain-prereq data-studio-synth__domain-prereq--${item.status}`}
                        key={`${library.id}:${item.id}`}
                    >
                        {item.label}
                    </span>
                ))}
            </div>
            <div className="data-studio-synth__domain-playbooks">
                {primaryPlaybooks.map((playbook) => (
                    <DomainPlaybookCard
                        key={playbook.id}
                        playbook={playbook}
                        onOpenSynthetic={onOpenSynthetic}
                    />
                ))}
            </div>
        </article>
    );
}

export default function DataStudioSyntheticPlaybookCenterPanel({
    projectId,
    onOpenSynthetic,
    onOpenTarget,
}: DataStudioSyntheticPlaybookCenterPanelProps) {
    const [center, setCenter] = useState<DataStudioSyntheticPlaybookCenter | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadCenter = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioSyntheticPlaybookCenter(projectId);
            setCenter(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Synthetic Playbook Center.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadCenter();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => center?.issues.slice(0, 3) ?? [],
        [center],
    );
    const topPlaybooks = useMemo(
        () => center?.catalog.preview_playbooks.slice(0, 4) ?? [],
        [center],
    );
    const topPendingGroups = useMemo(
        () => center?.review_queue.top_pending_groups.slice(0, 3) ?? [],
        [center],
    );
    const domainLibraries = useMemo(
        () => center?.domain_libraries?.libraries.slice(0, 2) ?? [],
        [center],
    );

    if (loading && !center) {
        return (
            <section className="data-studio-synth data-studio-synth--loading">
                <span>Loading Synthetic Playbook Center...</span>
            </section>
        );
    }

    if (error && !center) {
        return (
            <section className="data-studio-synth data-studio-synth--error">
                <div>
                    <h3>Synthetic Playbook Center</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadCenter()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!center) {
        return null;
    }

    const verdict = SYNTHETIC_VERDICT_COPY[center.verdict];
    const recipeLabel = center.recipe?.name || center.recipe?.id || 'No recipe';
    const ollamaReady = center.recommended_backend.available;
    const backendLabel = ollamaReady ? center.recommended_backend.describe : 'Ollama not ready';

    return (
        <section
            className={`data-studio-synth data-studio-synth--${center.verdict}`}
            data-testid="data-studio-synth-playbooks"
        >
            <div className="data-studio-synth__header">
                <div>
                    <p className="data-studio-synth__eyebrow">Synthetic</p>
                    <h3>Synthetic Playbook Center</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-synth__actions">
                    <span className={`data-studio-synth__verdict data-studio-synth__verdict--${center.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-synth__refresh"
                        onClick={() => void loadCenter()}
                        aria-label="Refresh Synthetic Playbook Center"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-synth__metrics" aria-label="Synthetic playbook metrics">
                <div className="data-studio-synth__metric">
                    <Library size={18} aria-hidden="true" />
                    <span>Compatible playbooks</span>
                    <strong>
                        {formatNumber(center.catalog.compatible_playbooks)}
                        {' / '}
                        {formatNumber(center.catalog.total_playbooks)}
                    </strong>
                </div>
                <div className="data-studio-synth__metric">
                    <Bot size={18} aria-hidden="true" />
                    <span>Local default</span>
                    <strong>{backendLabel}</strong>
                </div>
                <div className="data-studio-synth__metric">
                    <ListChecks size={18} aria-hidden="true" />
                    <span>Pending review</span>
                    <strong>{formatNumber(center.review_queue.total_pending)}</strong>
                </div>
                <div className="data-studio-synth__metric">
                    <Sparkles size={18} aria-hidden="true" />
                    <span>Accepted synthetic</span>
                    <strong>{formatNumber(center.review_queue.total_accepted)}</strong>
                </div>
            </div>

            <div className="data-studio-synth__entry">
                <div>
                    <strong>{center.entry_point.label}</strong>
                    <small>
                        {recipeLabel}
                        {' · '}
                        {center.recommended_backend.paid_required ? 'paid backend' : 'free local default'}
                    </small>
                </div>
                <button type="button" className="btn btn-primary" onClick={onOpenSynthetic}>
                    <ExternalLink size={16} aria-hidden="true" />
                    Open Synthetic workflow
                </button>
            </div>

            <div className="data-studio-synth__domain-libraries">
                <div className="data-studio-synth__domain-libraries-head">
                    <div>
                        <h4>Domain playbook libraries</h4>
                        <p>
                            Curated local-first generation plans for the detected or applied domain.
                        </p>
                    </div>
                    <div className="data-studio-synth__domain-library-metrics">
                        <span>{formatNumber(center.domain_libraries?.library_count)} libraries</span>
                        <span>{center.domain_libraries?.ollama_ready ? 'Ollama ready' : 'Ollama setup'}</span>
                        <span>{center.domain_libraries?.read_only === false ? 'Can mutate' : 'Read-only'}</span>
                    </div>
                </div>
                {domainLibraries.length > 0 ? (
                    <div className="data-studio-synth__domain-library-list">
                        {domainLibraries.map((library) => (
                            <DomainLibraryCard
                                key={library.id}
                                library={library}
                                onOpenSynthetic={onOpenSynthetic}
                            />
                        ))}
                    </div>
                ) : (
                    <p className="data-studio-synth__empty">
                        Domain-specific libraries appear after BrewSLM can infer or apply a training domain.
                    </p>
                )}
            </div>

            <div className="data-studio-synth__body">
                <div className="data-studio-synth__playbooks">
                    <h4>Available playbooks</h4>
                    {topPlaybooks.length > 0 ? (
                        <div className="data-studio-synth__playbook-list">
                            {topPlaybooks.map((playbook) => (
                                <article
                                    className="data-studio-synth__playbook"
                                    key={`${playbook.recipe_id}-${playbook.mode}`}
                                >
                                    <strong>{playbook.label}</strong>
                                    <small>
                                        {playbook.recipe_id}
                                        {' · '}
                                        {labelForStatus(playbook.mode)}
                                    </small>
                                </article>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth__empty">
                            Pick a recipe to see compatible playbook modes.
                        </p>
                    )}

                    <div className="data-studio-synth__prerequisites">
                        {center.prerequisites.map((item) => {
                            // Arc C — unmet prerequisites become a one-
                            // click route to the surface that resolves
                            // them. Met prerequisites stay as static
                            // info rows (a button on a checked item
                            // would be confusing).
                            const isMet = item.status === 'met';
                            if (isMet || !onOpenTarget) {
                                return (
                                    <div
                                        className={`data-studio-synth__prereq data-studio-synth__prereq--${item.status}`}
                                        key={item.id}
                                    >
                                        <span>{prerequisiteIcon(item)}</span>
                                        <div>
                                            <strong>{item.label}</strong>
                                            <small>{item.message}</small>
                                        </div>
                                    </div>
                                );
                            }
                            return (
                                <button
                                    type="button"
                                    className={`data-studio-synth__prereq data-studio-synth__prereq--${item.status} data-studio-synth__prereq--actionable`}
                                    key={item.id}
                                    onClick={() => onOpenTarget(item.target_tab)}
                                    data-testid={`data-studio-synth-prereq-${item.id}`}
                                >
                                    <span>{prerequisiteIcon(item)}</span>
                                    <div>
                                        <strong>{item.label}</strong>
                                        <small>{item.message}</small>
                                    </div>
                                    <b>{prerequisiteSetupLabel(item)}</b>
                                </button>
                            );
                        })}
                    </div>
                </div>

                <div className="data-studio-synth__review">
                    <h4>Review queue</h4>
                    {topPendingGroups.length > 0 ? (
                        <div className="data-studio-synth__queue-list">
                            {topPendingGroups.map((group) => (
                                <article className="data-studio-synth__queue-group" key={group.synth_source}>
                                    <strong>{group.synth_source}</strong>
                                    <small>
                                        {formatNumber(group.count)} pending
                                        {group.truncated ? ' · sample shown' : ''}
                                    </small>
                                </article>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-synth__empty">
                            No synthetic rows are waiting for review.
                        </p>
                    )}

                    {topIssues.length > 0 ? (
                        <ul className="data-studio-synth__issues">
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-synth__issue data-studio-synth__issue--${issue.severity}`}>
                                    <span>{issueIcon(issue.severity)}</span>
                                    <div>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-synth__empty">
                            Playbook prerequisites are clear.
                        </p>
                    )}
                </div>
            </div>

            <details className="data-studio-synth__details">
                <summary>Power details</summary>
                <pre>{compactJson(center)}</pre>
            </details>
        </section>
    );
}
