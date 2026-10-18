/**
 * Project-template gallery — surfaces every available template on
 * the project-list page above the project grid. Each card has a
 * "Use this starter" button that opens a small inline name-prompt;
 * confirming creates a new project from the template and navigates
 * the user into it.
 *
 * Templates are distinct from demo projects (the existing
 * DemoProjectTiles): demos are single-instance "try the platform"
 * experiences (one canonical "Demo · Support FAQ" project per
 * instance), templates are cloneable starting kits ("Acme Ticket
 * Router", "EU Ticket Router", … all from the same template).
 */

import { useEffect, useState } from 'react';
import {
    Database,
    FileSignature,
    Inbox,
    Mail,
    Package,
    ScrollText,
    Siren,
    Stethoscope,
    Wrench,
    type LucideIcon,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import {
    instantiateProjectTemplate,
    listProjectTemplates,
    type ProjectTemplateSummary,
} from '../../api/projectTemplates';
import { useToastStore } from '../../stores/toastStore';
import EmptyState from '../shared/EmptyState';
import LoadingState from '../shared/LoadingState';
import './ProjectTemplateGallery.css';

/** Template manifests name a lucide icon; anything else falls back to Package. */
const TEMPLATE_ICONS: Record<string, LucideIcon> = {
    wrench: Wrench,
    mail: Mail,
    'scroll-text': ScrollText,
    inbox: Inbox,
    siren: Siren,
    'file-signature': FileSignature,
    database: Database,
    stethoscope: Stethoscope,
    package: Package,
};

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) return detail;
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) return message;
    }
    return 'Unknown error';
}

interface ProjectTemplateGalleryProps {
    /** Hide the gallery section entirely when no templates exist
     * (e.g. test envs without the sample-data dir). The list-page
     * decides whether to render the wrapper card based on this. */
    hideWhenEmpty?: boolean;
}

export default function ProjectTemplateGallery({
    hideWhenEmpty = false,
}: ProjectTemplateGalleryProps) {
    const [templates, setTemplates] = useState<ProjectTemplateSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [activeSlug, setActiveSlug] = useState<string | null>(null);
    const [projectName, setProjectName] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string>('');
    const navigate = useNavigate();
    const { addToast } = useToastStore();

    useEffect(() => {
        let cancelled = false;
        listProjectTemplates()
            .then((res) => {
                if (!cancelled) setTemplates(res.templates ?? []);
            })
            .catch(() => {
                if (!cancelled) setTemplates([]);
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, []);

    if (loading) {
        return (
            <section data-testid="project-template-gallery-loading">
                <LoadingState label="Loading starter projects" lines={2} card />
            </section>
        );
    }

    if (templates.length === 0) {
        if (hideWhenEmpty) return null;
        return (
            <section data-testid="project-template-gallery-empty">
                <EmptyState
                    size="compact"
                    icon={<Package size={20} />}
                    title="No starter projects available yet."
                    description="Starter projects appear here once templates are installed."
                />
            </section>
        );
    }

    const handlePickTemplate = (slug: string, defaultName: string) => {
        setActiveSlug(slug);
        setProjectName(defaultName);
        setError('');
    };

    const handleCancel = () => {
        setActiveSlug(null);
        setProjectName('');
        setError('');
    };

    const handleSubmit = async () => {
        if (!activeSlug) return;
        const trimmed = projectName.trim();
        if (!trimmed) {
            setError('Pick a project name to continue.');
            return;
        }
        setSubmitting(true);
        setError('');
        try {
            const project = await instantiateProjectTemplate(
                activeSlug,
                trimmed,
            );
            addToast(
                `Created '${project.name}' from starter project`,
                'success',
                4000,
            );
            setActiveSlug(null);
            setProjectName('');
            navigate(`/project/${project.id}`);
        } catch (err) {
            setError(extractErrorMessage(err));
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <section data-testid="project-template-gallery" className="card tpl-gallery">
            <div>
                <h3 className="u-m0">Starter projects: with data and answer key</h3>
                <p className="u-meta tpl-gallery__intro">
                    Curated starting kits with pre-loaded data, answer keys, and
                    task type defaults. You can spin up multiple projects from the
                    same starter. Each gets its own data and experiments.
                </p>
            </div>

            <div className="tpl-gallery__grid">
                {templates.map((template) => {
                    const isActive = activeSlug === template.slug;
                    const Icon = TEMPLATE_ICONS[template.icon] ?? Package;
                    return (
                        <article
                            key={template.slug}
                            data-testid={`project-template-card-${template.slug}`}
                            className={`tpl-card${isActive ? ' tpl-card--active' : ''}`}
                        >
                            <div className="tpl-card__head">
                                <Icon size={22} aria-hidden="true" className="tpl-card__icon" />
                                <div>
                                    <div className="tpl-card__name">{template.name}</div>
                                    <div className="tpl-card__shape">
                                        {template.task_profile} · target{' '}
                                        {template.target_profile}
                                    </div>
                                </div>
                            </div>

                            <p
                                className="tpl-card__headline"
                                data-testid={`project-template-card-${template.slug}-headline`}
                            >
                                {template.headline}
                            </p>

                            <div className="tpl-card__badges">
                                {template.minimum_dataset_size > 0 && (
                                    <span className="badge badge-info">
                                        ≥ {template.minimum_dataset_size} rows
                                    </span>
                                )}
                                {template.recommended_base_models[0] && (
                                    <span className="badge badge-info">
                                        {template.recommended_base_models[0].split('/').pop()}
                                    </span>
                                )}
                                {template.labels.length > 0 && (
                                    <span className="badge badge-info">
                                        {template.labels.length}-way
                                    </span>
                                )}
                            </div>

                            {isActive ? (
                                <div
                                    className="tpl-card__form"
                                    data-testid={`project-template-name-form-${template.slug}`}
                                >
                                    <label className="form-label">
                                        Project name
                                    </label>
                                    <input
                                        className="input"
                                        value={projectName}
                                        onChange={(e) => setProjectName(e.target.value)}
                                        placeholder={`e.g. ${template.name} for Acme`}
                                        data-testid={`project-template-name-input-${template.slug}`}
                                        autoFocus
                                    />
                                    {error && (
                                        <div
                                            role="alert"
                                            className="tpl-card__error"
                                            data-testid={`project-template-error-${template.slug}`}
                                        >
                                            {error}
                                        </div>
                                    )}
                                    <div className="tpl-card__actions">
                                        <button
                                            type="button"
                                            className="btn btn-primary"
                                            onClick={handleSubmit}
                                            disabled={submitting}
                                            data-testid={`project-template-submit-${template.slug}`}
                                        >
                                            {submitting ? 'Creating…' : 'Create project'}
                                        </button>
                                        <button
                                            type="button"
                                            className="btn btn-ghost"
                                            onClick={handleCancel}
                                            disabled={submitting}
                                            data-testid={`project-template-cancel-${template.slug}`}
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <button
                                    type="button"
                                    className="btn btn-secondary tpl-card__pick"
                                    onClick={() =>
                                        handlePickTemplate(template.slug, template.name)
                                    }
                                    data-testid={`project-template-pick-${template.slug}`}
                                >
                                    Use this starter
                                </button>
                            )}
                        </article>
                    );
                })}
            </div>
        </section>
    );
}
