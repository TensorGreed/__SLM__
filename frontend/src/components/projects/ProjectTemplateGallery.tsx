/**
 * Project-template gallery — surfaces every available template on
 * the project-list page above the project grid. Each card has a
 * "Use this template" button that opens a small inline name-prompt;
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
import { useNavigate } from 'react-router-dom';

import {
    instantiateProjectTemplate,
    listProjectTemplates,
    type ProjectTemplateSummary,
} from '../../api/projectTemplates';
import { useToastStore } from '../../stores/toastStore';

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
            <section
                data-testid="project-template-gallery-loading"
                className="card"
                style={{ padding: 'var(--space-md)' }}
            >
                <div style={{ color: 'var(--text-secondary)' }}>
                    Loading templates…
                </div>
            </section>
        );
    }

    if (templates.length === 0) {
        if (hideWhenEmpty) return null;
        return (
            <section
                data-testid="project-template-gallery-empty"
                className="card"
                style={{ padding: 'var(--space-md)' }}
            >
                <div style={{ color: 'var(--text-secondary)' }}>
                    No project templates available yet.
                </div>
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
                `Created '${project.name}' from template`,
                'success',
                4000,
            );
            setActiveSlug(null);
            setProjectName('');
            navigate(`/project/${project.id}/guide`);
        } catch (err) {
            setError(extractErrorMessage(err));
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <section
            data-testid="project-template-gallery"
            className="card"
            style={{
                padding: 'var(--space-md)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-md)',
            }}
        >
            <div>
                <h3 style={{ margin: 0 }}>Start from a template</h3>
                <p
                    style={{
                        margin: '4px 0 0',
                        color: 'var(--text-secondary)',
                        fontSize: '0.9rem',
                    }}
                >
                    Curated starting kits with pre-loaded data, gold sets, and
                    recipe defaults. You can spin up multiple projects from the
                    same template — each gets its own data + experiments.
                </p>
            </div>

            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: 'var(--space-md)',
                }}
            >
                {templates.map((template) => {
                    const isActive = activeSlug === template.slug;
                    return (
                        <article
                            key={template.slug}
                            data-testid={`project-template-card-${template.slug}`}
                            style={{
                                padding: 'var(--space-md)',
                                borderRadius: 'var(--radius-md)',
                                border: isActive
                                    ? '2px solid var(--accent-primary)'
                                    : '1px solid var(--border-color)',
                                background: 'var(--bg-card)',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 'var(--space-sm)',
                            }}
                        >
                            <div
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 'var(--space-sm)',
                                }}
                            >
                                <span style={{ fontSize: '1.6rem' }} aria-hidden="true">
                                    {template.icon}
                                </span>
                                <div>
                                    <div style={{ fontWeight: 600 }}>
                                        {template.name}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: '0.8rem',
                                            color: 'var(--text-secondary)',
                                        }}
                                    >
                                        {template.task_profile} · target{' '}
                                        {template.target_profile}
                                    </div>
                                </div>
                            </div>

                            <p
                                style={{
                                    fontSize: '0.85rem',
                                    color: 'var(--text-secondary)',
                                    margin: 0,
                                }}
                                data-testid={`project-template-card-${template.slug}-headline`}
                            >
                                {template.headline}
                            </p>

                            <div
                                style={{
                                    display: 'flex',
                                    flexWrap: 'wrap',
                                    gap: 4,
                                    fontSize: '0.75rem',
                                    color: 'var(--text-secondary)',
                                }}
                            >
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
                                    style={{
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: 'var(--space-sm)',
                                        marginTop: 'var(--space-xs)',
                                    }}
                                    data-testid={`project-template-name-form-${template.slug}`}
                                >
                                    <label
                                        className="form-label"
                                        style={{ fontSize: '0.85rem' }}
                                    >
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
                                            style={{
                                                color: 'var(--color-error)',
                                                fontSize: '0.8rem',
                                            }}
                                            data-testid={`project-template-error-${template.slug}`}
                                        >
                                            {error}
                                        </div>
                                    )}
                                    <div
                                        style={{
                                            display: 'flex',
                                            gap: 'var(--space-sm)',
                                        }}
                                    >
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
                                    className="btn btn-secondary"
                                    onClick={() =>
                                        handlePickTemplate(template.slug, template.name)
                                    }
                                    data-testid={`project-template-pick-${template.slug}`}
                                    style={{ marginTop: 'auto' }}
                                >
                                    Use this template
                                </button>
                            )}
                        </article>
                    );
                })}
            </div>
        </section>
    );
}
