/**
 * EmptyState — shared component for "this page has no data yet" surfaces.
 *
 * Drops a centered card with an optional icon, a title that says what
 * the page is for, a 1–2 line description, up to two action buttons,
 * and an optional "Learn more →" docs link. Designed for newbies who
 * land on a blank table and don't know what to do next.
 *
 * Supports two prop shapes for backwards compatibility:
 *   <EmptyState icon="📂" title="..." description="..." action={<button/>} />
 *   <EmptyState
 *       icon={<Folder/>} title="..." description="..."
 *       primary={{ label: "Create", onClick: () => ... }}
 *       secondary={{ label: "Import", href: "/x" }}
 *       docsHref="https://localhost:3001/docs/..."
 *    />
 */

import type { ReactNode } from 'react';

import './EmptyState.css';

interface ActionDescriptor {
    label: string;
    onClick?: () => void;
    href?: string;
}

interface EmptyStateProps {
    /** Icon node (preferred) or emoji string (legacy). */
    icon?: ReactNode | string;
    title: string;
    description: string;
    /** Preferred action shape — renders an inverted-black primary button. */
    primary?: ActionDescriptor;
    /** Secondary action — renders as an outlined button next to primary. */
    secondary?: ActionDescriptor;
    /** Renders a small "Learn more →" link below the buttons. */
    docsHref?: string;
    /** Legacy free-form action slot. New code should use primary/secondary. */
    action?: ReactNode;
}

function ActionButton({
    descriptor,
    variant,
}: {
    descriptor: ActionDescriptor;
    variant: 'primary' | 'secondary';
}) {
    const className = `btn ${variant === 'primary' ? 'btn-primary' : 'btn-secondary'} btn-sm`;
    if (descriptor.href) {
        return (
            <a
                href={descriptor.href}
                target={descriptor.href.startsWith('http') ? '_blank' : undefined}
                rel={descriptor.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                className={className}
            >
                {descriptor.label}
            </a>
        );
    }
    return (
        <button type="button" className={className} onClick={descriptor.onClick}>
            {descriptor.label}
        </button>
    );
}

export default function EmptyState({
    icon,
    title,
    description,
    primary,
    secondary,
    docsHref,
    action,
}: EmptyStateProps) {
    const hasStructuredActions = Boolean(primary || secondary);
    const renderedIcon = typeof icon === 'string' ? (
        <span aria-hidden="true">{icon}</span>
    ) : (
        icon ?? null
    );

    return (
        <div className="empty-state-card" role="status">
            {renderedIcon && (
                <div className="empty-state-card-icon">{renderedIcon}</div>
            )}
            <h4 className="empty-state-card-title">{title}</h4>
            <p className="empty-state-card-description">{description}</p>
            {(hasStructuredActions || action) && (
                <div className="empty-state-card-actions">
                    {primary && <ActionButton descriptor={primary} variant="primary" />}
                    {secondary && <ActionButton descriptor={secondary} variant="secondary" />}
                    {action}
                </div>
            )}
            {docsHref && (
                <a
                    href={docsHref}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="empty-state-card-docs"
                >
                    Learn more →
                </a>
            )}
        </div>
    );
}
