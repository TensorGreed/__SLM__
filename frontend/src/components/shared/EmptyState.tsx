/**
 * EmptyState — the one way a surface says "nothing here yet" (Epic U
 * phase U7).
 *
 * A centred dashed card with an icon (a lucide icon node; the default is
 * an inbox), a title that says what the surface is for, a 1–2 line
 * description, up to two actions and an optional docs link.
 *
 *   <EmptyState
 *       icon={<Database />} title="No data yet" description="…"
 *       primary={{ label: 'Import', onClick }}
 *       secondary={{ label: 'Upload', href: '/x' }}
 *       docsHref="https://…"
 *   />
 *
 * ``action`` is a free-form slot for callers whose buttons carry state.
 */

import type { ReactNode } from 'react';
import { Inbox } from 'lucide-react';

import './EmptyState.css';

interface ActionDescriptor {
    label: string;
    onClick?: () => void;
    href?: string;
}

interface EmptyStateProps {
    /** A lucide icon element; the inbox when omitted. ``null`` hides the icon. */
    icon?: ReactNode | null;
    title: string;
    description: string;
    primary?: ActionDescriptor;
    secondary?: ActionDescriptor;
    /** Renders a small "Learn more" link below the buttons. */
    docsHref?: string;
    /** Free-form action slot; prefer primary / secondary. */
    action?: ReactNode;
    /** ``compact`` = less padding, for empty lists inside a card. */
    size?: 'default' | 'compact';
    testId?: string;
}

function ActionButton({ descriptor, variant }: { descriptor: ActionDescriptor; variant: 'primary' | 'secondary' }) {
    const className = `btn ${variant === 'primary' ? 'btn-primary' : 'btn-secondary'} btn-sm`;
    if (descriptor.href) {
        const external = descriptor.href.startsWith('http');
        return (
            <a
                href={descriptor.href}
                target={external ? '_blank' : undefined}
                rel={external ? 'noopener noreferrer' : undefined}
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
    size = 'default',
    testId,
}: EmptyStateProps) {
    const renderedIcon = icon === null ? null : (icon ?? <Inbox aria-hidden="true" />);
    const hasActions = Boolean(primary || secondary || action);

    return (
        <div
            className={`empty-state-card${size === 'compact' ? ' empty-state-card--compact' : ''}`}
            role="status"
            data-testid={testId ?? 'empty-state'}
        >
            {renderedIcon && <div className="empty-state-card-icon">{renderedIcon}</div>}
            <h4 className="empty-state-card-title">{title}</h4>
            <p className="empty-state-card-description">{description}</p>
            {hasActions && (
                <div className="empty-state-card-actions">
                    {primary && <ActionButton descriptor={primary} variant="primary" />}
                    {secondary && <ActionButton descriptor={secondary} variant="secondary" />}
                    {action}
                </div>
            )}
            {docsHref && (
                <a href={docsHref} target="_blank" rel="noopener noreferrer" className="empty-state-card-docs">
                    Learn more
                </a>
            )}
        </div>
    );
}
