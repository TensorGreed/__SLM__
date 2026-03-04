import React from 'react';

interface EmptyStateProps {
    icon?: string;
    title: string;
    description: string;
    action?: React.ReactNode;
}

export default function EmptyState({ icon = '📂', title, description, action }: EmptyStateProps) {
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 'var(--space-2xl) var(--space-xl)',
            textAlign: 'center',
            background: 'var(--bg-tertiary)',
            borderRadius: 'var(--radius-lg)',
            border: '1px dashed var(--border-color)',
            margin: 'var(--space-md) 0'
        }}>
            <div style={{
                fontSize: '3rem',
                marginBottom: 'var(--space-md)',
                opacity: 0.8
            }}>
                {icon}
            </div>
            <h4 style={{
                fontSize: 'var(--font-size-lg)',
                fontWeight: 600,
                color: 'var(--text-primary)',
                marginBottom: 'var(--space-sm)'
            }}>
                {title}
            </h4>
            <p style={{
                fontSize: 'var(--font-size-md)',
                color: 'var(--text-secondary)',
                maxWidth: '400px',
                marginBottom: action ? 'var(--space-lg)' : 0,
                lineHeight: 1.5
            }}>
                {description}
            </p>
            {action && (
                <div>{action}</div>
            )}
        </div>
    );
}
