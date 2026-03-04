import React from 'react';

interface SkeletonProps {
    width?: string | number;
    height?: string | number;
    borderRadius?: string | number;
    className?: string;
    style?: React.CSSProperties;
    animated?: boolean;
}

export default function Skeleton({
    width = '100%',
    height = '1rem',
    borderRadius = 'var(--radius-md)',
    className = '',
    style = {},
    animated = true
}: SkeletonProps) {
    const defaultStyles: React.CSSProperties = {
        width,
        height,
        borderRadius,
        backgroundColor: 'var(--bg-tertiary)',
        display: 'inline-block',
        ...style
    };

    return (
        <span
            className={`skeleton ${animated ? 'skeleton-pulse' : ''} ${className}`}
            style={defaultStyles}
            aria-hidden="true"
        />
    );
}

// Add these keyframes to your global css file (e.g., index.css)
// @keyframes skeleton-pulse {
//     0% { opacity: 0.5; }
//     50% { opacity: 1; }
//     100% { opacity: 0.5; }
// }
// .skeleton-pulse { animation: skeleton-pulse 1.5s ease-in-out infinite; }
