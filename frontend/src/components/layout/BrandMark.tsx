/**
 * BrandMark — BrewSLM logo glyph.
 *
 * Two steam wisps rising over a flat cup rim. The cup itself is
 * implied by negative space below the rim. Single-stroke,
 * ``currentColor`` so the mark inherits the surrounding text color.
 *
 * Renders at any size; designed for 22–32px in chrome and 60–96px+
 * in marketing surfaces.
 */

interface BrandMarkProps {
    size?: number;
    title?: string;
    className?: string;
}

export default function BrandMark({
    size = 22,
    title = 'BrewSLM',
    className,
}: BrandMarkProps) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={className}
            role="img"
            aria-label={title}
        >
            <title>{title}</title>
            {/* Steam wisp — left */}
            <path d="M 9 3.5 Q 7.5 6 9 8.5 Q 10.5 11 9 13.5" />
            {/* Steam wisp — right */}
            <path d="M 15 3.5 Q 13.5 6 15 8.5 Q 16.5 11 15 13.5" />
            {/* Cup rim — flat horizontal */}
            <path d="M 4 16 L 20 16" strokeWidth="1.8" />
        </svg>
    );
}
