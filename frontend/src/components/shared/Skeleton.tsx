/**
 * Skeleton — a loading placeholder (Epic U phase U7).
 *
 * Variants are CSS classes in ``index.css`` (``.skeleton--<variant>``),
 * so a loading state never writes an inline size:
 *   line        one line of body text
 *   line-short  a 60 % line (the last line of a paragraph)
 *   title       a heading
 *   text        a small meta line
 *   row         a table / list row
 *   block       a chart or card body
 *   card        a whole card
 *   inline      a few words inside running text
 */

export type SkeletonVariant = 'line' | 'line-short' | 'title' | 'text' | 'row' | 'block' | 'card' | 'inline';

interface SkeletonProps {
    variant?: SkeletonVariant;
    className?: string;
    animated?: boolean;
}

export default function Skeleton({ variant = 'line', className = '', animated = true }: SkeletonProps) {
    const classes = ['skeleton', `skeleton--${variant}`, animated ? '' : 'skeleton--static', className]
        .filter(Boolean)
        .join(' ');
    return <span className={classes} aria-hidden="true" />;
}
