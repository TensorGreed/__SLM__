/**
 * The project name "+ New Project" uses when the name field is empty
 * (UI simplification phase 6): the brief's first sentence, cut to a few
 * words. Pure, no React. The create dialog shows it and lets the user edit
 * it; the create payload sends it exactly like a typed name.
 */

const MAX_WORDS = 6;
const MAX_CHARS = 60;

/** "Answer customer support questions about accounts, billing…" → "Answer customer support questions about accounts". */
export function nameFromBrief(brief: string): string {
    const firstSentence = brief
        .split(/[.!?\n]/)
        .map((part) => part.trim())
        .find(Boolean) ?? '';
    const words = firstSentence.split(/\s+/).filter(Boolean).slice(0, MAX_WORDS);
    let name = words.join(' ');
    if (name.length > MAX_CHARS) {
        name = name.slice(0, MAX_CHARS);
        const lastSpace = name.lastIndexOf(' ');
        if (lastSpace > 0) name = name.slice(0, lastSpace);
    }
    name = name.replace(/[\s,;:()[\]{}"'`-]+$/u, '').replace(/^[\s,;:()[\]{}"'`-]+/u, '');
    return name ? name.charAt(0).toUpperCase() + name.slice(1) : '';
}

/**
 * ``nameFromBrief`` made unique against the names already listed (project
 * names are unique server side): "X", then "X 2", "X 3"… Case-insensitive.
 */
export function uniqueNameFromBrief(brief: string, taken: Iterable<string>): string {
    const base = nameFromBrief(brief);
    if (!base) return '';
    const used = new Set(Array.from(taken, (name) => name.trim().toLowerCase()));
    if (!used.has(base.toLowerCase())) return base;
    for (let n = 2; n < 1000; n += 1) {
        const candidate = `${base} ${n}`;
        if (!used.has(candidate.toLowerCase())) return candidate;
    }
    return base;
}
