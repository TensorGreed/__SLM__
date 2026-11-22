/**
 * The project name "+ New Project" uses when the name field is empty
 * (UI simplification phase 6): the brief's first sentence, cut to a few
 * words. Pure, no React. The create dialog shows it and lets the user edit
 * it; the create payload sends it exactly like a typed name.
 */

const MAX_WORDS = 6;
const MAX_CHARS = 60;

/** Initialisms ("U.S.", "e.g.") and a few short abbreviations a "." does not end a sentence after. */
const ABBREVIATION = /(?:^|\s)(?:(?:\p{L}\.)+|vs\.|etc\.|approx\.|incl\.)$/iu;

/**
 * The text up to the first sentence end: a newline, "\u3002", or ".", "!" or "?"
 * followed by whitespace or the end of the text. So "Node.js" and "v2.5" stay
 * whole, and a "." after an initialism ("U.S. tax law") is not an end either.
 */
export function firstSentenceOf(text: string): string {
    const parts: string[] = [];
    let start = 0;
    for (let i = 0; i < text.length; i += 1) {
        const ch = text[i];
        if (ch === '\n') {
            parts.push(text.slice(start, i));
            start = i + 1;
            continue;
        }
        if (!'.!?\u3002'.includes(ch)) continue;
        const next = text[i + 1];
        // "\u3002" (the ideographic full stop) needs no space after it.
        if (ch !== '\u3002' && next !== undefined && !/\s/u.test(next)) continue;
        if (ch === '.' && ABBREVIATION.test(text.slice(start, i + 1))) continue;
        parts.push(text.slice(start, i));
        start = i + 1;
    }
    parts.push(text.slice(start));
    return parts.map((part) => part.trim()).find(Boolean) ?? '';
}

/** "Answer customer support questions about accounts, billing…" → "Answer customer support questions about accounts". */
export function nameFromBrief(brief: string): string {
    const firstSentence = firstSentenceOf(brief);
    const words = firstSentence.split(/\s+/).filter(Boolean).slice(0, MAX_WORDS);
    let name = words.join(' ');
    if (name.length > MAX_CHARS) {
        name = name.slice(0, MAX_CHARS);
        const lastSpace = name.lastIndexOf(' ');
        if (lastSpace > 0) name = name.slice(0, lastSpace);
    }
    name = name.replace(/[\s,;:.!?\u3002()[\]{}"'`-]+$/u, '').replace(/^[\s,;:()[\]{}"'`-]+/u, '');
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

/** How many times the create dialog retries a derived name the server says is taken. */
export const MAX_DERIVED_NAME_RETRIES = 5;

/**
 * True for the create endpoint's duplicate-name 400 (``error_code:
 * PROJECT_NAME_TAKEN`` in the structured-error envelope).
 */
export function isProjectNameTakenError(error: unknown): boolean {
    const response = (error as { response?: { status?: number; data?: { error_code?: unknown } } })?.response;
    return response?.status === 400 && response?.data?.error_code === 'PROJECT_NAME_TAKEN';
}
