/**
 * The project name "+ New Project" uses when the name field is empty
 * (UI simplification phase 6): the brief's first sentence when it is short,
 * else its first clause, else a few words (phase 7b: never cut mid-phrase,
 * "Turn a user request into exactly" read as a typo). Pure, no React. The create dialog shows it and lets the user edit
 * it; the create payload sends it exactly like a typed name.
 */

const MAX_WORDS = 6;
const MAX_CHARS = 60;
/** A first sentence up to this many words (and ``MAX_CHARS``) is the name as is. */
const WHOLE_SENTENCE_WORDS = 10;
/** A first clause needs at least this many words to stand as a name. */
const MIN_CLAUSE_WORDS = 3;
/** Words a cut name never ends on ("… into exactly" keeps "exactly"; "… questions about" drops "about"). */
const TRAILING_FUNCTION_WORDS = new Set([
    'a', 'an', 'the', 'of', 'to', 'into', 'for', 'and', 'or', 'with', 'about', 'in', 'on', 'as', 'by', 'from',
    'at', 'that', 'which', 'its', 'their', 'your', 'our', 'is', 'are', 'be',
]);

function tidy(name: string): string {
    return name.replace(/[\s,;:.!?\u3002()[\]{}"'`-]+$/u, '').replace(/^[\s,;:()[\]{}"'`-]+/u, '');
}

function wordsOf(text: string): string[] {
    return text.split(/\s+/).filter(Boolean);
}

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

/**
 * "Turn a user request into exactly one JSON tool call." → the whole sentence (short enough);
 * "Answer customer support questions about accounts, billing and teams from the help center." → its first clause
 * "Answer customer support questions about accounts"; else six words, never ending on "a" / "into" / "about"….
 */
export function nameFromBrief(brief: string): string {
    const firstSentence = tidy(firstSentenceOf(brief));
    const sentenceWords = wordsOf(firstSentence);
    let name: string;
    if (sentenceWords.length <= WHOLE_SENTENCE_WORDS && firstSentence.length <= MAX_CHARS) {
        name = sentenceWords.join(' ');
    } else {
        // The first clause: up to a comma, semicolon, colon or a spaced dash.
        const clause = tidy(firstSentence.split(/[,;:]\s|\s[-\u2013\u2014]\s/u)[0] ?? '');
        const clauseWords = wordsOf(clause);
        if (clauseWords.length >= MIN_CLAUSE_WORDS && clauseWords.length <= WHOLE_SENTENCE_WORDS && clause.length <= MAX_CHARS) {
            name = clauseWords.join(' ');
        } else {
            const words = sentenceWords.slice(0, MAX_WORDS);
            while (words.length > 2 && TRAILING_FUNCTION_WORDS.has(tidy(words[words.length - 1]).toLowerCase())) words.pop();
            name = words.join(' ');
            if (name.length > MAX_CHARS) {
                name = name.slice(0, MAX_CHARS);
                const lastSpace = name.lastIndexOf(' ');
                if (lastSpace > 0) name = name.slice(0, lastSpace);
            }
        }
    }
    name = tidy(name);
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
