import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
// Stub the LLM-generate panel — its own tests cover its behavior, and
// it does its own fetches on mount that we don't want bleeding into
// these focused entries-list tests.
vi.mock('./LlmGoldGeneratePanel', () => ({
    default: () => <div data-testid="llm-gold-generate-stub" />,
}));
vi.mock('../coach/CoachStrip', () => ({
    default: () => null,
}));

import GoldSetPanel from './GoldSetPanel';


function makeQaEntry(
    overrides: Partial<{
        question: string;
        answer: string;
        difficulty: string;
        is_hallucination_trap: boolean;
    }> = {},
) {
    return {
        question: 'Q?',
        answer: 'A.',
        difficulty: 'medium',
        is_hallucination_trap: false,
        ...overrides,
    };
}


/**
 * Route-aware GET mock — the panel fires:
 *   * /projects/{id} (one-shot, for recipe lookup)
 *   * /projects/{id}/gold/entries?dataset_type=… (re-fetched on
 *     dataset switch)
 */
function installGetRouter(opts: {
    recipeId?: string | null;
    entries?: unknown[];
} = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/gold/entries')) {
            return { data: { entries: opts.entries ?? [] } };
        }
        if (/\/projects\/\d+(\?|$)/.test(url)) {
            return {
                data: {
                    selected_recipe: opts.recipeId
                        ? { recipe_id: opts.recipeId }
                        : undefined,
                },
            };
        }
        return { data: {} };
    });
}


describe('GoldSetPanel — entries filter + mix summary', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders mix summary for qa-sft with the correct difficulty + trap counts', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            entries: [
                makeQaEntry({ difficulty: 'easy' }),
                makeQaEntry({ difficulty: 'easy' }),
                makeQaEntry({ difficulty: 'medium' }),
                makeQaEntry({ difficulty: 'hard' }),
                // Trap rows ALSO count in their difficulty bucket
                // (a row is "hard + trap", not "trap instead of hard").
                makeQaEntry({ difficulty: 'hard', is_hallucination_trap: true }),
                makeQaEntry({ difficulty: 'medium', is_hallucination_trap: true }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);

        const summary = await screen.findByTestId('gold-entries-mix-summary');
        expect(summary.textContent).toMatch(/6 entries/);
        expect(
            screen.getByTestId('gold-entries-mix-easy').textContent,
        ).toBe('2 easy');
        expect(
            screen.getByTestId('gold-entries-mix-medium').textContent,
        ).toBe('2 medium');
        expect(
            screen.getByTestId('gold-entries-mix-hard').textContent,
        ).toBe('2 hard');
        expect(
            screen.getByTestId('gold-entries-mix-traps').textContent,
        ).toBe('2 hallucination traps');
    });

    it('normalizes missing/unknown difficulty to "medium" in the mix summary', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            entries: [
                // Pre-tagging-era entries: no ``difficulty`` field at all.
                { question: 'Q1', answer: 'A1' },
                // ``difficulty: ""`` — also treat as medium.
                makeQaEntry({ difficulty: '' }),
                // Unknown synonym — treat as medium.
                makeQaEntry({ difficulty: 'expert' }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entries-mix-summary');
        expect(
            screen.getByTestId('gold-entries-mix-medium').textContent,
        ).toBe('3 medium');
        expect(
            screen.getByTestId('gold-entries-mix-easy').textContent,
        ).toBe('0 easy');
        expect(
            screen.getByTestId('gold-entries-mix-hard').textContent,
        ).toBe('0 hard');
    });

    it('filter dropdown narrows the entries list + shows the active-filter banner', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            entries: [
                makeQaEntry({ question: 'easy-1', difficulty: 'easy' }),
                makeQaEntry({ question: 'med-1', difficulty: 'medium' }),
                makeQaEntry({ question: 'hard-1', difficulty: 'hard' }),
                makeQaEntry({ question: 'hard-2', difficulty: 'hard' }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entries-filter');

        // Default: all 4 visible, no banner.
        expect(screen.getByText(/easy-1/)).toBeInTheDocument();
        expect(screen.getByText(/hard-1/)).toBeInTheDocument();
        expect(
            screen.queryByTestId('gold-entries-filter-banner'),
        ).not.toBeInTheDocument();

        // Filter to hard only.
        await userEvent.selectOptions(
            screen.getByTestId('gold-entries-filter'),
            'hard',
        );
        await waitFor(() => {
            expect(screen.queryByText(/easy-1/)).not.toBeInTheDocument();
        });
        expect(screen.queryByText(/med-1/)).not.toBeInTheDocument();
        expect(screen.getByText(/hard-1/)).toBeInTheDocument();
        expect(screen.getByText(/hard-2/)).toBeInTheDocument();

        // Active-filter banner surfaces "Showing N of M".
        const banner = screen.getByTestId('gold-entries-filter-banner');
        expect(banner.textContent).toMatch(/Showing\s*2\s*of\s*4/);

        // Clear filter restores all entries + hides the banner.
        await userEvent.click(screen.getByTestId('gold-entries-filter-clear'));
        await waitFor(() => {
            expect(
                screen.queryByTestId('gold-entries-filter-banner'),
            ).not.toBeInTheDocument();
        });
        expect(screen.getByText(/easy-1/)).toBeInTheDocument();
    });

    it('"Hallucination traps only" filter shows only trap rows', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            entries: [
                makeQaEntry({ question: 'plain', difficulty: 'easy' }),
                makeQaEntry({
                    question: 'trap-A',
                    difficulty: 'hard',
                    is_hallucination_trap: true,
                }),
                makeQaEntry({
                    question: 'trap-B',
                    difficulty: 'medium',
                    is_hallucination_trap: true,
                }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entries-filter');

        await userEvent.selectOptions(
            screen.getByTestId('gold-entries-filter'),
            'traps',
        );
        await waitFor(() => {
            expect(screen.queryByText(/plain/)).not.toBeInTheDocument();
        });
        expect(screen.getByText(/trap-A/)).toBeInTheDocument();
        expect(screen.getByText(/trap-B/)).toBeInTheDocument();
        expect(
            screen.getByTestId('gold-entries-filter-banner').textContent,
        ).toMatch(/Showing\s*2\s*of\s*3/);
    });

    it('filter narrowed to zero matches surfaces the filtered-empty hint', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            entries: [
                makeQaEntry({ difficulty: 'easy' }),
                makeQaEntry({ difficulty: 'medium' }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entries-filter');

        await userEvent.selectOptions(
            screen.getByTestId('gold-entries-filter'),
            'hard',
        );
        const emptyHint = await screen.findByTestId(
            'gold-entries-filtered-empty',
        );
        expect(emptyHint.textContent).toMatch(/No entries match/);
    });

    it('non-qa-sft recipe hides BOTH the filter dropdown and the mix summary', async () => {
        installGetRouter({
            recipeId: 'classification',
            // Classification rows do still carry difficulty=medium /
            // trap=false defaults from the import path, but those
            // fields are meaningless for the recipe so the UX is
            // hidden entirely.
            entries: [
                {
                    text: 'good experience',
                    label: 'positive',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
                {
                    text: 'terrible app',
                    label: 'negative',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        // The LLM-generate stub renders → indicates the recipe lookup
        // completed. After that, the qa-sft-only controls should be
        // absent.
        await screen.findByTestId('llm-gold-generate-stub');
        expect(
            screen.queryByTestId('gold-entries-mix-summary'),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByTestId('gold-entries-filter'),
        ).not.toBeInTheDocument();
        // Entries badge (count) is still there.
        expect(screen.getAllByText('2').length).toBeGreaterThan(0);
    });

    // ── Per-recipe entries-list rendering (via shared body) ─────────

    it('qa-sft entries render Q/A + difficulty/trap badges via the shared body', async () => {
        installGetRouter({
            recipeId: 'qa-sft',
            entries: [
                makeQaEntry({
                    question: 'How do refunds work?',
                    answer: 'Visit Settings → Billing.',
                    difficulty: 'easy',
                    is_hallucination_trap: false,
                }),
                makeQaEntry({
                    question: 'What is the meaning of life?',
                    answer: "I don't know.",
                    difficulty: 'hard',
                    is_hallucination_trap: true,
                }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        // Row 0: difficulty badge present, trap absent, Q/A visible.
        expect(
            screen.getByTestId('gold-entry-row-0-difficulty').textContent,
        ).toBe('easy');
        expect(
            screen.queryByTestId('gold-entry-row-0-trap'),
        ).not.toBeInTheDocument();
        expect(screen.getByText(/How do refunds work\?/)).toBeInTheDocument();
        expect(screen.getByText(/Visit Settings/)).toBeInTheDocument();

        // Row 1: difficulty badge + trap badge both present.
        expect(
            screen.getByTestId('gold-entry-row-1-difficulty').textContent,
        ).toBe('hard');
        expect(
            screen.getByTestId('gold-entry-row-1-trap').textContent,
        ).toMatch(/trap/);
    });

    it('classification entries render text + label badge (not Q/A)', async () => {
        installGetRouter({
            recipeId: 'classification',
            entries: [
                {
                    text: 'I love this product!',
                    label: 'positive',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
                {
                    text: 'Returned it the next day.',
                    label: 'negative',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        // text + label render via the recipe-specific path.
        expect(
            screen.getByTestId('gold-entry-row-0-text').textContent,
        ).toBe('I love this product!');
        expect(
            screen.getByTestId('gold-entry-row-0-label').textContent,
        ).toContain('positive');
        expect(
            screen.getByTestId('gold-entry-row-1-label').textContent,
        ).toContain('negative');

        // qa-sft framing is absent — no Q:/A: prefixes, no
        // difficulty badge surfaced for non-qa rows.
        expect(screen.queryByText(/^Q:/)).not.toBeInTheDocument();
        expect(
            screen.queryByTestId('gold-entry-row-0-difficulty'),
        ).not.toBeInTheDocument();
    });

    it('span-extraction entries render text + entity list with offsets', async () => {
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    text: 'Contact jane@example.com today',
                    entities: [
                        {
                            type: 'email',
                            start: 8,
                            end: 24,
                            text: 'jane@example.com',
                        },
                    ],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
                {
                    text: 'no PII in this row',
                    entities: [],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        // Row 0: text + entity badge + offsets visible.
        const entities0 = screen.getByTestId('gold-entry-row-0-entities');
        expect(entities0.textContent).toContain('email');
        expect(entities0.textContent).toContain('jane@example.com');
        expect(entities0.textContent).toContain('[8:24]');

        // Row 1: empty entities surfaces the negative-example hint.
        const entities1 = screen.getByTestId('gold-entry-row-1-entities');
        expect(entities1.textContent).toMatch(/negative example/i);
    });

    it('summarization entries render document (collapsed) + summary', async () => {
        const longDoc = 'This is a long meeting transcript. '.repeat(6);
        installGetRouter({
            recipeId: 'summarization',
            entries: [
                {
                    document: longDoc,
                    summary: 'Meeting covered hiring, budget, and OKRs.',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        const docBlock = screen.getByTestId('gold-entry-row-0-document');
        expect(docBlock.tagName.toLowerCase()).toBe('details');
        expect(docBlock.textContent).toMatch(/Document \(\d+ chars\)/);
        expect(
            screen.getByTestId('gold-entry-row-0-summary').textContent,
        ).toContain('Meeting covered hiring, budget, and OKRs.');
    });

    it('legacy entries without a project recipe fall back to qa-sft rendering', async () => {
        // Old projects that never had ``selected_recipe`` set — the
        // entries still have ``question``/``answer`` keys, so we
        // render them with the qa-sft body (the default fallback in
        // the panel's recipe narrowing).
        installGetRouter({
            recipeId: null,
            entries: [
                makeQaEntry({ question: 'Legacy Q?', answer: 'Legacy A.' }),
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');
        expect(screen.getByText(/Legacy Q\?/)).toBeInTheDocument();
        expect(screen.getByText(/Legacy A\./)).toBeInTheDocument();
        // qa-sft fallback also surfaces the (normalized) difficulty badge.
        expect(
            screen.getByTestId('gold-entry-row-0-difficulty').textContent,
        ).toBe('medium');
    });

    it('empty gold set hides the filter dropdown (nothing to filter)', async () => {
        installGetRouter({ recipeId: 'qa-sft', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        // Wait for the LLM-generate stub so we know the recipe lookup
        // completed and the entries fetch landed.
        await screen.findByTestId('llm-gold-generate-stub');
        expect(
            screen.queryByTestId('gold-entries-filter'),
        ).not.toBeInTheDocument();
        // Mix summary also hidden when total is 0.
        expect(
            screen.queryByTestId('gold-entries-mix-summary'),
        ).not.toBeInTheDocument();
    });

    // ── Per-recipe inline add form ────────────────────────────────

    it('no recipe set: add form is hidden + a recipe-hint banner shows', async () => {
        installGetRouter({ recipeId: null, entries: [] });
        render(<GoldSetPanel projectId={1} />);
        // Banner present, telling the user where to pick a recipe.
        // The recipe-lookup GET resolves async, so wait on the
        // hint itself (the LLM-generate stub is gated off too in
        // this case, can't wait on it).
        const hint = await screen.findByTestId('gold-add-form-hidden-hint');
        expect(hint.textContent).toMatch(/Pick a recipe/);
        // Form gone.
        expect(screen.queryByTestId('gold-add-form')).not.toBeInTheDocument();
    });

    it('qa-sft: form submits a Q+A pair with difficulty + trap flag', async () => {
        installGetRouter({ recipeId: 'qa-sft', entries: [] });
        apiMock.post.mockResolvedValue({ data: { id: 1 } });
        render(<GoldSetPanel projectId={42} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-question'), {
            target: { value: 'What is the refund policy?' },
        });
        fireEvent.change(screen.getByTestId('gold-add-answer'), {
            target: { value: '30 days, no questions asked.' },
        });
        await userEvent.selectOptions(
            screen.getByTestId('gold-add-difficulty'),
            'hard',
        );
        await userEvent.click(screen.getByTestId('gold-add-trap'));
        await userEvent.click(screen.getByTestId('gold-add-submit'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/gold/add',
                expect.objectContaining({
                    question: 'What is the refund policy?',
                    answer: '30 days, no questions asked.',
                    difficulty: 'hard',
                    is_hallucination_trap: true,
                    dataset_type: 'gold_dev',
                }),
            );
        });
    });

    it('classification: form submits text + label, no Q/A keys', async () => {
        installGetRouter({
            recipeId: 'classification',
            entries: [
                // Seed an existing label so the combobox surfaces it.
                {
                    text: 'seed',
                    label: 'billing',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        apiMock.post.mockResolvedValue({ data: { id: 2 } });
        render(<GoldSetPanel projectId={42} />);
        await screen.findByTestId('gold-add-form');

        // Combobox hint surfaces known labels.
        expect(
            screen.getByTestId('gold-add-label-hint').textContent,
        ).toMatch(/billing/);

        // No qa-sft inputs in the form when recipe is classification.
        expect(screen.queryByTestId('gold-add-question')).not.toBeInTheDocument();
        expect(screen.queryByTestId('gold-add-answer')).not.toBeInTheDocument();

        fireEvent.change(screen.getByTestId('gold-add-text'), {
            target: { value: "Where's my refund?" },
        });
        fireEvent.change(screen.getByTestId('gold-add-label'), {
            target: { value: 'billing' },
        });
        await userEvent.click(screen.getByTestId('gold-add-submit'));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/gold/add',
                expect.objectContaining({
                    text: "Where's my refund?",
                    label: 'billing',
                    dataset_type: 'gold_dev',
                }),
            );
        });
        // The shape sent should NOT include qa-sft keys.
        const lastCall = apiMock.post.mock.calls.at(-1) as unknown[];
        const body = lastCall[1] as Record<string, unknown>;
        expect(body.question).toBeUndefined();
        expect(body.answer).toBeUndefined();
    });

    it('span-extraction: invalid JSON disables submit + shows error', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-span-text'), {
            target: { value: 'Contact jane@example.com today' },
        });
        // Mis-typed JSON (trailing comma).
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: '[{"type":"email","start":8,"end":24,},]' },
        });
        // Error visible + submit disabled.
        const err = await screen.findByTestId('gold-add-entities-error');
        expect(err.textContent).toMatch(/Invalid JSON/);
        expect(
            (screen.getByTestId('gold-add-submit') as HTMLButtonElement).disabled,
        ).toBe(true);
        // No POST fired.
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('span-extraction: offset mismatch disables submit + names the bad span', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-span-text'), {
            target: { value: 'abcdefghij' },
        });
        // Offsets 0:3 = "abc" but the user claimed "xyz" — offset
        // mismatch must be caught BEFORE submit.
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: '[{"type":"tag","start":0,"end":3,"text":"xyz"}]' },
        });
        const err = await screen.findByTestId('gold-add-entities-error');
        expect(err.textContent).toMatch(/offset mismatch/);
        expect(
            (screen.getByTestId('gold-add-submit') as HTMLButtonElement).disabled,
        ).toBe(true);
    });

    it('span-extraction: valid JSON shows the verified-entities hint', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        apiMock.post.mockResolvedValue({ data: { id: 3 } });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-span-text'), {
            target: { value: 'Contact jane@example.com today' },
        });
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: {
                value: '[{"type":"email","start":8,"end":24,"text":"jane@example.com"}]',
            },
        });
        expect(
            screen.getByTestId('gold-add-entities-valid').textContent,
        ).toMatch(/1 entity parsed/);

        await userEvent.click(screen.getByTestId('gold-add-submit'));
        await waitFor(() => {
            const body = (apiMock.post.mock.calls.at(-1) as unknown[])[1] as Record<
                string,
                unknown
            >;
            expect(body.entities).toEqual([
                {
                    type: 'email',
                    start: 8,
                    end: 24,
                    text: 'jane@example.com',
                },
            ]);
            expect(body.text).toBe('Contact jane@example.com today');
        });
    });

    it('span-extraction: empty entities JSON is treated as a valid negative example', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        apiMock.post.mockResolvedValue({ data: { id: 4 } });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-span-text'), {
            target: { value: 'clean text, no PII' },
        });
        // Leave entities JSON blank — that's a negative example.
        // No error, no validation hint, submit ENABLED.
        expect(
            screen.queryByTestId('gold-add-entities-error'),
        ).not.toBeInTheDocument();
        expect(
            (screen.getByTestId('gold-add-submit') as HTMLButtonElement).disabled,
        ).toBe(false);
        await userEvent.click(screen.getByTestId('gold-add-submit'));
        await waitFor(() => {
            const body = (apiMock.post.mock.calls.at(-1) as unknown[])[1] as Record<
                string,
                unknown
            >;
            expect(body.entities).toEqual([]);
        });
    });

    it('summarization: form submits document + summary', async () => {
        installGetRouter({ recipeId: 'summarization', entries: [] });
        apiMock.post.mockResolvedValue({ data: { id: 5 } });
        render(<GoldSetPanel projectId={42} />);
        await screen.findByTestId('gold-add-form');

        const longDoc = 'The board met on March 14 to review three topics. '.repeat(3);
        fireEvent.change(screen.getByTestId('gold-add-document'), {
            target: { value: longDoc },
        });
        fireEvent.change(screen.getByTestId('gold-add-summary'), {
            target: { value: 'Board reviewed three topics on March 14.' },
        });
        await userEvent.click(screen.getByTestId('gold-add-submit'));

        await waitFor(() => {
            // The form trims the document on submit — assert against
            // the trimmed value, not the raw repeat-with-trailing-space.
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/gold/add',
                expect.objectContaining({
                    document: longDoc.trim(),
                    summary: 'Board reviewed three topics on March 14.',
                    dataset_type: 'gold_dev',
                }),
            );
        });
    });

    it('submit is disabled until required fields are filled (qa-sft)', async () => {
        installGetRouter({ recipeId: 'qa-sft', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');
        const submit = screen.getByTestId('gold-add-submit') as HTMLButtonElement;
        expect(submit.disabled).toBe(true);

        fireEvent.change(screen.getByTestId('gold-add-question'), {
            target: { value: 'Q?' },
        });
        // Still disabled — answer empty.
        expect(submit.disabled).toBe(true);

        fireEvent.change(screen.getByTestId('gold-add-answer'), {
            target: { value: 'A.' },
        });
        expect(submit.disabled).toBe(false);
    });

    // ── Legacy template-seeded rows (Q+A on disk) ──────────────────
    //
    // Templates (ticket-router, contract-clause-extractor, security-
    // alert-summarizer) pre-date the per-recipe panel. The shared
    // materialization path flattens every recipe shape into the
    // legacy ``{question, answer}`` Q+A keys on disk:
    //   * classification   → answer = "<label>"
    //   * span-extraction  → answer = JSON.stringify({"entities": [...]})
    //   * summarization    → answer = JSON.stringify({"summary": "..."})
    // The panel normalizes these on read so rows don't render as
    // empty divs in non-qa-sft projects.

    it('legacy classification: template-seeded {question, answer:"label"} renders correctly', async () => {
        installGetRouter({
            recipeId: 'classification',
            entries: [
                {
                    id: 1,
                    question: 'Charged $49 yesterday but I cancelled.',
                    answer: 'billing',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
                {
                    id: 2,
                    question: 'App crashes on Android.',
                    answer: 'technical',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        // Text mapped from ``question``, label mapped from ``answer``.
        expect(
            screen.getByTestId('gold-entry-row-0-text').textContent,
        ).toBe('Charged $49 yesterday but I cancelled.');
        expect(
            screen.getByTestId('gold-entry-row-0-label').textContent,
        ).toContain('billing');
        expect(
            screen.getByTestId('gold-entry-row-1-label').textContent,
        ).toContain('technical');
    });

    it('legacy span-extraction: JSON-encoded entities in answer get parsed + rendered', async () => {
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    question: "Provider's total liability shall not exceed fees.",
                    // ↓ Exactly what _materialize_demo_bundle_into_project
                    // writes for the contract-clause-extractor template.
                    answer: JSON.stringify({
                        entities: [
                            {
                                type: 'liability_cap',
                                start: 0,
                                end: 48,
                                text: "Provider's total liability shall not exceed fees",
                            },
                        ],
                    }),
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        // Text mapped from ``question``.
        expect(
            screen.getByTestId('gold-entry-row-0-text').textContent,
        ).toContain("Provider's total liability");
        // Entities parsed out of the JSON-encoded answer.
        const ents = screen.getByTestId('gold-entry-row-0-entities');
        expect(ents.textContent).toContain('liability_cap');
        expect(ents.textContent).toContain('[0:48]');
    });

    it('legacy span-extraction with no entities (negative example) shows the hint', async () => {
        // Some templates / hand-curated rows might not have an
        // entities key at all and ``answer`` is something else.
        // The normalizer defaults to empty array, which renders as
        // "No entities (negative example)".
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    question: 'plain text with nothing notable',
                    answer: '',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');
        expect(
            screen.getByTestId('gold-entry-row-0-entities').textContent,
        ).toMatch(/negative example/i);
    });

    it('legacy summarization: JSON-encoded summary in answer gets parsed + rendered', async () => {
        installGetRouter({
            recipeId: 'summarization',
            entries: [
                {
                    id: 1,
                    question: 'Cisco IOS XE devices with the management web interface enabled are affected by CVE-2023-20198 (CVSS 10.0)...',
                    // ↓ Exactly what _materialize_demo_bundle_into_project
                    // writes for the security-alert-summarizer template.
                    answer: JSON.stringify({
                        summary: 'Critical unauth account-create + RCE in Cisco IOS XE management web interface (CVE-2023-20198).',
                    }),
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');

        // Document collapses behind <details> by default — its
        // textContent still includes the advisory text.
        const doc = screen.getByTestId('gold-entry-row-0-document');
        expect(doc.textContent).toContain('Cisco IOS XE');
        // Summary parsed from the JSON-encoded answer.
        expect(
            screen.getByTestId('gold-entry-row-0-summary').textContent,
        ).toContain('CVE-2023-20198');
    });

    it('legacy rows: half-migrated row (both shapes present) prefers the new shape', async () => {
        // Defensive — if a row carries BOTH the legacy {question, answer}
        // AND the recipe-shaped {text, label}, the normalizer must
        // NOT overwrite the recipe shape with the legacy one.
        installGetRouter({
            recipeId: 'classification',
            entries: [
                {
                    id: 1,
                    question: 'OLD QUESTION (should be ignored)',
                    answer: 'OLD ANSWER (should be ignored)',
                    text: 'NEW TEXT (preferred)',
                    label: 'NEW LABEL (preferred)',
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');
        expect(
            screen.getByTestId('gold-entry-row-0-text').textContent,
        ).toBe('NEW TEXT (preferred)');
        expect(
            screen.getByTestId('gold-entry-row-0-label').textContent,
        ).toContain('NEW LABEL (preferred)');
    });

    it('legacy classification with nested expected.label also parses', async () => {
        // gold_set_workbench writes rows with the expected shape
        // (``{input: {...}, expected: {label: ...}}``) directly when
        // not flattening — the normalizer reaches into ``expected``
        // as a fallback so workbench-style rows also render.
        installGetRouter({
            recipeId: 'classification',
            entries: [
                {
                    id: 1,
                    question: 'A ticket text.',
                    expected: { label: 'billing' },
                    // ``answer`` deliberately omitted — exercising the
                    // expected-dict fallback path.
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-entry-row-0');
        expect(
            screen.getByTestId('gold-entry-row-0-label').textContent,
        ).toContain('billing');
    });

    // ── Highlight-to-select helper for span-extraction ──────────────

    /** Programmatically set a textarea's selection range + fire the
     *  matching select event. jsdom doesn't trigger React's onSelect
     *  automatically when setSelectionRange runs, so we dispatch it. */
    function selectRange(
        textarea: HTMLTextAreaElement,
        start: number,
        end: number,
    ): void {
        textarea.focus();
        textarea.setSelectionRange(start, end);
        fireEvent.select(textarea);
    }

    it('span-extraction helper: starts with no selection + button disabled', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');
        const addBtn = screen.getByTestId('gold-add-span-helper-add') as HTMLButtonElement;
        expect(addBtn.disabled).toBe(true);
        // Preview shows the "highlight a range" hint.
        expect(
            screen.getByTestId('gold-add-span-helper-preview').textContent,
        ).toMatch(/Highlight a range/);
    });

    it('span-extraction helper: highlighting + type fills the preview + enables the button', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const text = 'Contact jane@example.com today';
        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: text } });
        // Indices [8, 24] cover "jane@example.com".
        selectRange(textarea, 8, 24);

        // Preview surfaces the selected text + offsets.
        const preview = screen.getByTestId('gold-add-span-helper-preview');
        expect(preview.textContent).toMatch(/jane@example\.com/);
        expect(preview.textContent).toMatch(/\[8:24\]/);

        // Button still disabled because type is empty.
        const addBtn = screen.getByTestId('gold-add-span-helper-add') as HTMLButtonElement;
        expect(addBtn.disabled).toBe(true);

        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'email' },
        });
        expect(addBtn.disabled).toBe(false);
    });

    it('span-extraction helper: click appends the span to entities JSON', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const text = 'Contact jane@example.com today';
        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: text } });
        selectRange(textarea, 8, 24);
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'email' },
        });
        await userEvent.click(screen.getByTestId('gold-add-span-helper-add'));

        // Pretty-printed JSON lands in the entities textarea.
        const entitiesArea = screen.getByTestId('gold-add-entities') as HTMLTextAreaElement;
        const parsed = JSON.parse(entitiesArea.value);
        expect(parsed).toEqual([{
            type: 'email',
            start: 8,
            end: 24,
            text: 'jane@example.com',
        }]);
        // Pretty-print uses 2-space indent.
        expect(entitiesArea.value).toContain('  "type"');

        // Type input clears after add so the user is primed for the
        // next span — selection persists in case they want to label
        // the same range with a second type.
        expect(
            (screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement).value,
        ).toBe('');
    });

    it('span-extraction helper: appending preserves existing hand-edited entries', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const text = 'Email me at user@host.com or call +1-555-0100.';
        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: text } });

        // User pre-populates the JSON editor with a hand-typed span.
        // text[12:25] is "user@host.com" — valid.
        const handTypedEntities = JSON.stringify(
            [{ type: 'email', start: 12, end: 25, text: 'user@host.com' }],
            null,
            2,
        );
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: handTypedEntities },
        });

        // Then the user highlights the phone and clicks add.
        // text[34:45] is "+1-555-0100".
        selectRange(textarea, 34, 45);
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'phone' },
        });
        await userEvent.click(screen.getByTestId('gold-add-span-helper-add'));

        const entitiesArea = screen.getByTestId('gold-add-entities') as HTMLTextAreaElement;
        const parsed = JSON.parse(entitiesArea.value);
        expect(parsed).toHaveLength(2);
        expect(parsed[0]).toEqual({
            type: 'email',
            start: 12,
            end: 25,
            text: 'user@host.com',
        });
        expect(parsed[1]).toEqual({
            type: 'phone',
            start: 34,
            end: 45,
            text: '+1-555-0100',
        });
    });

    it('span-extraction helper: editing the source text clears the captured selection', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: 'first text version' } });
        selectRange(textarea, 0, 5);
        // Preview shows a selection.
        expect(
            screen.getByTestId('gold-add-span-helper-preview').textContent,
        ).toMatch(/first/);

        // Edit the text — selection state must reset (the old offsets
        // mean a different thing now).
        fireEvent.change(textarea, { target: { value: 'totally different text now' } });
        await waitFor(() => {
            expect(
                screen.getByTestId('gold-add-span-helper-preview').textContent,
            ).toMatch(/Highlight a range/);
        });
        expect(
            (screen.getByTestId('gold-add-span-helper-add') as HTMLButtonElement).disabled,
        ).toBe(true);
    });

    it('span-extraction helper: refuses to add when the JSON editor is broken', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: 'Contact jane@example.com today' } });
        selectRange(textarea, 8, 24);
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'email' },
        });
        // User typed an invalid JSON edit BEFORE clicking the helper —
        // the helper must not blow it away.
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: '[{"type":"broken"' },  // unclosed
        });
        const addBtn = screen.getByTestId('gold-add-span-helper-add') as HTMLButtonElement;
        expect(addBtn.disabled).toBe(true);
        // Title attribute steers the user to the fix.
        expect(addBtn.title).toMatch(/Fix the JSON/);
        // The user's broken edit is still in the textarea — not stomped.
        expect(
            (screen.getByTestId('gold-add-entities') as HTMLTextAreaElement).value,
        ).toBe('[{"type":"broken"');
    });

    it('span-extraction helper: cursor-only "selection" (start == end) does not enable add', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: 'some text' } });
        // Just a cursor position, no range.
        selectRange(textarea, 4, 4);
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'tag' },
        });
        expect(
            (screen.getByTestId('gold-add-span-helper-add') as HTMLButtonElement).disabled,
        ).toBe(true);
        expect(
            screen.getByTestId('gold-add-span-helper-preview').textContent,
        ).toMatch(/Highlight a range/);
    });

    // ── Deletable chip list above the Entities JSON editor ─────────

    it('span-extraction chips: list derives from parsed JSON; one chip per entity', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const initialJson = JSON.stringify(
            [
                { type: 'email', start: 8, end: 24, text: 'jane@example.com' },
                { type: 'phone', start: 34, end: 45, text: '+1-555-0100' },
            ],
            null,
            2,
        );
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: initialJson },
        });

        // Two chips render, surfacing type + text + offsets each.
        const chip0 = await screen.findByTestId('gold-add-span-chip-0');
        expect(chip0.textContent).toContain('email');
        expect(chip0.textContent).toContain('jane@example.com');
        expect(chip0.textContent).toContain('[8:24]');
        const chip1 = screen.getByTestId('gold-add-span-chip-1');
        expect(chip1.textContent).toContain('phone');
        expect(chip1.textContent).toContain('+1-555-0100');
        expect(chip1.textContent).toContain('[34:45]');
        // No third chip.
        expect(screen.queryByTestId('gold-add-span-chip-2')).not.toBeInTheDocument();
    });

    it('span-extraction chips: clicking ✕ removes the entity + re-pretty-prints', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const initialJson = JSON.stringify(
            [
                { type: 'email', start: 8, end: 24, text: 'jane@example.com' },
                { type: 'phone', start: 34, end: 45, text: '+1-555-0100' },
            ],
            null,
            2,
        );
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: initialJson },
        });

        await userEvent.click(screen.getByTestId('gold-add-span-chip-0-remove'));

        // The remaining chip slides into index 0.
        await waitFor(() => {
            expect(
                screen.getByTestId('gold-add-span-chip-0').textContent,
            ).toContain('phone');
        });
        expect(
            screen.queryByTestId('gold-add-span-chip-1'),
        ).not.toBeInTheDocument();

        // JSON editor reflects the deletion with pretty-printed
        // (2-space) output — single entity remaining.
        const editor = screen.getByTestId('gold-add-entities') as HTMLTextAreaElement;
        const parsed = JSON.parse(editor.value);
        expect(parsed).toEqual([
            { type: 'phone', start: 34, end: 45, text: '+1-555-0100' },
        ]);
        expect(editor.value).toContain('  "type"');
    });

    it('span-extraction chips: removing the last chip clears the editor to empty', async () => {
        // Empty string (not "[]") is the canonical "no entities"
        // state — matches the form's initial state + the
        // "empty = negative example" handling.
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: {
                value: JSON.stringify(
                    [{ type: 'email', start: 0, end: 5, text: 'hello' }],
                ),
            },
        });
        await userEvent.click(screen.getByTestId('gold-add-span-chip-0-remove'));

        await waitFor(() => {
            expect(
                screen.queryByTestId('gold-add-span-chip-0'),
            ).not.toBeInTheDocument();
        });
        // Chip container hidden when no chips.
        expect(
            screen.queryByTestId('gold-add-span-chips'),
        ).not.toBeInTheDocument();
        // Editor cleared.
        const editor = screen.getByTestId('gold-add-entities') as HTMLTextAreaElement;
        expect(editor.value).toBe('');
    });

    it('span-extraction chips: unparseable JSON hides the chip list entirely', async () => {
        // Power users editing JSON by hand will pass through invalid
        // intermediate states (mid-typing). Chips should disappear
        // until the JSON re-parses — they never render stale state.
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: '[{"type":"email"' },  // unclosed
        });
        expect(
            screen.queryByTestId('gold-add-span-chips'),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByTestId('gold-add-span-chip-0'),
        ).not.toBeInTheDocument();
    });

    it('span-extraction chips: chips render even when individual offsets fail validation', async () => {
        // A user could have a row with one broken span and want to
        // delete it via ✕ rather than hand-fix the JSON. The chip
        // list parses the raw JSON, NOT spanValidation.spans (which
        // is empty when validation fails), so chips remain
        // actionable in this state.
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-span-text'), {
            target: { value: 'abcdefghij' },
        });
        // text[0:3] = "abc" but the user wrote "xyz" — offset
        // mismatch, validation will reject the row.
        const brokenJson = JSON.stringify(
            [{ type: 'tag', start: 0, end: 3, text: 'xyz' }],
        );
        fireEvent.change(screen.getByTestId('gold-add-entities'), {
            target: { value: brokenJson },
        });

        // Validation error visible.
        expect(
            screen.getByTestId('gold-add-entities-error').textContent,
        ).toMatch(/offset mismatch/);
        // BUT the chip still renders so the user can delete it.
        const chip = screen.getByTestId('gold-add-span-chip-0');
        expect(chip.textContent).toContain('tag');
        expect(chip.textContent).toContain('xyz');

        await userEvent.click(screen.getByTestId('gold-add-span-chip-0-remove'));
        await waitFor(() => {
            expect(
                screen.queryByTestId('gold-add-entities-error'),
            ).not.toBeInTheDocument();
        });
    });

    it('span-extraction chips: helper-added span appears as a chip immediately', async () => {
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: 'Contact jane@example.com today' } });
        selectRange(textarea, 8, 24);
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'email' },
        });
        await userEvent.click(screen.getByTestId('gold-add-span-helper-add'));

        // The helper's append + the chip list's parse derive from
        // the same JSON — chip lands immediately.
        const chip = await screen.findByTestId('gold-add-span-chip-0');
        expect(chip.textContent).toContain('email');
        expect(chip.textContent).toContain('jane@example.com');
    });

    // ── Span-type autocomplete + new-type amber warning ─────────────

    it('span-type datalist surfaces existing types from the project gold rows', async () => {
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    text: 'Contact jane@example.com today',
                    entities: [
                        { type: 'email', start: 8, end: 24, text: 'jane@example.com' },
                    ],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
                {
                    id: 2,
                    text: 'Call +1-555-0100 between 9-5.',
                    entities: [
                        { type: 'phone', start: 5, end: 16, text: '+1-555-0100' },
                    ],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        // Existing-types hint renders next to the Type label.
        const hint = screen.getByTestId('gold-add-span-helper-type-hint');
        expect(hint.textContent).toMatch(/email/);
        expect(hint.textContent).toMatch(/phone/);

        // Input is wired up to a datalist (HTML5 combobox pattern).
        const input = screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement;
        expect(input.getAttribute('list')).toBe('gold-add-span-helper-known-types');
    });

    it('span-type datalist also merges in legacy-shape entity types', async () => {
        // Template-seeded rows store entities JSON-encoded in
        // ``answer``. The extractor normalizes each entry first so
        // these legacy types still surface in the autocomplete.
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    question: "Provider's total liability shall not exceed fees.",
                    answer: JSON.stringify({
                        entities: [
                            {
                                type: 'liability_cap',
                                start: 0,
                                end: 48,
                                text: "Provider's total liability shall not exceed fees",
                            },
                        ],
                    }),
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');
        expect(
            screen.getByTestId('gold-add-span-helper-type-hint').textContent,
        ).toMatch(/liability_cap/);
    });

    it('span-type input: typing an existing type does NOT trigger the amber-border warning', async () => {
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    text: 'email@example.com',
                    entities: [
                        { type: 'email', start: 0, end: 17, text: 'email@example.com' },
                    ],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const typeInput = screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement;
        fireEvent.change(typeInput, { target: { value: 'email' } });
        expect(typeInput.getAttribute('data-new-type')).toBe('false');
        expect(
            screen.queryByTestId('gold-add-span-helper-type-new-hint'),
        ).not.toBeInTheDocument();
    });

    it('span-type input: typing a brand-new type tints the border amber + shows the hint', async () => {
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    text: 'sample',
                    entities: [
                        { type: 'email', start: 0, end: 6, text: 'sample' },
                    ],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const typeInput = screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement;
        fireEvent.change(typeInput, { target: { value: 'ssn' } });
        expect(typeInput.getAttribute('data-new-type')).toBe('true');
        const hint = await screen.findByTestId('gold-add-span-helper-type-new-hint');
        expect(hint.textContent).toMatch(/New type/);
    });

    it('span-type match is case-insensitive (Email == email)', async () => {
        installGetRouter({
            recipeId: 'span-extraction',
            entries: [
                {
                    id: 1,
                    text: 'sample',
                    entities: [
                        { type: 'email', start: 0, end: 6, text: 'sample' },
                    ],
                    difficulty: 'medium',
                    is_hallucination_trap: false,
                },
            ],
        });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const typeInput = screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement;
        fireEvent.change(typeInput, { target: { value: 'Email' } });
        // Existing "email" matches case-insensitively → not flagged
        // as new (avoids amber-border noise when the user just
        // capitalizes a known type).
        expect(typeInput.getAttribute('data-new-type')).toBe('false');
    });

    it('span-type datalist merges in-progress types (added via helper) without saving', async () => {
        // The user just added a "person" span via the helper; the
        // chip is in the JSON editor but the row hasn't been saved
        // to the gold set yet. "person" should ALREADY show up in
        // the datalist for the next add.
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: 'Alice met Bob today.' } });
        selectRange(textarea, 0, 5);  // "Alice"
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'person' },
        });
        await userEvent.click(screen.getByTestId('gold-add-span-helper-add'));

        // After add, type input cleared. Now the user starts a
        // second span — the datalist hint should already include
        // "person" because it's in the in-progress entities.
        await waitFor(() => {
            const hint = screen.getByTestId('gold-add-span-helper-type-hint');
            expect(hint.textContent).toMatch(/person/);
        });
        // And typing "person" again is NOT flagged as a new type.
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'person' },
        });
        expect(
            (screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement)
                .getAttribute('data-new-type'),
        ).toBe('false');
    });

    it('span-type warning does not block submit — soft hint only', async () => {
        // Amber border is signal, not a blocker. With a valid
        // selection + a brand-new type, the helper add button
        // stays enabled and the click still succeeds.
        installGetRouter({ recipeId: 'span-extraction', entries: [] });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        const textarea = screen.getByTestId('gold-add-span-text') as HTMLTextAreaElement;
        fireEvent.change(textarea, { target: { value: 'My SSN is 123-45-6789.' } });
        selectRange(textarea, 10, 21);  // "123-45-6789"
        // No existing types → "ssn" is brand-new.
        fireEvent.change(screen.getByTestId('gold-add-span-helper-type'), {
            target: { value: 'ssn' },
        });
        expect(
            (screen.getByTestId('gold-add-span-helper-type') as HTMLInputElement)
                .getAttribute('data-new-type'),
        ).toBe('true');
        // But the add button is enabled.
        expect(
            (screen.getByTestId('gold-add-span-helper-add') as HTMLButtonElement).disabled,
        ).toBe(false);
        await userEvent.click(screen.getByTestId('gold-add-span-helper-add'));
        // Chip lands.
        await screen.findByTestId('gold-add-span-chip-0');
    });

    it('form resets after a successful submit', async () => {
        installGetRouter({ recipeId: 'qa-sft', entries: [] });
        apiMock.post.mockResolvedValue({ data: { id: 1 } });
        render(<GoldSetPanel projectId={1} />);
        await screen.findByTestId('gold-add-form');

        fireEvent.change(screen.getByTestId('gold-add-question'), {
            target: { value: 'Reset Q?' },
        });
        fireEvent.change(screen.getByTestId('gold-add-answer'), {
            target: { value: 'Reset A.' },
        });
        await userEvent.click(screen.getByTestId('gold-add-submit'));

        await waitFor(() => {
            expect(
                (screen.getByTestId('gold-add-question') as HTMLInputElement).value,
            ).toBe('');
        });
        expect(
            (screen.getByTestId('gold-add-answer') as HTMLTextAreaElement).value,
        ).toBe('');
    });
});
