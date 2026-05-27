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
