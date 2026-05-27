import { render, screen, waitFor } from '@testing-library/react';
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
});
