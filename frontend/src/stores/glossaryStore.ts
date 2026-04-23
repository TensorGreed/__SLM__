import { create } from 'zustand';
import api from '../api/client';

export interface GlossaryEntry {
    term: string;
    plain_language: string;
    category: string;
    example?: string | null;
}

interface GlossaryState {
    entries: Record<string, GlossaryEntry>;
    loading: boolean;
    loaded: boolean;
    error: string | null;
    fetchGlossary: () => Promise<void>;
    reset: () => void;
}

export const useGlossaryStore = create<GlossaryState>((set, get) => ({
    entries: {},
    loading: false,
    loaded: false,
    error: null,

    fetchGlossary: async () => {
        const state = get();
        if (state.loaded || state.loading) {
            return;
        }
        set({ loading: true, error: null });
        try {
            const res = await api.get('/domain-blueprints/glossary/help');
            const raw: GlossaryEntry[] = res.data?.entries || [];
            const byTerm: Record<string, GlossaryEntry> = {};
            for (const entry of raw) {
                if (entry?.term) {
                    byTerm[entry.term.toLowerCase()] = entry;
                }
            }
            set({ entries: byTerm, loading: false, loaded: true, error: null });
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load glossary.';
            // Mark loaded=true even on error so we don't retry forever in the same session.
            set({ loading: false, loaded: true, error: message });
        }
    },

    reset: () => set({ entries: {}, loading: false, loaded: false, error: null }),
}));

/**
 * Look up a glossary entry by backend term key (case-insensitive).
 * Returns null if the glossary hasn't been loaded or the term isn't in the glossary.
 */
export function lookupGlossaryEntry(term: string): GlossaryEntry | null {
    const entries = useGlossaryStore.getState().entries;
    return entries[term.toLowerCase()] || null;
}
