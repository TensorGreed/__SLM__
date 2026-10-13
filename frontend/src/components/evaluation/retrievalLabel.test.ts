import { describe, expect, it } from 'vitest';

import { retrievalKey, retrievalLabel, retrievalPinBody } from './retrievalLabel';

const RERANK = 'cross-encoder/ms-marco-MiniLM-L-6-v2';

describe('retrievalLabel', () => {
    it('mirrors the backend label: pool only with a reranker, passage size when re-chunked', () => {
        expect(retrievalLabel({ k: 3, reranker: null })).toBe('top-3 (BM25 only)');
        expect(retrievalLabel({ k: 3, reranker: null }, false)).toBe('top-3');
        expect(retrievalLabel({ k: 5, reranker: RERANK, pool: 50, chunk_size: 600 })).toBe('top-5 + reranker ms-marco-MiniLM-L-6-v2 · pool 50 · 600-char passages');
        expect(retrievalLabel({ k: 3, reranker: null, pool: 50, chunk_size: 1200 })).toBe('top-3 (BM25 only) · 1200-char passages');
    });

    it('keys and pin bodies ignore a pool without a reranker', () => {
        expect(retrievalKey({ k: 3, reranker: null, pool: 20 })).toBe(retrievalKey({ k: 3, reranker: null }));
        expect(retrievalKey({ k: 3, reranker: RERANK, pool: 20 })).not.toBe(retrievalKey({ k: 3, reranker: RERANK }));
        expect(retrievalPinBody({ k: 3, reranker: null, pool: 20 })).toEqual({ k: 3, reranker: null });
        expect(retrievalPinBody({ k: 5, reranker: RERANK, pool: 50, chunk_size: 600 })).toEqual({ k: 5, reranker: RERANK, pool: 50, chunk_size: 600 });
    });
});
