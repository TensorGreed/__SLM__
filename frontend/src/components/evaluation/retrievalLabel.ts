/** A retrieval setting as the backend normalises it
 *  (`auto_rag_service.normalize_retrieval`): top-k, reranker (null = BM25
 *  only), the reranker's BM25 candidate pool (null = the default rule) and
 *  a re-chunked passage variant in chars (null = the cleaning's chunks). */
export interface RetrievalSetting {
    k: number;
    reranker: string | null;
    pool?: number | null;
    chunk_size?: number | null;
}

/** "top-5 + reranker ms-marco-MiniLM-L-6-v2 · pool 50 · 600-char passages"
 *  — the backend's `retrieval_label`, with "(BM25 only)" spelled out when
 *  `bm25Note` is set. */
export function retrievalLabel(r: RetrievalSetting, bm25Note = true): string {
    let label = `top-${r.k}`;
    if (r.reranker) {
        label += ` + reranker ${r.reranker.split('/').pop()}`;
        if (r.pool) label += ` · pool ${r.pool}`;
    } else if (bm25Note) {
        label += ' (BM25 only)';
    }
    if (r.chunk_size) label += ` · ${r.chunk_size}-char passages`;
    return label;
}

/** One key per setting — the same four fields the sweep and the pin log use. */
export function retrievalKey(r: RetrievalSetting | null | undefined): string {
    if (!r) return '';
    return `${r.k}|${r.reranker || ''}|${(r.reranker && r.pool) || ''}|${r.chunk_size || ''}`;
}

/** The PUT /auto-rag/retrieval body: pool / chunk_size only when set. */
export function retrievalPinBody(r: RetrievalSetting): Record<string, number | string | null> {
    const body: Record<string, number | string | null> = { k: r.k, reranker: r.reranker };
    if (r.reranker && r.pool) body.pool = r.pool;
    if (r.chunk_size) body.chunk_size = r.chunk_size;
    return body;
}
