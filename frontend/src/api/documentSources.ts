/**
 * Typed client for the document-source importers (website crawl, git
 * repository). Mirrors ``backend/app/api/document_sources.py``: both
 * endpoints start a Job (``?async_job=true``, 202) whose ``result`` is a
 * ``DocumentSourceResult``.
 */

import api from './client';
import type { Job } from './jobs';


export interface WebCrawlRequest {
    seed_urls: string[];
    allowed_prefixes?: string[];
    sitemap_url?: string | null;
    max_pages?: number;
    max_depth?: number;
    delay_seconds?: number;
    include_patterns?: string[];
    exclude_patterns?: string[];
    /** Licence of the crawled site, recorded in the provenance ledger. */
    licence?: string | null;
}

export interface GitRepoRequest {
    url: string;
    ref?: string;
    include_globs?: string[];
    exclude_globs?: string[];
    /** Skip vendored / build folders (node_modules, dist, …) — default true. */
    default_excludes?: boolean;
    max_file_bytes?: number;
}

export interface ImportedDocument {
    id: number;
    filename: string;
    source_url?: string | null;
    path?: string | null;
}

export interface RejectedSource {
    url?: string | null;
    path?: string | null;
    reason: string;
    detail?: string | number | null;
}

export interface DocumentSourceResult {
    source: 'web_crawl' | 'git_repo';
    documents_created: number;
    document_ids: number[];
    documents: ImportedDocument[];
    rejected: RejectedSource[];
    rejection_counts: Record<string, number>;
    limit_reached: number;
    commit?: string | null;
    ref?: string | null;
    licence?: string | null;
}


export async function startWebCrawl(projectId: number, body: WebCrawlRequest): Promise<Job> {
    const resp = await api.post<Job>(
        `/projects/${projectId}/document-sources/web-crawl?async_job=true`, body);
    return resp.data;
}


export async function startGitRepoImport(projectId: number, body: GitRepoRequest): Promise<Job> {
    const resp = await api.post<Job>(
        `/projects/${projectId}/document-sources/git-repo?async_job=true`, body);
    return resp.data;
}
