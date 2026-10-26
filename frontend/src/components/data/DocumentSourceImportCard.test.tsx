import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import DocumentSourceImportCard from './DocumentSourceImportCard';

const webResult = {
    source: 'web_crawl', documents_created: 2, document_ids: [11, 12],
    documents: [
        { id: 11, filename: 'guide.html', source_url: 'https://docs.example.com/guide/' },
        { id: 12, filename: 'guide__intro.html', source_url: 'https://docs.example.com/guide/intro' },
    ],
    rejected: [
        { url: 'https://docs.example.com/guide/a', reason: 'http_error', detail: 404 },
        { url: 'https://docs.example.com/guide/b', reason: 'http_error', detail: 500 },
        { url: 'https://docs.example.com/guide/c.pdf', reason: 'not_html', detail: 'application/pdf' },
    ],
    rejection_counts: { http_error: 2, not_html: 1 }, limit_reached: 0,
};

const job = (over: Record<string, unknown>) => ({ id: 7, kind: 'document_source_web_crawl', status: 'queued', project_id: 5, ...over });

describe('DocumentSourceImportCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.post.mockReset(); apiMock.delete.mockReset();
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('starts a crawl, polls progress, shows provenance and grouped rejections', async () => {
        apiMock.post.mockResolvedValue({ data: job({}) });
        apiMock.get
            .mockResolvedValueOnce({ data: job({ status: 'running', progress_message: 'Fetched 1/200 pages' }) })
            .mockResolvedValue({ data: job({ status: 'succeeded', result: webResult }) });
        const user = userEvent.setup();
        const onDone = vi.fn();
        render(<DocumentSourceImportCard projectId={5} onDone={onDone} pollMs={10} />);
        await user.type(screen.getByTestId('dsi-seeds'), 'https://docs.example.com/guide/');
        await user.click(screen.getByTestId('dsi-start-web'));
        expect(apiMock.post).toHaveBeenCalledWith('/projects/5/document-sources/web-crawl?async_job=true', expect.objectContaining({
            seed_urls: ['https://docs.example.com/guide/'], max_pages: 200, max_depth: 3, delay_seconds: 1,
        }));
        await waitFor(() => expect(screen.getByTestId('dsi-progress')).toHaveTextContent('Fetched 1/200 pages'));
        await waitFor(() => expect(screen.getByTestId('dsi-summary')).toHaveTextContent('2 documents imported'));
        expect(onDone).toHaveBeenCalled();
        expect(screen.getByTestId('dsi-doc-label-11')).toHaveTextContent('https://docs.example.com/guide/');
        expect(screen.getByTestId('dsi-group-http_error')).toHaveTextContent('http error (2)');
        expect(screen.getByTestId('dsi-group-not_html')).toHaveTextContent('not html (1)');
    });

    it('retries selected rejected URLs as a depth-0 crawl', async () => {
        apiMock.post.mockResolvedValueOnce({ data: job({}) }).mockResolvedValueOnce({ data: job({ id: 8 }) });
        apiMock.get.mockResolvedValue({ data: job({ status: 'succeeded', result: webResult }) });
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.type(screen.getByTestId('dsi-seeds'), 'https://docs.example.com/guide/');
        await user.click(screen.getByTestId('dsi-start-web'));
        await waitFor(() => screen.getByTestId('dsi-group-all-http_error'));
        await user.click(screen.getByText(/http error/));
        await user.click(screen.getByTestId('dsi-group-all-http_error'));
        await user.click(screen.getByTestId('dsi-retry-selected'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        expect(apiMock.post).toHaveBeenLastCalledWith('/projects/5/document-sources/web-crawl?async_job=true', expect.objectContaining({
            seed_urls: ['https://docs.example.com/guide/a', 'https://docs.example.com/guide/b'], max_depth: 0,
        }));
    });

    it('retry sends only the picked URLs: no include / exclude patterns, no sitemap', async () => {
        apiMock.post.mockResolvedValueOnce({ data: job({}) }).mockResolvedValueOnce({ data: job({ id: 8 }) });
        apiMock.get.mockResolvedValue({ data: job({ status: 'succeeded', result: webResult }) });
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.type(screen.getByTestId('dsi-seeds'), 'https://docs.example.com/guide/');
        await user.click(screen.getByText('Advanced'));
        await user.type(screen.getByTestId('dsi-web-include'), 'guide');
        await user.type(screen.getByTestId('dsi-web-exclude'), 'private');
        await user.type(screen.getByTestId('dsi-sitemap'), 'https://docs.example.com/sitemap.xml');
        await user.click(screen.getByTestId('dsi-start-web'));
        expect(apiMock.post).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({
            include_patterns: ['guide'], exclude_patterns: ['private'], sitemap_url: 'https://docs.example.com/sitemap.xml',
        }));
        await waitFor(() => screen.getByTestId('dsi-group-all-http_error'));
        await user.click(screen.getByTestId('dsi-group-all-http_error'));
        await user.click(screen.getByTestId('dsi-retry-selected'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(2));
        const body = apiMock.post.mock.calls[1][1] as Record<string, unknown>;
        expect(body.include_patterns).toEqual([]);
        expect(body.exclude_patterns).toEqual([]);
        expect(body.sitemap_url).toBeUndefined();
    });

    it('lists duplicate rejected rows without duplicate-key warnings', async () => {
        const dup = { url: 'https://docs.example.com/guide/a', reason: 'http_error', detail: 404 };
        apiMock.post.mockResolvedValue({ data: job({}) });
        apiMock.get.mockResolvedValue({ data: job({ status: 'succeeded', result: { ...webResult, rejected: [dup, dup] } }) });
        const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined);
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.type(screen.getByTestId('dsi-seeds'), 'https://docs.example.com/guide/');
        await user.click(screen.getByTestId('dsi-start-web'));
        await waitFor(() => screen.getByTestId('dsi-group-http_error'));
        expect(errors.mock.calls.some((c) => String(c[0]).includes('same key'))).toBe(false);
        errors.mockRestore();
    });

    it('sends default_excludes, switchable from the Skip vendored / build folders checkbox', async () => {
        apiMock.post.mockResolvedValue({ data: job({ kind: 'document_source_git_repo' }) });
        apiMock.get.mockResolvedValue({ data: job({ kind: 'document_source_git_repo', status: 'running' }) });
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.click(screen.getByTestId('dsi-tab-git'));
        await user.type(screen.getByTestId('dsi-repo-url'), 'https://github.com/org/repo');
        await user.click(screen.getByText('Advanced'));
        expect(screen.getByTestId('dsi-default-excludes')).toBeChecked();
        expect(screen.getByText(/added to the built-in defaults/)).toBeInTheDocument();
        await user.click(screen.getByTestId('dsi-start-git'));
        expect(apiMock.post).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({ default_excludes: true }));
    });

    it('sends default_excludes=false when the box is cleared', async () => {
        apiMock.post.mockResolvedValue({ data: job({ kind: 'document_source_git_repo' }) });
        apiMock.get.mockResolvedValue({ data: job({ kind: 'document_source_git_repo', status: 'running' }) });
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.click(screen.getByTestId('dsi-tab-git'));
        await user.type(screen.getByTestId('dsi-repo-url'), 'https://github.com/org/repo');
        await user.click(screen.getByText('Advanced'));
        await user.click(screen.getByTestId('dsi-default-excludes'));
        await user.click(screen.getByTestId('dsi-start-git'));
        expect(apiMock.post).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({ default_excludes: false }));
    });

    it('starts a git import and labels files path @ commit with the licence; bulk-deletes selected', async () => {
        apiMock.post.mockResolvedValue({ data: job({ kind: 'document_source_git_repo' }) });
        apiMock.delete.mockResolvedValue({ data: {} });
        apiMock.get.mockResolvedValue({ data: job({ kind: 'document_source_git_repo', status: 'succeeded', result: {
            source: 'git_repo', documents_created: 2, document_ids: [21, 22], commit: 'abcdef1234567890', ref: 'v1.2', licence: 'MIT',
            documents: [{ id: 21, filename: 'src__Greeter.java', path: 'src/Greeter.java' }, { id: 22, filename: 'README.md', path: 'README.md' }],
            rejected: [], rejection_counts: {}, limit_reached: 0,
        } }) });
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.click(screen.getByTestId('dsi-tab-git'));
        await user.type(screen.getByTestId('dsi-repo-url'), 'https://github.com/org/repo');
        await user.clear(screen.getByTestId('dsi-ref'));
        await user.type(screen.getByTestId('dsi-ref'), 'v1.2');
        await user.click(screen.getByTestId('dsi-start-git'));
        expect(apiMock.post).toHaveBeenCalledWith('/projects/5/document-sources/git-repo?async_job=true', expect.objectContaining({
            url: 'https://github.com/org/repo', ref: 'v1.2', max_file_bytes: 200000,
        }));
        await waitFor(() => expect(screen.getByTestId('dsi-doc-label-21')).toHaveTextContent('src/Greeter.java @ abcdef1 · MIT'));
        await user.click(screen.getByTestId('dsi-doc-21'));
        await user.click(screen.getByTestId('dsi-delete-selected'));
        await waitFor(() => expect(apiMock.delete).toHaveBeenCalledWith('/projects/5/ingestion/documents/21'));
        await waitFor(() => expect(screen.queryByTestId('dsi-doc-21')).toBeNull());
        expect(screen.getByTestId('dsi-doc-22')).toBeInTheDocument();
    });

    it('shows the failure message when the job fails', async () => {
        apiMock.post.mockResolvedValue({ data: job({}) });
        apiMock.get.mockResolvedValue({ data: job({ status: 'failed', error: 'No pages were imported: robots_disallowed=3' }) });
        const user = userEvent.setup();
        render(<DocumentSourceImportCard projectId={5} pollMs={10} />);
        await user.type(screen.getByTestId('dsi-seeds'), 'https://docs.example.com/guide/');
        await user.click(screen.getByTestId('dsi-start-web'));
        await waitFor(() => expect(screen.getByTestId('dsi-error')).toHaveTextContent('robots_disallowed=3'));
    });
});
