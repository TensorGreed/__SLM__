/**
 * Exercises the REAL shared axios client (no `vi.mock('./client')`): a capturing adapter
 * stands in for the network, so the test sees the request exactly as axios would send it
 * after interceptors + transformRequest. Every upload helper's unit test mocks the client,
 * which is how a JSON default header silently turned FormData bodies into `{"train":{}}`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import api from './client';
import { importPredefinedSplit } from './predefinedSplit';
import { uploadEvalSet } from './evalSets';
import { importGoldFile } from './goldFiles';
import { uploadImportFile } from './datasetImport';
import { importPack, previewPackImport } from './domainPacks';
import { uploadHeldout } from './pipelines';

interface SentRequest {
    method: string;
    url: string;
    headers: Record<string, string>;
    body: unknown;
}

let sent: SentRequest[] = [];
const originalAdapter = api.defaults.adapter;

/** Stands in for the browser's XMLHttpRequest so axios's real xhr adapter (and its
 *  browser-only FormData handling) runs; records what would go on the wire. */
class RecordingXHR {
    readyState = 0;
    status = 0;
    statusText = '';
    responseText = '';
    response: unknown = null;
    responseType = '';
    responseURL = '';
    timeout = 0;
    withCredentials = false;
    upload = { addEventListener: () => undefined };
    onloadend: null | (() => void) = null;
    onabort: null | (() => void) = null;
    onerror: null | (() => void) = null;
    ontimeout: null | (() => void) = null;
    private req: SentRequest = { method: '', url: '', headers: {}, body: null };
    open(method: string, url: string) {
        this.req = { method, url, headers: {}, body: null };
    }
    setRequestHeader(name: string, value: string) {
        this.req.headers[name.toLowerCase()] = value;
    }
    addEventListener() {}
    getAllResponseHeaders() {
        return 'content-type: application/json\r\n';
    }
    abort() {}
    send(body: unknown) {
        this.req.body = body;
        sent.push(this.req);
        setTimeout(() => {
            this.readyState = 4;
            this.status = 200;
            this.responseText = '{}';
            this.response = '{}';
            this.onloadend?.();
        }, 0);
    }
}

function contentType(req: SentRequest): string | undefined {
    return req.headers['content-type'];
}

function file(name: string): File {
    return new File(['{"q":"a"}\n'], name, { type: 'application/jsonl' });
}

beforeEach(() => {
    sent = [];
    // Some Node versions expose a stub global localStorage that shadows jsdom's; pin a working one.
    const store = new Map<string, string>([['slm_token', 't']]);
    vi.stubGlobal('localStorage', {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => store.set(k, v),
        removeItem: (k: string) => store.delete(k),
    });
    vi.stubGlobal('XMLHttpRequest', RecordingXHR);
    api.defaults.adapter = 'xhr';
});

afterEach(() => {
    api.defaults.adapter = originalAdapter;
    vi.unstubAllGlobals();
});

describe('shared api client: FormData uploads', () => {
    it('sends a FormData body as FormData with no forced Content-Type (the browser adds the boundary)', async () => {
        const form = new FormData();
        form.append('file', file('rows.jsonl'));
        await api.post('/x', form);
        expect(sent).toHaveLength(1);
        expect(sent[0].body).toBeInstanceOf(FormData);
        expect((sent[0].body as FormData).get('file')).toBeInstanceOf(File);
        expect(contentType(sent[0])).toBeUndefined();
        expect(sent[0].headers.authorization).toBe('Bearer t');
    });

    it('also clears an explicit boundary-less multipart header', async () => {
        const form = new FormData();
        form.append('file', file('rows.jsonl'));
        await api.post('/x', form, { headers: { 'Content-Type': 'multipart/form-data' } });
        expect(sent[0].body).toBeInstanceOf(FormData);
        expect(contentType(sent[0])).toBeUndefined();
    });

    it('still sends plain objects as JSON', async () => {
        await api.post('/x', { a: 1 });
        expect(sent[0].body).toBe('{"a":1}');
        expect(contentType(sent[0])).toContain('application/json');
    });

    it('every upload helper sends its file(s) as multipart', async () => {
        await importPredefinedSplit(1, { train: file('train.jsonl'), val: file('val.jsonl') }, { onLeak: 'reject' as never });
        await uploadEvalSet(1, file('eval.jsonl'), { name: 'fresh' });
        await uploadEvalSet(1, file('eval.jsonl'), { slug: 'fresh' });
        await importGoldFile(1, file('gold.jsonl'), { datasetType: 'gold_test' as never, mode: 'append' as never });
        await uploadImportFile(1, file('rows.csv'));
        await previewPackImport(file('pack.zip'));
        await importPack(file('pack.zip'));
        await uploadHeldout('p1', file('heldout.jsonl'));
        expect(sent).toHaveLength(8);
        for (const req of sent) {
            expect(req.body, req.url).toBeInstanceOf(FormData);
            expect(contentType(req), req.url).toBeUndefined();
        }
        const split = sent[0].body as FormData;
        expect(split.get('train')).toBeInstanceOf(File);
        expect(split.get('val')).toBeInstanceOf(File);
    });
});
