import { test, expect } from '@playwright/test';

/**
 * Golden-path gate for a DOCUMENTS project (Epic G, phase G2).
 *
 * A documents-only project (no Q&A rows) must get to a measured decision
 * without the user knowing the steps: upload → clean → the Training tab's
 * "Turn your documents into a Q&A assistant" card → one click → the flow
 * generates pairs + an answer key, splits, runs the pre-training retrieval
 * gate (base model with vs without the passages, judged) and either stops
 * ("retrieval already answers", reroute offered) or trains on the simulate
 * runtime. The card must show which.
 *
 * Generation and judging use the deterministic template backend (the CI
 * stack boots with BREWSLM_TEMPLATE_SYNTH=1 EVAL_JUDGE_BACKEND=template);
 * the base-model inference in the gate is real (SmolLM2-135M, CPU, cached).
 * Path-integrity gate: the gate's verdict may go either way on synthetic
 * text; what must hold is that it ran, was judged, and the UI says so.
 */

const ADMIN = 'admin';
const ADMIN_KEY = 'sk-mock-admin-key';

function statuteHtml(sections = 30): string {
  // Mirrors backend/tests/test_real_training_path.py::_statute_html — keep in sync.
  const duties = ['publish', 'audit', 'seal', 'index', 'translate', 'archive', 'certify', 'redact', 'transmit', 'catalogue',
    'inspect', 'register', 'appraise', 'digitise', 'witness', 'annotate', 'summarise', 'escrow', 'license', 'notarise',
    'quarantine', 'reconcile', 'encrypt', 'tabulate', 'forecast', 'benchmark', 'garnish', 'survey', 'calibrate', 'underwrite'];
  const objects = ['the disclosure register', 'the annual report', 'each notice of refusal', 'the fee schedule', 'every access request',
    'the correction log', 'the retention record', 'the complaint file', 'the exemption list', 'the consent forms',
    'the mining ledger', 'the harbour permits', 'the grain tallies', 'the vaccine lots', 'the broadcast licences',
    'the pension rolls', 'the ferry manifests', 'the timber quotas', 'the census returns', 'the patent abstracts',
    'the rail tariffs', 'the fishery charts', 'the dam inspections', 'the postal routes', 'the bridge tolls',
    'the orchard grants', 'the lighthouse logs', 'the canal leases', 'the museum loans', 'the vineyard bonds'];
  const deadlines = ['thirty days', 'ninety days', 'two years', 'ten business days', 'one fiscal year', 'six weeks', 'the next quarter'];
  const penalties = ['a fine of two hundred dollars', 'suspension of the licence', 'a written reprimand', 'forfeiture of the fee',
    'referral to the tribunal', 'publication of the default', 'a daily penalty of fifty dollars'];
  const paras: string[] = [];
  for (let i = 1; i <= sections; i++) {
    const duty = duties[(i - 1) % duties.length];
    const obj = objects[(i - 1) % objects.length];
    const deadline = deadlines[i % deadlines.length];
    const penalty = penalties[(i * 2) % penalties.length];
    paras.push(`<p>Section ${i}. The officer responsible for section ${i} shall ${duty} ${obj} within ${deadline} of receiving it. `
      + `Failure to ${duty} ${obj} in time carries ${penalty} under section ${i}. `
      + `Once ${obj} has been ${duty}d, the officer shall tell the applicant in writing that section ${i} is complete.</p>`);
  }
  return `<html><body><h1>Synthetic Information Act</h1>${paras.join('')}</body></html>`;
}

test('golden path (documents): upload → flow card → gate verdict on the card', async ({ page }) => {
  test.setTimeout(15 * 60_000);
  // ── 1. Login ──────────────────────────────────────────────────────────
  await page.goto('/login');
  await page.getByPlaceholder('Enter your username').fill(ADMIN);
  await page.getByPlaceholder('API Key or Password').fill(ADMIN_KEY);
  await page.getByRole('button', { name: /^Sign in$/ }).click();
  await page.waitForURL((url) => url.pathname === '/', { timeout: 20_000 });
  const token = await page.evaluate(() => localStorage.getItem('slm_token'));
  const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

  // ── 2. A documents-only project: upload + clean through the API ─────
  const created = await page.request.post('/api/projects', {
    headers: authHeaders,
    data: { name: `golden-docs-${Date.now()}`, description: 'Golden path: documents project' },
  });
  expect(created.ok()).toBeTruthy();
  const projectId = (await created.json()).id as number;
  const api = `/api/projects/${projectId}`;
  const upload = await page.request.post(`${api}/ingestion/upload`, {
    headers: authHeaders,
    multipart: { file: { name: 'synthetic-act.html', mimeType: 'text/html', buffer: Buffer.from(statuteHtml(), 'utf-8') } },
  });
  expect(upload.ok()).toBeTruthy();
  const docId = (await upload.json()).id as number;
  expect((await page.request.post(`${api}/ingestion/documents/${docId}/process`, { headers: authHeaders })).ok()).toBeTruthy();
  expect((await page.request.post(`${api}/cleaning/clean`, {
    headers: authHeaders, data: { document_id: docId, chunk_size: 320, chunk_overlap: 0 },
  })).ok()).toBeTruthy();

  // ── 3. The Training tab offers the flow ──────────────────────────────
  await page.goto(`/project/${projectId}/pipeline/training`);
  const card = page.getByTestId('documents-qa-flow');
  await expect(card).toBeVisible({ timeout: 30_000 });
  await expect(card).toContainText('Turn your documents into a Q&A assistant');
  await expect(page.getByTestId('documents-qa-flow-plan')).toContainText('Before training');

  // ── 4. One click; the Job does the rest ──────────────────────────────
  await page.getByTestId('documents-qa-flow-start').click();
  await expect(page.getByTestId('documents-qa-flow-running')).toBeVisible({ timeout: 30_000 });

  // Poll the Job through the API (robust against UI polling cadence).
  let job: { status: string; error?: string | null; result?: Record<string, unknown> } | null = null;
  const deadline = Date.now() + 12 * 60_000;
  while (Date.now() < deadline) {
    const res = await page.request.get('/api/jobs/active', { headers: authHeaders, params: { include_recently_completed: 'true', limit: '50' } });
    const jobs = ((await res.json()).jobs as Array<Record<string, unknown>>).filter(
      (j) => j.kind === 'documents_qa_flow' && j.project_id === projectId,
    );
    const mine = jobs[0] as typeof job;
    if (mine && ['succeeded', 'failed', 'cancelled'].includes(mine.status)) { job = mine; break; }
    await page.waitForTimeout(5_000);
  }
  expect(job, 'the flow Job did not finish').not.toBeNull();
  expect(job!.status, `flow failed: ${job!.error}`).toBe('succeeded');
  const result = job!.result as { training_pairs: number; passages_gate?: { status: string; judge?: string } ; experiment_id: number | null };
  expect(result.training_pairs).toBeGreaterThanOrEqual(40);
  expect(['retrieval_ready', 'retrieval_promising', 'retrieval_weak']).toContain(result.passages_gate?.status);

  // ── 5. The card says what the gate decided ───────────────────────────
  const gate = page.getByTestId('documents-qa-flow-gate');
  await expect(gate).toBeVisible({ timeout: 30_000 });
  await expect(gate).toContainText('Before training');
  if (result.passages_gate?.status === 'retrieval_ready') {
    await expect(gate).toContainText('already got');
    await expect(page.getByTestId('documents-qa-flow-reroute')).toBeVisible();
    await expect(page.getByTestId('documents-qa-flow-train-anyway')).toBeVisible();
    expect(result.experiment_id).toBeNull();
  } else {
    await expect(gate).toContainText('training went ahead');  // promising or weak
    expect(result.experiment_id).not.toBeNull();
    await expect(page.getByTestId('documents-qa-flow-result')).toContainText(`Training run #${result.experiment_id} started`);
  }
});
