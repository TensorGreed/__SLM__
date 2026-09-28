import { render, screen, waitFor, within } from '@testing-library/react';
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

vi.mock('../../api/client', () => ({
  default: apiMock,
}));

import ChatPlaygroundPanel from './ChatPlaygroundPanel';

describe('ChatPlaygroundPanel', () => {
  beforeEach(() => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/playground/sessions/') && url.endsWith('/7')) {
        return {
          data: {
            id: 7,
            title: 'Saved Session',
            provider: 'mock',
            model_name: 'acme/test-model',
            messages: [
              { role: 'user', content: 'What is saved?' },
              { role: 'assistant', content: 'This response came from history.' },
            ],
          },
        };
      }
      if (url.includes('/training/playground/sessions')) {
        return {
          data: {
            sessions: [
              {
                id: 7,
                title: 'Saved Session',
                provider: 'mock',
                model_name: 'acme/test-model',
                message_count: 2,
              },
            ],
          },
        };
      }
      if (url.includes('/training/playground/models')) {
        return {
          data: {
            default_model_name: 'acme/test-model',
            models: [
              {
                model_name: 'acme/test-model',
                label: 'Test Model',
                source: 'project',
              },
            ],
          },
        };
      }
      if (url.includes('/training/playground/providers')) {
        return {
          data: {
            providers: [
              { provider: 'mock', label: 'Mock' },
              { provider: 'openai_compatible', label: 'OpenAI-Compatible' },
              { provider: 'llama_cpp', label: 'llama.cpp' },
            ],
          },
        };
      }
      if (url.includes('/training/playground/logs')) {
        return {
          data: {
            summary: {
              event_count: 0,
              positive_count: 0,
              negative_count: 0,
            },
            events: [],
          },
        };
      }
      return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: {} });
  });

  it('restores messages when opening a saved session', async () => {
    const user = userEvent.setup();
    render(<ChatPlaygroundPanel projectId={42} />);

    const sessionButton = await screen.findByRole('button', { name: /Saved Session/i });
    await user.click(sessionButton);

    await waitFor(() => {
      expect(screen.getByText('This response came from history.')).toBeInTheDocument();
      expect(screen.getByText('What is saved?')).toBeInTheDocument();
    });
  });

  it('saves feedback log for assistant response', async () => {
    const user = userEvent.setup();
    render(<ChatPlaygroundPanel projectId={42} />);

    const sessionButton = await screen.findByRole('button', { name: /Saved Session/i });
    await user.click(sessionButton);

    await screen.findByText('This response came from history.');
    await user.click(screen.getByRole('button', { name: 'Mark Good' }));
    await user.click(screen.getByRole('button', { name: 'Save Feedback Log' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/42/training/playground/logs',
        expect.objectContaining({
          rating: 1,
          reply: 'This response came from history.',
        }),
      );
    });
  });

  // ─────────────────────────────────────────────────────────────────
  // Arc 1 — per-turn provenance footer (adapter id + RAG hits +
  // latency vs. session average). Non-streaming path so the test
  // can use the typed apiMock instead of intercepting fetch.
  // ─────────────────────────────────────────────────────────────────

  async function _sendChatWithMock(
    chatResponse: Record<string, unknown>,
    {
      input = 'Why does my refund take 7 days?',
    }: { input?: string } = {},
  ) {
    const user = userEvent.setup();
    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/training/playground/chat') && !url.includes('/stream')) {
        return { data: chatResponse };
      }
      return { data: {} };
    });
    render(<ChatPlaygroundPanel projectId={42} />);
    // Disable streaming so the panel uses the typed apiMock.post path.
    await user.click(screen.getByRole('checkbox', { name: /Stream responses/i }));
    const textarea = await screen.findByPlaceholderText(/Write your prompt/i);
    await user.clear(textarea);
    await user.type(textarea, input);
    await user.click(screen.getByRole('button', { name: /^Send$/i }));
  }

  it('renders the per-turn provenance footer with adapter + RAG hits + latency', async () => {
    await _sendChatWithMock({
      reply: 'Refunds are processed within 5-7 business days.',
      session_id: 11,
      provider: 'mock',
      model_name: 'acme/test-model',
      resolved_model_name: 'projects/42/experiments/9/adapter',
      resolved_provider: 'mock',
      latency_ms: 320.5,
      auto_rag: {
        applied: true,
        k: 2,
        query: 'refund processing time',
        retrieved: [
          {
            row_id: 'gold-12',
            score: 14.5,
            payload: { question: 'How long do refunds take?', answer: 'Up to 7 days.' },
          },
          {
            row_id: 'gold-44',
            score: 9.1,
            payload: { question: 'Refund policy', answer: 'Within 7 business days.' },
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Refunds are processed within 5-7 business days.')).toBeInTheDocument();
    });
    // Footer exists once for the new assistant turn.
    const footers = screen.getAllByTestId(/^playground-provenance-assistant-\d+$/);
    expect(footers.length).toBe(1);

    // Adapter chip carries a truncated id wrapped in <code>.
    const footer = footers[0];
    expect(footer.textContent).toMatch(/via/);
    expect(footer.textContent).toMatch(/projects\/42|adapter/);

    // Latency chip carries the raw ms value (1 sample → no delta yet).
    expect(footer.textContent).toMatch(/320\s?ms|321\s?ms/);

    // RAG chip carries the hit count + top score.
    expect(footer.textContent).toMatch(/RAG: 2 hits/);
    expect(footer.textContent).toMatch(/top 14\.50/);
  });

  it('expands retrieved RAG chunks on click and collapses again', async () => {
    await _sendChatWithMock({
      reply: 'Refunds: 5-7 days.',
      session_id: 11,
      latency_ms: 200,
      auto_rag: {
        applied: true,
        retrieved: [
          {
            row_id: 'gold-12',
            score: 12.5,
            payload: { question: 'How long do refunds take?', answer: 'Up to 7 days.' },
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Refunds: 5-7 days.')).toBeInTheDocument();
    });
    const ragButton = screen.getByTestId(/playground-provenance-.*-rag/);
    // Closed by default.
    expect(screen.queryByTestId(/playground-provenance-.*-hits/)).toBeNull();

    const user = userEvent.setup();
    await user.click(ragButton);

    const hits = screen.getByTestId(/playground-provenance-.*-hits/);
    expect(hits.textContent).toMatch(/gold-12/);
    expect(hits.textContent).toMatch(/score 12\.500/);
    expect(hits.textContent).toMatch(/How long do refunds take\?/);

    await user.click(ragButton);
    expect(screen.queryByTestId(/playground-provenance-.*-hits/)).toBeNull();
  });

  it('flags RAG skip_reason when retrieval was intended but fell back', async () => {
    await _sendChatWithMock({
      reply: 'Best-effort answer without index.',
      session_id: 11,
      latency_ms: 180,
      auto_rag: {
        applied: false,
        retrieved: [],
        skip_reason: 'no bm25 index for project',
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Best-effort answer without index.')).toBeInTheDocument();
    });
    const ragButton = screen.getByTestId(/playground-provenance-.*-rag/);
    expect(ragButton.textContent).toMatch(/RAG skipped/);
    expect(ragButton.textContent).toMatch(/no bm25 index/);
    // Button is disabled — nothing to expand.
    expect((ragButton as HTMLButtonElement).disabled).toBe(true);
  });

  it('does not render the provenance footer when the response omits all metadata', async () => {
    await _sendChatWithMock({
      reply: 'Plain reply.',
      session_id: 11,
    });

    await waitFor(() => {
      expect(screen.getByText('Plain reply.')).toBeInTheDocument();
    });
    // No adapter id, no latency, no auto_rag → footer's chip row is
    // empty, but the wrapper itself still mounts. Verify by ensuring
    // none of the chip testids exist.
    expect(screen.queryByTestId(/playground-provenance-.*-adapter/)).toBeNull();
    expect(screen.queryByTestId(/playground-provenance-.*-latency/)).toBeNull();
    expect(screen.queryByTestId(/playground-provenance-.*-rag/)).toBeNull();
  });

  it('opens on the latest completed run and chats with it in-process', async () => {
    const baseGet = apiMock.get.getMockImplementation();
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/training/experiments')) {
        return {
          data: [
            { id: 3, name: 'first', status: 'completed', base_model: 'acme/base', output_dir: '/d/3', config: {} },
            { id: 5, name: 'latest', status: 'completed', base_model: 'acme/base', output_dir: '/d/5', config: {} },
            { id: 6, name: 'running', status: 'running', base_model: 'acme/base', output_dir: '/d/6', config: {} },
            { id: 7, name: 'Baseline', status: 'completed', base_model: 'acme/base', output_dir: null, config: { is_baseline: true } },
          ],
        };
      }
      return baseGet ? baseGet(url) : { data: {} };
    });
    await _sendChatWithMock({ reply: 'Routed to Billing.', provider: 'experiment' });

    const picker = (await screen.findByLabelText('Training run')) as HTMLSelectElement;
    expect(picker.value).toBe('5');
    expect(Array.from(picker.options).map((o) => o.value)).toEqual(['5', '3']);
    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/42/training/playground/chat',
        expect.objectContaining({ provider: 'experiment', experiment_id: 5, api_url: undefined }),
      );
    });
  });

  it('grounds answers in document passages by default and shows citations', async () => {
    const baseGet = apiMock.get.getMockImplementation();
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.endsWith('/auto-rag/documents')) {
        return { data: { available: true, passages: 42 } };
      }
      return baseGet ? baseGet(url) : { data: {} };
    });
    await _sendChatWithMock({
      reply: 'The K-7 has a 14-month warranty [1].',
      provider: 'mock',
      auto_rag: {
        applied: true,
        corpus: 'documents',
        retrieved: [
          { row_id: '5:2', score: 7.1, payload: { text: 'The K-7 carries a 14-month warranty.', source_doc: 'manual.pdf', chunk_id: 2 } },
          { row_id: '5:7', score: 3.2, payload: { text: 'Descale every 6 weeks.', source_doc: 'manual.pdf', chunk_id: 7 } },
        ],
      },
    });

    const toggle = within(await screen.findByTestId('playground-ground-docs')).getByRole('checkbox') as HTMLInputElement;
    expect(toggle.checked).toBe(true);
    expect(screen.getByTestId('playground-ground-docs').textContent).toContain('42 passages');
    await waitFor(() =>
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/42/training/playground/chat',
        expect.objectContaining({ auto_rag: true }),
      ),
    );
    const chip = await screen.findByTestId(/playground-provenance-assistant-\d+-rag/);
    expect(chip.textContent).toContain('Sources: 2 passages');
    await userEvent.setup().click(chip);
    expect(screen.getByTestId(/playground-provenance-assistant-\d+-cite-1/).textContent).toContain('[1] manual.pdf · passage 3');
  });
});
