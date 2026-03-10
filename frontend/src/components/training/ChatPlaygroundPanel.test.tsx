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
});
