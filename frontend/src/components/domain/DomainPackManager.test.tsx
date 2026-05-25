import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    put: vi.fn(),
    post: vi.fn(),
  },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DomainPackManager from './DomainPackManager';

const packs = [
  {
    id: 1,
    pack_id: 'general-pack-v1',
    version: '1.0.0',
    display_name: 'General Domain Pack',
    description: 'Default pack',
    owner: 'platform',
    status: 'active',
    schema_ref: 'slm.domain-pack/v1',
    default_profile_id: 'generic-domain-v1',
    is_system: true,
  },
  {
    id: 42,
    pack_id: 'policy-qa-pack-v1',
    version: '1.0.0',
    display_name: 'Policy Q&A Pack',
    description: 'Draft from Data Studio',
    owner: 'workspace',
    status: 'draft',
    schema_ref: 'slm.domain-pack/v1',
    default_profile_id: 'policy-qa-profile-v1',
    is_system: false,
  },
] as const;

function mockPackApi() {
  apiMock.get.mockImplementation((url: string) => {
    if (url === '/domain-packs') {
      return Promise.resolve({ data: { packs, count: packs.length } });
    }
    if (url === '/domain-packs/hooks/catalog') {
      return Promise.resolve({
        data: {
          normalizers: { 'default-normalizer': 'Default normalizer' },
          validators: { 'default-validator': 'Default validator' },
          evaluators: { 'default-evaluator': 'Default evaluator' },
          plugin_modules_loaded: [],
          plugin_load_errors: {},
          plugin_hook_sources: {},
        },
      });
    }
    if (url === '/domain-packs/policy-qa-pack-v1') {
      return Promise.resolve({
        data: {
          ...packs[1],
          contract: {
            $schema: 'slm.domain-pack/v1',
            pack_id: 'policy-qa-pack-v1',
            display_name: 'Policy Q&A Pack',
            status: 'draft',
            default_profile_id: 'policy-qa-profile-v1',
            hooks: {
              normalizer: { id: 'default-normalizer', config: {} },
              validator: { id: 'default-validator', config: {} },
              evaluator: { id: 'default-evaluator', config: {} },
            },
          },
        },
      });
    }
    return Promise.reject(new Error(`Unhandled GET ${url}`));
  });
  apiMock.put.mockResolvedValue({
    data: {
      id: 7,
      name: 'Policy project',
      domain_pack_id: 42,
      domain_profile_id: 43,
    },
  });
}

describe('DomainPackManager', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.put.mockReset();
    apiMock.post.mockReset();
    mockPackApi();
  });

  it('keeps a user-selected draft pack selected and opens that contract', async () => {
    render(<DomainPackManager projectId={7} activeDomainPackId={1} />);

    const select = await screen.findByRole('combobox');
    expect(select).toHaveValue('general-pack-v1');

    fireEvent.change(select, { target: { value: 'policy-qa-pack-v1' } });

    await waitFor(() => {
      expect(select).toHaveValue('policy-qa-pack-v1');
    });

    fireEvent.click(screen.getByRole('button', { name: /View\/Edit Contract/i }));

    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/domain-packs/policy-qa-pack-v1');
    });
    expect(await screen.findByText(/Edit policy-qa-pack-v1/i)).toBeInTheDocument();
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toContain('policy-qa-pack-v1');
  });

  it('lets users select a draft pack from the list before assigning it', async () => {
    render(<DomainPackManager projectId={7} activeDomainPackId={1} />);

    fireEvent.click(await screen.findByRole('button', { name: /Policy Q&A Pack/i }));
    fireEvent.click(screen.getByRole('button', { name: /Assign to Project/i }));

    await waitFor(() => {
      expect(apiMock.put).toHaveBeenCalledWith('/projects/7/domain-pack', {
        pack_id: 'policy-qa-pack-v1',
        adopt_pack_default_profile: true,
      });
    });
  });
});
