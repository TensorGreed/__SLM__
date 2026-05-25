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

import DomainProfileManager from './DomainProfileManager';

const profiles = [
  {
    id: 1,
    profile_id: 'generic-domain-v1',
    version: '1.0.0',
    display_name: 'Generic Domain',
    description: 'Default profile',
    owner: 'platform',
    status: 'active',
    schema_ref: 'slm.domain-profile/v1',
    is_system: true,
  },
  {
    id: 43,
    profile_id: 'policy-qa-profile-v1',
    version: '1.0.0',
    display_name: 'Policy Q&A Profile',
    description: 'Draft from Data Studio',
    owner: 'workspace',
    status: 'draft',
    schema_ref: 'slm.domain-profile/v1',
    is_system: false,
  },
] as const;

function mockProfileApi() {
  apiMock.get.mockImplementation((url: string) => {
    if (url === '/domain-profiles') {
      return Promise.resolve({ data: { profiles, count: profiles.length } });
    }
    if (url === '/domain-profiles/policy-qa-profile-v1') {
      return Promise.resolve({
        data: {
          ...profiles[1],
          contract: {
            $schema: 'slm.domain-profile/v1',
            profile_id: 'policy-qa-profile-v1',
            display_name: 'Policy Q&A Profile',
            status: 'draft',
            tasks: [
              {
                task_id: 'policy-qa',
                output_mode: 'text',
                required_fields: ['question', 'answer'],
                optional_fields: ['context', 'policy_section'],
              },
            ],
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

describe('DomainProfileManager', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.put.mockReset();
    apiMock.post.mockReset();
    mockProfileApi();
  });

  it('keeps a user-selected draft profile selected and opens that contract', async () => {
    render(<DomainProfileManager projectId={7} activeDomainProfileId={1} />);

    const select = await screen.findByRole('combobox');
    expect(select).toHaveValue('generic-domain-v1');

    fireEvent.change(select, { target: { value: 'policy-qa-profile-v1' } });

    await waitFor(() => {
      expect(select).toHaveValue('policy-qa-profile-v1');
    });

    fireEvent.click(screen.getByRole('button', { name: /View\/Edit Contract/i }));

    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/domain-profiles/policy-qa-profile-v1');
    });
    expect(await screen.findByText(/Edit policy-qa-profile-v1/i)).toBeInTheDocument();
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toContain('policy-qa-profile-v1');
  });

  it('lets users select a draft profile from the list before assigning it', async () => {
    render(<DomainProfileManager projectId={7} activeDomainProfileId={1} />);

    fireEvent.click(await screen.findByRole('button', { name: /Policy Q&A Profile/i }));
    fireEvent.click(screen.getByRole('button', { name: /Assign to Project/i }));

    await waitFor(() => {
      expect(apiMock.put).toHaveBeenCalledWith('/projects/7/domain-profile', {
        profile_id: 'policy-qa-profile-v1',
      });
    });
  });
});
