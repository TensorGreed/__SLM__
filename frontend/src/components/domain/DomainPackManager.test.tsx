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
          normalizers: {
            'default-normalizer': 'No-op normalizer; uses canonical record as-is.',
            'safe-cleanup-normalizer': 'Opinionated bundle: HTML-entity decode then whitespace collapse. Recommended starter swap-in for default-normalizer.',
            'whitespace-collapse-normalizer': 'Collapses runs of whitespace + trims string fields.',
          },
          validators: { 'default-validator': 'No-op validator with pass-through summary.' },
          evaluators: {
            'default-evaluator': 'No-op evaluator; metrics unchanged.',
            'metric-coverage-evaluator': 'Tags missing_metric_ids + metric_coverage from config.expected_metric_ids.',
          },
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
    // Slice 2 added per-hook config textareas, so the modal now has
    // multiple textboxes. Target the main contract JSON textarea
    // explicitly via its label so adding more hook fields later
    // doesn't break this assertion again.
    const contractTextarea = screen.getByLabelText('Contract JSON') as HTMLTextAreaElement;
    expect(contractTextarea.value).toContain('policy-qa-pack-v1');
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

  // ─────────────────────────────────────────────────────────────────
  // Gap-#1/#2 slice 2: hook picker with auto-sync, descriptions,
  // recommended/no-op badges, config textarea.
  // ─────────────────────────────────────────────────────────────────

  async function openEditModalForPolicyPack() {
    render(<DomainPackManager projectId={7} activeDomainPackId={1} />);
    const select = await screen.findByRole('combobox');
    fireEvent.change(select, { target: { value: 'policy-qa-pack-v1' } });
    await waitFor(() => {
      expect(select).toHaveValue('policy-qa-pack-v1');
    });
    fireEvent.click(screen.getByRole('button', { name: /View\/Edit Contract/i }));
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/domain-packs/policy-qa-pack-v1');
    });
    await screen.findByText(/Edit policy-qa-pack-v1/i);
  }

  it('hook picker renders description inline + flags the no-op default', async () => {
    await openEditModalForPolicyPack();
    // The normalizer picker preselects default-normalizer (from the
    // contract). Its description should surface inline so the user
    // isn't guessing what's active.
    const description = screen.getByTestId('domain-pack-hook-normalizer-description');
    expect(description.textContent).toMatch(/No-op normalizer/);
    expect(description.className).toMatch(/is-noop/);
  });

  it('hook picker dropdown badges the recommended starter normalizer with a star', async () => {
    await openEditModalForPolicyPack();
    const select = screen.getByTestId('domain-pack-hook-normalizer-select') as HTMLSelectElement;
    // safe-cleanup-normalizer is the slice-1 starter and the slice-2
    // RECOMMENDED_HOOK_IDS entry — its option text starts with ★.
    const recommendedOption = Array.from(select.options).find(
      (opt) => opt.value === 'safe-cleanup-normalizer',
    );
    expect(recommendedOption).toBeTruthy();
    expect(recommendedOption?.textContent).toMatch(/^★/);
  });

  it('changing the normalizer dropdown auto-applies the new id into the contract JSON', async () => {
    await openEditModalForPolicyPack();
    const select = screen.getByTestId('domain-pack-hook-normalizer-select') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'safe-cleanup-normalizer' } });

    const contractTextarea = screen.getByLabelText('Contract JSON') as HTMLTextAreaElement;
    const parsed = JSON.parse(contractTextarea.value);
    expect(parsed.hooks.normalizer.id).toBe('safe-cleanup-normalizer');
    // Other hooks untouched.
    expect(parsed.hooks.validator.id).toBe('default-validator');
  });

  it('editing the per-hook config textarea auto-applies the parsed config into the contract', async () => {
    await openEditModalForPolicyPack();
    const configTextarea = screen.getByTestId('domain-pack-hook-normalizer-config') as HTMLTextAreaElement;
    fireEvent.change(configTextarea, {
      target: { value: '{"target_fields": ["text", "answer"]}' },
    });

    const contractTextarea = screen.getByLabelText('Contract JSON') as HTMLTextAreaElement;
    const parsed = JSON.parse(contractTextarea.value);
    expect(parsed.hooks.normalizer.config).toEqual({
      target_fields: ['text', 'answer'],
    });
  });

  it('invalid config JSON surfaces an inline error + does NOT write a broken config into the contract', async () => {
    await openEditModalForPolicyPack();
    const configTextarea = screen.getByTestId('domain-pack-hook-normalizer-config') as HTMLTextAreaElement;
    fireEvent.change(configTextarea, { target: { value: '{not: valid JSON' } });

    expect(screen.getByText(/Normalizer Hook config JSON is invalid/i)).toBeInTheDocument();
    // Contract still has the previous (empty) config — the malformed
    // edit was rejected, not silently written.
    const contractTextarea = screen.getByLabelText('Contract JSON') as HTMLTextAreaElement;
    const parsed = JSON.parse(contractTextarea.value);
    expect(parsed.hooks.normalizer.config).toEqual({});
  });
});
