import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ComplianceBundlePicker, { mergeEntityTypes, textToTypeRegimes, typeRegimesToText } from './ComplianceBundlePicker';

const BUNDLES = {
    builtin: [
        { id: 'pci_dss', regime: 'pci_dss', regime_label: 'PCI-DSS', name: 'PCI-DSS', description: 'cards', applied: false,
            types: [{ type: 'credit_card', definition: 'a card' }, { type: 'cvv', definition: 'a code' }] },
        { id: 'gdpr', regime: 'gdpr', regime_label: 'GDPR', name: 'GDPR personal data', description: 'people', applied: true,
            types: [{ type: 'email', definition: 'an email' }] },
    ],
    bundles: ['gdpr'],
    custom: [],
    type_regimes: { email: ['gdpr'] },
    entity_types: ['email'],
    definitions: {},
    regimes: [],
};

describe('ComplianceBundlePicker', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
    });

    it('merges entity types and round-trips regime lines', () => {
        expect(mergeEntityTypes(['ssn', 'email', 'diagnosis'], ['email', 'diagnosis'], ['email', 'cvv'])).toEqual(['ssn', 'email', 'cvv']);
        const map = { credit_card: ['pci_dss', 'vault'], email: ['gdpr'] };
        expect(textToTypeRegimes(typeRegimesToText(map))).toEqual(map);
        expect(textToTypeRegimes('no colon\nx:\n: y')).toEqual({});
    });

    it('applies newly picked bundles, removes unpicked ones, and fills the entity types', async () => {
        apiMock.get.mockResolvedValue({ data: BUNDLES });
        apiMock.post.mockResolvedValue({
            data: { ...BUNDLES, bundles: ['pci_dss'], type_regimes: { credit_card: ['pci_dss'], cvv: ['pci_dss'] },
                entity_types: ['credit_card', 'cvv'], definitions_added: ['credit_card'], definitions_kept: ['cvv'] },
        });
        const onChange = vi.fn();
        const user = userEvent.setup();
        render(<ComplianceBundlePicker projectId={31} entityTypes={['ssn', 'email']} onEntityTypesChange={onChange} />);
        await user.click(screen.getByTestId('compliance-bundles-toggle'));
        await waitFor(() => expect(screen.getByTestId('compliance-bundle-check-gdpr')).toBeChecked());
        expect(screen.getByTestId('compliance-bundle-gdpr')).toHaveTextContent('(applied)');
        expect(screen.getByTestId('compliance-bundles-apply')).toBeDisabled();
        await user.click(screen.getByTestId('compliance-bundle-check-pci_dss'));
        await user.click(screen.getByTestId('compliance-bundle-check-gdpr'));
        await user.click(screen.getByTestId('compliance-bundles-apply'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/31/compliance/apply', {
            bundle_ids: ['pci_dss'], remove_bundle_ids: ['gdpr'],
        }));
        await waitFor(() => expect(onChange).toHaveBeenCalledWith(['ssn', 'credit_card', 'cvv']));
        expect(screen.getByTestId('compliance-type-regimes-text')).toHaveValue('credit_card: pci_dss\ncvv: pci_dss');
    });

    it('saves edited regime tags, custom ones included', async () => {
        apiMock.get.mockResolvedValue({ data: BUNDLES });
        apiMock.put.mockImplementation(async (_url: string, body: { type_regimes: Record<string, string[]> }) => ({
            data: { ...BUNDLES, type_regimes: body.type_regimes, entity_types: Object.keys(body.type_regimes), custom: ['vault'] },
        }));
        const user = userEvent.setup();
        render(<ComplianceBundlePicker projectId={31} entityTypes={['email']} onEntityTypesChange={vi.fn()} />);
        await user.click(screen.getByTestId('compliance-bundles-toggle'));
        const text = await screen.findByTestId('compliance-type-regimes-text');
        await user.clear(text);
        await user.type(text, 'email: gdpr, vault');
        await user.click(screen.getByTestId('compliance-type-regimes-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/31/compliance/type-regimes', {
            type_regimes: { email: ['gdpr', 'vault'] },
        }));
    });
});
