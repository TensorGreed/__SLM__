import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import SpanTypeDefinitionsEditor, { definitionsToText, textToDefinitions } from './SpanTypeDefinitionsEditor';

describe('SpanTypeDefinitionsEditor', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('round-trips "type: definition" lines', () => {
        const defs = { date_of_birth: 'birth date only', iban: 'an IBAN: two letters, two digits' };
        expect(textToDefinitions(definitionsToText(defs))).toEqual(defs);
        expect(textToDefinitions('no colon here\n: empty type\nemail:')).toEqual({});
    });

    it('saves the scanned-text toggle on its own, without touching the definitions', async () => {
        apiMock.get.mockResolvedValue({ data: { definitions: { email: 'an email address' }, types: ['email'], suggestions: {}, scan_tolerant: false } });
        apiMock.put.mockResolvedValue({ data: { definitions: { email: 'an email address' }, scan_tolerant: true } });
        const user = userEvent.setup();
        render(<SpanTypeDefinitionsEditor projectId={31} />);
        await user.click(screen.getByTestId('span-type-definitions-toggle'));
        const box = await screen.findByTestId('span-scan-tolerant');
        await waitFor(() => expect(box).not.toBeChecked());
        await user.click(box);
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/31/synthetic/type-definitions', { scan_tolerant: true }));
        expect(box).toBeChecked();
    });

    it('loads, adds suggested wording, and saves', async () => {
        apiMock.get.mockResolvedValue({ data: { definitions: { email: 'an email address' }, types: ['email', 'bank_account'], suggestions: { bank_account: 'a domestic account; an IBAN is iban' } } });
        apiMock.put.mockImplementation(async (_url: string, body: { definitions: Record<string, string> }) => ({ data: { definitions: body.definitions } }));
        const user = userEvent.setup();
        render(<SpanTypeDefinitionsEditor projectId={31} />);
        await user.click(screen.getByTestId('span-type-definitions-toggle'));
        await waitFor(() => expect(screen.getByTestId('span-type-definitions-text')).toHaveValue('email: an email address'));
        await user.click(screen.getByTestId('span-type-definitions-suggest'));
        expect(screen.getByTestId('span-type-definitions-text')).toHaveValue('bank_account: a domestic account; an IBAN is iban\nemail: an email address');
        await user.click(screen.getByTestId('span-type-definitions-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/31/synthetic/type-definitions', {
            definitions: { bank_account: 'a domestic account; an IBAN is iban', email: 'an email address' },
        }));
    });
});
