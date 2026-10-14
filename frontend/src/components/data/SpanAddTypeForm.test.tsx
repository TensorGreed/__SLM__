import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import SpanAddTypeForm, { normaliseTypeName } from './SpanAddTypeForm';

const PATTERNS = {
    patterns: [{ type: 'ticket_id', pattern: '\\bTCK-\\d{4}\\b', description: '', flags: '' }],
    errors: [],
    builtin_types: ['email', 'phone'],
    limits: { max_patterns: 100, max_pattern_chars: 300, flags: 'ims' },
};

describe('SpanAddTypeForm', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        apiMock.put.mockReset();
        apiMock.get.mockImplementation(async (url: string) => (
            url.endsWith('/type-patterns')
                ? { data: PATTERNS }
                : { data: { definitions: { email: 'an email address' }, types: [], suggestions: {} } }
        ));
        apiMock.put.mockImplementation(async (_url: string, body: unknown) => ({ data: body }));
    });

    it('normalises type names like the backend', () => {
        expect(normaliseTypeName('  Employee ID ')).toBe('employee_id');
    });

    it('tests a pattern and shows the matches with provenance', async () => {
        apiMock.post.mockResolvedValue({ data: {
            valid: true, error: null, type: 'employee_id', texts_scanned: 12, texts_matched: 2, count: 3,
            matches: [{ text_index: 4, source: 'passage', start: 6, end: 15, match: 'EMP-00007', before: 'Staff ', after: ' and' }],
        } });
        const user = userEvent.setup();
        render(<SpanAddTypeForm projectId={7} />);
        await user.click(screen.getByTestId('span-add-type-toggle'));
        await waitFor(() => expect(screen.getByTestId('span-add-type-saved')).toHaveTextContent('ticket_id'));
        await user.type(screen.getByTestId('span-add-type-name'), 'Employee ID');
        await user.type(screen.getByTestId('span-add-type-pattern'), 'EMP-\\d{{5}');
        await user.click(screen.getByTestId('span-add-type-test'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/7/synthetic/type-patterns/test', {
            pattern: 'EMP-\\d{5}', type: 'employee_id', flags: '',
        }));
        const result = await screen.findByTestId('span-add-type-test-result');
        expect(result).toHaveTextContent('3 matches in 2 of 12 texts');
        expect(result).toHaveTextContent('passage #4, chars 6–15');
        expect(screen.getByText('EMP-00007').tagName).toBe('MARK');
    });

    it('blocks Add on a rejected pattern', async () => {
        apiMock.post.mockResolvedValue({ data: {
            valid: false, error: 'pattern matches the empty string', type: 'x', texts_scanned: 0, texts_matched: 0, count: 0, matches: [],
        } });
        const user = userEvent.setup();
        render(<SpanAddTypeForm projectId={7} />);
        await user.click(screen.getByTestId('span-add-type-toggle'));
        await user.type(screen.getByTestId('span-add-type-name'), 'x');
        await user.type(screen.getByTestId('span-add-type-pattern'), '\\d*');
        await user.click(screen.getByTestId('span-add-type-test'));
        expect(await screen.findByTestId('span-add-type-test-error')).toHaveTextContent('matches the empty string');
        expect(screen.getByTestId('span-add-type-save')).toBeDisabled();
    });

    it('adds the type: pattern appended, definition merged, type reported', async () => {
        const onTypeAdded = vi.fn();
        const user = userEvent.setup();
        render(<SpanAddTypeForm projectId={7} onTypeAdded={onTypeAdded} />);
        await user.click(screen.getByTestId('span-add-type-toggle'));
        await waitFor(() => expect(screen.getByTestId('span-add-type-saved')).toBeInTheDocument());
        await user.type(screen.getByTestId('span-add-type-name'), 'Employee ID');
        await user.type(screen.getByTestId('span-add-type-definition'), 'a staff id');
        await user.type(screen.getByTestId('span-add-type-pattern'), 'EMP-\\d{{5}');
        await user.click(screen.getByTestId('span-add-type-ignore-case'));
        await user.click(screen.getByTestId('span-add-type-save'));
        await waitFor(() => expect(onTypeAdded).toHaveBeenCalledWith('employee_id'));
        expect(apiMock.put).toHaveBeenCalledWith('/projects/7/synthetic/type-patterns', {
            patterns: [
                PATTERNS.patterns[0],
                { type: 'employee_id', pattern: 'EMP-\\d{5}', description: 'a staff id', flags: 'i' },
            ],
        });
        expect(apiMock.put).toHaveBeenCalledWith('/projects/7/synthetic/type-definitions', {
            definitions: { email: 'an email address', employee_id: 'a staff id' },
        });
        expect(screen.getByTestId('span-add-type-name')).toHaveValue('');
    });

    it('a type without a pattern saves only the definition', async () => {
        const onTypeAdded = vi.fn();
        const user = userEvent.setup();
        render(<SpanAddTypeForm projectId={7} onTypeAdded={onTypeAdded} />);
        await user.click(screen.getByTestId('span-add-type-toggle'));
        await user.type(screen.getByTestId('span-add-type-name'), 'diagnosis');
        await user.type(screen.getByTestId('span-add-type-definition'), 'a named condition');
        await user.click(screen.getByTestId('span-add-type-save'));
        await waitFor(() => expect(onTypeAdded).toHaveBeenCalledWith('diagnosis'));
        expect(apiMock.put).toHaveBeenCalledTimes(1);
        expect(apiMock.put.mock.calls[0][0]).toBe('/projects/7/synthetic/type-definitions');
    });

    it('removes a saved pattern', async () => {
        const user = userEvent.setup();
        render(<SpanAddTypeForm projectId={7} />);
        await user.click(screen.getByTestId('span-add-type-toggle'));
        await user.click(await screen.findByTestId('span-add-type-remove-0'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/7/synthetic/type-patterns', { patterns: [] }));
    });
});
