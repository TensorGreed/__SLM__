import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import OutputSchemaEditor from './OutputSchemaEditor';

const SCHEMA = '{"properties": {"damage": {"type": ["string", "null"], "enum": ["minor", null]}}}';

describe('OutputSchemaEditor', () => {
    beforeEach(() => {
        apiMock.put.mockReset();
    });

    it('saves a valid schema to the output-schema endpoint', async () => {
        apiMock.put.mockResolvedValue({ data: {} });
        const onSaved = vi.fn();
        render(<OutputSchemaEditor projectId={7} onSaved={onSaved} />);
        const box = screen.getByTestId('output-schema-text');
        await userEvent.click(box);
        await userEvent.paste(SCHEMA);
        await userEvent.click(screen.getByTestId('output-schema-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledTimes(1));
        expect(apiMock.put).toHaveBeenCalledWith('/projects/7/recipe/output-schema', {
            output_schema: JSON.parse(SCHEMA),
        });
        expect(onSaved).toHaveBeenCalled();
        expect(await screen.findByTestId('output-schema-saved')).toBeInTheDocument();
    });

    it('shows an inline error for text that is not JSON and does not call the API', async () => {
        render(<OutputSchemaEditor projectId={7} />);
        await userEvent.click(screen.getByTestId('output-schema-text'));
        await userEvent.paste('{not json');
        await userEvent.click(screen.getByTestId('output-schema-save'));
        expect(await screen.findByTestId('output-schema-error')).toHaveTextContent(/Not valid JSON/);
        expect(apiMock.put).not.toHaveBeenCalled();
    });

    it('rejects a JSON array before calling the API', async () => {
        render(<OutputSchemaEditor projectId={7} />);
        await userEvent.click(screen.getByTestId('output-schema-text'));
        await userEvent.paste('[1, 2]');
        await userEvent.click(screen.getByTestId('output-schema-save'));
        expect(await screen.findByTestId('output-schema-error')).toHaveTextContent(/JSON object/);
        expect(apiMock.put).not.toHaveBeenCalled();
    });

    it('surfaces the server validation message', async () => {
        apiMock.put.mockRejectedValue({
            response: { data: { detail: 'Invalid output schema — properties.a.type: unsupported type' } },
        });
        render(<OutputSchemaEditor projectId={7} />);
        await userEvent.click(screen.getByTestId('output-schema-text'));
        await userEvent.paste(SCHEMA);
        await userEvent.click(screen.getByTestId('output-schema-save'));
        expect(await screen.findByTestId('output-schema-error')).toHaveTextContent(/unsupported type/);
    });

    it('clears the schema when the box is emptied', async () => {
        apiMock.put.mockResolvedValue({ data: {} });
        render(<OutputSchemaEditor projectId={7} initialSchema={JSON.parse(SCHEMA)} />);
        await userEvent.clear(screen.getByTestId('output-schema-text'));
        await userEvent.click(screen.getByTestId('output-schema-save'));
        await waitFor(() =>
            expect(apiMock.put).toHaveBeenCalledWith('/projects/7/recipe/output-schema', { output_schema: null }),
        );
    });
});
