import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import StructuredPlanConfigCard from './StructuredPlanConfigCard';

const EMPTY = {
    config: {},
    effective: { judge_enabled: true, retrieve_passages: false, top_k: 3, pointers: [] },
    schema: null,
    rubric: null,
    guided_json_present: false,
};

const SAVED = {
    config: {
        schema_files: { 'plan.schema.json': { type: 'object' }, 'step.schema.json': { type: 'object' } },
        root_schema: 'plan.schema.json',
        rubric_yaml: 'id: plan-quality\ncriteria: []',
        pointers: ['/environment', '/steps/*/action'],
        judge_enabled: true,
        retrieve_passages: false,
        top_k: 3,
    },
    effective: { judge_enabled: true, retrieve_passages: false, top_k: 3, pointers: ['/environment', '/steps/*/action'] },
    schema: { root: 'plan.schema.json', files: ['plan.schema.json', 'step.schema.json'], sha256: 'abcdef0123456789', valid: true, errors: [] },
    rubric: {
        id: 'plan-quality', title: 'Plan quality', sha256: '1234567890abcdef',
        criteria: [
            { id: 'safe_order', description: 'ordering', weight: 2, critical: true },
            { id: 'clarity', description: 'clear', weight: 1, critical: false },
        ],
    },
    guided_json_present: true,
};

function jsonFile(name: string, body: unknown) {
    return new File([JSON.stringify(body)], name, { type: 'application/json' });
}

describe('StructuredPlanConfigCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
        apiMock.post.mockReset();
    });

    it('shows the saved schema and rubric with short shas', async () => {
        apiMock.get.mockResolvedValue({ data: SAVED });
        render(<StructuredPlanConfigCard projectId={4} />);
        expect(await screen.findByTestId('sp-schema-provenance')).toHaveTextContent('plan.schema.json');
        expect(screen.getByTestId('sp-schema-provenance')).toHaveTextContent('sha abcdef01');
        expect(screen.getByTestId('sp-rubric-provenance')).toHaveTextContent('plan-quality');
        expect(screen.getByTestId('sp-rubric-provenance')).toHaveTextContent('sha 12345678');
        expect(screen.getByTestId('sp-criterion-safe_order')).toHaveTextContent('critical');
        expect(screen.getByTestId('sp-pointers')).toHaveValue('/environment\n/steps/*/action');
        expect(screen.getByTestId('sp-guided')).toBeInTheDocument();
    });

    it('uploads schema files, picks a root and saves them', async () => {
        apiMock.get.mockResolvedValue({ data: EMPTY });
        apiMock.put.mockResolvedValue({ data: SAVED });
        const user = userEvent.setup();
        render(<StructuredPlanConfigCard projectId={4} />);
        const input = await screen.findByTestId('sp-schema-upload');
        await user.upload(input, [jsonFile('plan.schema.json', { type: 'object' }), jsonFile('step.schema.json', { type: 'object' })]);
        await screen.findByTestId('sp-file-step.schema.json');
        expect(screen.getByTestId('sp-root-select')).toHaveValue('plan.schema.json');
        await user.type(screen.getByTestId('sp-pointers'), '/environment');
        await user.click(screen.getByTestId('sp-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        const [url, body] = apiMock.put.mock.calls[0];
        expect(url).toBe('/projects/4/structured-plan-config');
        expect(body.root_schema).toBe('plan.schema.json');
        expect(Object.keys(body.schema_files)).toEqual(['plan.schema.json', 'step.schema.json']);
        expect(body.pointers).toEqual(['/environment']);
        expect(body.judge_enabled).toBe(true);
        expect('retrieve_passages' in body).toBe(false);
    });

    it('sends retrieve_passages only after the user toggles it', async () => {
        apiMock.get.mockResolvedValue({ data: SAVED });
        apiMock.put.mockResolvedValue({ data: SAVED });
        const user = userEvent.setup();
        render(<StructuredPlanConfigCard projectId={4} />);
        await user.click(await screen.findByTestId('sp-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledTimes(1));
        expect('retrieve_passages' in apiMock.put.mock.calls[0][1]).toBe(false);
        await user.click(await screen.findByTestId('sp-retrieve-toggle'));
        await user.click(screen.getByTestId('sp-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledTimes(2));
        expect(apiMock.put.mock.calls[1][1].retrieve_passages).toBe(true);
    });

    it('rejects a non-JSON schema file locally', async () => {
        apiMock.get.mockResolvedValue({ data: EMPTY });
        const user = userEvent.setup();
        render(<StructuredPlanConfigCard projectId={4} />);
        await user.upload(await screen.findByTestId('sp-schema-upload'), new File(['not json'], 'bad.json', { type: 'application/json' }));
        expect(await screen.findByTestId('sp-upload-error')).toHaveTextContent('bad.json is not valid JSON');
    });

    it('shows the 422 errors from metadata.errors', async () => {
        apiMock.get.mockResolvedValue({ data: SAVED });
        apiMock.put.mockRejectedValue({
            response: { status: 422, data: { error_code: 'INVALID_STRUCTURED_PLAN_CONFIG', detail: 'invalid', metadata: { errors: ['step.schema.json: unresolved $ref', 'pointer "x" must start with /'] } } },
        });
        const user = userEvent.setup();
        render(<StructuredPlanConfigCard projectId={4} />);
        await user.click(await screen.findByTestId('sp-save'));
        const list = await screen.findByTestId('sp-save-errors');
        expect(list).toHaveTextContent('unresolved $ref');
        expect(list).toHaveTextContent('must start with /');
    });

    it('validates a pasted plan against the saved schema', async () => {
        apiMock.get.mockResolvedValue({ data: SAVED });
        apiMock.post.mockResolvedValue({ data: { valid: false, errors: [{ pointer: '/steps/0/action', message: "'x' is not one of ['create']", validator: 'enum' }] } });
        render(<StructuredPlanConfigCard projectId={4} />);
        const box = await screen.findByTestId('sp-paste');
        fireEvent.change(box, { target: { value: '{"steps":[{"action":"x"}]}' } });
        await userEvent.click(screen.getByTestId('sp-check'));
        expect(await screen.findByTestId('sp-check-errors')).toHaveTextContent('/steps/0/action');
        expect(apiMock.post).toHaveBeenCalledWith('/projects/4/structured-plan/validate', { value: { steps: [{ action: 'x' }] } });
    });

    it('reports pasted text that is not JSON without calling the server', async () => {
        apiMock.get.mockResolvedValue({ data: SAVED });
        render(<StructuredPlanConfigCard projectId={4} />);
        fireEvent.change(await screen.findByTestId('sp-paste'), { target: { value: '{oops' } });
        await userEvent.click(screen.getByTestId('sp-check'));
        expect(screen.getByTestId('sp-check-error')).toHaveTextContent('not valid JSON');
        expect(apiMock.post).not.toHaveBeenCalled();
    });
});
