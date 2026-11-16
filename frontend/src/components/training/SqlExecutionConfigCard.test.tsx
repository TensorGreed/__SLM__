import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import SqlExecutionConfigCard from './SqlExecutionConfigCard';

const EFFECTIVE = { enabled: false, db_paths: [], timeout_ms: 2000, row_cap: 10000, float_rel_tol: 1e-6, order: 'auto', refusal_marker: 'NO_QUERY', active: false };
const OFF = { config: {}, effective: EFFECTIVE, databases: [], data_dir: '/data/projects/3' };
const ON = {
    config: { enabled: true, db_paths: ['dbs/main.db'] },
    effective: { ...EFFECTIVE, enabled: true, db_paths: ['dbs/main.db'], active: true },
    databases: [{ path: 'dbs/main.db', ok: true, error: null, size_bytes: 8192 }],
    data_dir: '/data/projects/3',
};

describe('SqlExecutionConfigCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.put.mockReset();
        apiMock.get.mockResolvedValue({ data: OFF });
        apiMock.put.mockResolvedValue({ data: ON });
    });

    it('says it is off by default and where databases must live', async () => {
        render(<SqlExecutionConfigCard projectId={3} />);
        expect(await screen.findByTestId('sx-status-line')).toHaveTextContent('Off: evals score answers as before.');
        expect(screen.getByTestId('sx-status-line')).toHaveTextContent('/data/projects/3');
        expect(screen.getByTestId('sx-enabled')).not.toBeChecked();
    });

    it('saves the databases and settings', async () => {
        const user = userEvent.setup();
        render(<SqlExecutionConfigCard projectId={3} />);
        await screen.findByTestId('sx-config');
        await user.click(screen.getByTestId('sx-enabled'));
        await user.type(screen.getByTestId('sx-paths'), 'dbs/main.db');
        await user.selectOptions(screen.getByTestId('sx-order'), 'never');
        await user.click(screen.getByTestId('sx-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        const [url, body] = apiMock.put.mock.calls[0];
        expect(url).toBe('/projects/3/sql-execution-config');
        expect(body).toMatchObject({ enabled: true, db_paths: ['dbs/main.db'], order: 'never', timeout_ms: 2000, row_cap: 10000, refusal_marker: 'NO_QUERY' });
        expect(await screen.findByTestId('sx-databases')).toHaveTextContent('dbs/main.db: found (8192 bytes)');
    });

    it('shows the validation errors from the server', async () => {
        apiMock.put.mockRejectedValue({ response: { data: { error_code: 'INVALID_SQL_EXECUTION_CONFIG', metadata: { errors: ["database path '/etc/x' is outside the project data directory"] } } } });
        const user = userEvent.setup();
        render(<SqlExecutionConfigCard projectId={3} />);
        await screen.findByTestId('sx-config');
        await user.click(screen.getByTestId('sx-save'));
        expect(await screen.findByTestId('sx-save-errors')).toHaveTextContent('outside the project data directory');
    });
});
