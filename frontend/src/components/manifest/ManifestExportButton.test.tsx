import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import ManifestExportButton from './ManifestExportButton';

describe('ManifestExportButton', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('GETs /projects/:id/manifest/export?format=yaml and downloads a YAML file', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: 'api_version: brewslm/v1\nkind: Project\n',
        });

        const createObjectURL = vi.fn(() => 'blob://stub');
        const revokeObjectURL = vi.fn();
        const originalCreate = URL.createObjectURL;
        const originalRevoke = URL.revokeObjectURL;
        URL.createObjectURL = createObjectURL as typeof URL.createObjectURL;
        URL.revokeObjectURL = revokeObjectURL as typeof URL.revokeObjectURL;

        const clickSpy = vi.fn();
        const originalClick = HTMLAnchorElement.prototype.click;
        HTMLAnchorElement.prototype.click = clickSpy;

        try {
            const user = userEvent.setup();
            render(<ManifestExportButton projectId={7} projectName="Demo Project" />);
            await user.click(screen.getByRole('button', { name: /Export YAML/i }));

            await waitFor(() => {
                expect(apiMock.get).toHaveBeenCalledWith(
                    '/projects/7/manifest/export',
                    expect.objectContaining({ params: { format: 'yaml' } }),
                );
            });
            expect(createObjectURL).toHaveBeenCalled();
            expect(clickSpy).toHaveBeenCalled();
            // "Exported" confirmation surfaces.
            expect(await screen.findByRole('status')).toHaveTextContent(/Exported/);
        } finally {
            URL.createObjectURL = originalCreate;
            URL.revokeObjectURL = originalRevoke;
            HTMLAnchorElement.prototype.click = originalClick;
        }
    });

    it('shows an error if the export request fails', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { data: { detail: 'project_not_found' } },
        });

        const user = userEvent.setup();
        render(<ManifestExportButton projectId={9999} projectName="Nope" />);
        await user.click(screen.getByRole('button', { name: /Export YAML/i }));

        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('project_not_found');
    });
});
