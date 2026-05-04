import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { DeploymentVersion } from '../../types/deployment';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DeployedVersionsList from './DeployedVersionsList';

function makeVersion(overrides: Partial<DeploymentVersion>): DeploymentVersion {
    return {
        id: 1,
        project_id: 7,
        export_id: 11,
        registry_entry_id: null,
        version: 1,
        target_id: 'sdk.apple_coreml_stub',
        target_kind: 'sdk',
        endpoint_name: null,
        endpoint_handle: null,
        region: null,
        instance_type: null,
        status: 'pending',
        plan_payload: {},
        promoted_reason: null,
        rejected_reason: null,
        rolled_back_reason: null,
        rolled_back_to_id: null,
        actor: 'system',
        created_at: '2026-05-04T00:00:00Z',
        promoted_at: null,
        rejected_at: null,
        rolled_back_at: null,
        superseded_at: null,
        ...overrides,
    };
}

beforeEach(() => {
    apiMock.post.mockReset();
});

describe('DeployedVersionsList', () => {
    it('renders status badges and version numbers', () => {
        const versions = [
            makeVersion({ id: 10, version: 2, status: 'promoted' }),
            makeVersion({ id: 11, version: 3, status: 'pending' }),
        ];
        const onSelect = vi.fn();
        const onRefresh = vi.fn();
        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={10}
                onSelect={onSelect}
                onRefresh={onRefresh}
            />,
        );
        expect(screen.getByText(/v2/)).toBeInTheDocument();
        expect(screen.getByText(/v3/)).toBeInTheDocument();
        expect(screen.getByText('promoted')).toBeInTheDocument();
        expect(screen.getByText('pending')).toBeInTheDocument();
    });

    it('shows empty state when no versions', () => {
        render(
            <DeployedVersionsList
                versions={[]}
                selectedDeploymentId={null}
                onSelect={vi.fn()}
                onRefresh={vi.fn()}
            />,
        );
        expect(screen.getByText(/No deployment versions yet/i)).toBeInTheDocument();
    });

    it('PENDING row exposes Promote and Reject; promote POSTs and refreshes', async () => {
        const versions = [makeVersion({ id: 5, version: 1, status: 'pending' })];
        const onRefresh = vi.fn().mockResolvedValue(undefined);
        apiMock.post.mockResolvedValueOnce({ data: {} });
        const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('ready for prod');

        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={5}
                onSelect={vi.fn()}
                onRefresh={onRefresh}
            />,
        );

        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Promote/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/deployments/5/promote', {
                reason: 'ready for prod',
            });
        });
        expect(onRefresh).toHaveBeenCalled();
        promptSpy.mockRestore();
    });

    it('cancelling the prompt aborts the action', async () => {
        const versions = [makeVersion({ id: 6, version: 1, status: 'pending' })];
        const onRefresh = vi.fn();
        const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue(null);

        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={6}
                onSelect={vi.fn()}
                onRefresh={onRefresh}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Reject/i }));

        expect(apiMock.post).not.toHaveBeenCalled();
        expect(onRefresh).not.toHaveBeenCalled();
        promptSpy.mockRestore();
    });

    it('PROMOTED row only exposes Rollback; surfaces stable error code', async () => {
        // The PROMOTED v4 has a SUPERSEDED v3 predecessor on the same
        // (export_id, target_id) so the client-side preflight passes
        // and the API call goes out.
        const versions = [
            makeVersion({ id: 8, version: 3, status: 'superseded' }),
            makeVersion({ id: 9, version: 4, status: 'promoted' }),
        ];
        const onRefresh = vi.fn();
        apiMock.post.mockRejectedValueOnce({
            response: { status: 409, data: { detail: 'no_promoted_predecessor' } },
        });
        vi.spyOn(window, 'prompt').mockReturnValue('regression');

        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={9}
                onSelect={vi.fn()}
                onRefresh={onRefresh}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Rollback/i }));
        expect(await screen.findByRole('alert')).toHaveTextContent('no_promoted_predecessor');
        expect(onRefresh).not.toHaveBeenCalled();
    });

    it('Rollback prompt surfaces the predecessor version it will re-promote', async () => {
        const versions = [
            makeVersion({ id: 8, version: 3, status: 'superseded' }),
            makeVersion({ id: 9, version: 4, status: 'promoted' }),
        ];
        const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('rollback');
        apiMock.post.mockResolvedValueOnce({ data: {} });

        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={9}
                onSelect={vi.fn()}
                onRefresh={vi.fn().mockResolvedValue(undefined)}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Rollback/i }));

        // The prompt should mention the destination version explicitly.
        expect(promptSpy).toHaveBeenCalledTimes(1);
        const promptText = promptSpy.mock.calls[0][0] as string;
        expect(promptText).toMatch(/Roll back v4/);
        expect(promptText).toMatch(/v3 \(#8\)/);
        promptSpy.mockRestore();
    });

    it('Rollback short-circuits client-side when no predecessor exists', async () => {
        // Only the promoted row, no SUPERSEDED sibling.
        const versions = [
            makeVersion({ id: 9, version: 1, status: 'promoted' }),
        ];
        const promptSpy = vi.spyOn(window, 'prompt');
        const onRefresh = vi.fn();

        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={9}
                onSelect={vi.fn()}
                onRefresh={onRefresh}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Rollback/i }));

        // No prompt opened, no API call, error rendered inline.
        expect(promptSpy).not.toHaveBeenCalled();
        expect(apiMock.post).not.toHaveBeenCalled();
        expect(await screen.findByRole('alert')).toHaveTextContent(
            /no superseded predecessor/i,
        );
        expect(onRefresh).not.toHaveBeenCalled();
        promptSpy.mockRestore();
    });

    it('REJECTED / ROLLED_BACK / SUPERSEDED rows have no action buttons', () => {
        const versions = [
            makeVersion({ id: 1, version: 1, status: 'rejected' }),
            makeVersion({ id: 2, version: 2, status: 'rolled_back' }),
            makeVersion({ id: 3, version: 3, status: 'superseded' }),
        ];
        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={1}
                onSelect={vi.fn()}
                onRefresh={vi.fn()}
            />,
        );
        expect(screen.queryByRole('button', { name: /Promote/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Reject/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Rollback/i })).not.toBeInTheDocument();
    });

    it('clicking the version button calls onSelect', async () => {
        const versions = [makeVersion({ id: 42, version: 7, status: 'pending' })];
        const onSelect = vi.fn();
        render(
            <DeployedVersionsList
                versions={versions}
                selectedDeploymentId={null}
                onSelect={onSelect}
                onRefresh={vi.fn()}
            />,
        );
        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Select deployment version 7/i }));
        expect(onSelect).toHaveBeenCalledWith(42);
    });
});
