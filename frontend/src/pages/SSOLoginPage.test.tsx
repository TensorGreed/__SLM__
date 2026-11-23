import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));
vi.mock('../api/client', () => ({ default: apiMock }));

import SSOLoginPage from './SSOLoginPage';

describe('SSOLoginPage', () => {
    beforeEach(() => {
        apiMock.get.mockResolvedValue({ data: { auth_enabled: true, sso_enabled: false } });
    });

    it('shows the product mark with an accessible name, not the BS text logo', () => {
        const { container } = render(
            <MemoryRouter>
                <SSOLoginPage />
            </MemoryRouter>,
        );
        expect(screen.getByRole('img', { name: 'BrewSLM logo' })).toBeTruthy();
        expect(container.querySelector('.auth-logo svg')).not.toBeNull();
        expect(container.querySelector('.auth-logo span')).toBeNull();
    });
});
