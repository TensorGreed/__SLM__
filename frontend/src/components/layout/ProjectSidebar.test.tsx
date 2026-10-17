import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';

import ProjectSidebar from './ProjectSidebar';

function LocationProbe() {
    const location = useLocation();
    return <div data-testid="location">{location.pathname}</div>;
}

function renderSidebar(initialPath: string) {
    return render(
        <MemoryRouter initialEntries={[initialPath]}>
            <Routes>
                <Route
                    path="/project/:id/*"
                    element={
                        <>
                            <ProjectSidebar projectId={1} projectName="Demo" />
                            <LocationProbe />
                        </>
                    }
                />
            </Routes>
        </MemoryRouter>,
    );
}

describe('ProjectSidebar (six destinations)', () => {
    it('lists exactly Overview, Data, Train, Results, Ship and Settings', () => {
        renderSidebar('/project/1');
        const nav = screen.getByRole('navigation', { name: 'Workspace' });
        const labels = Array.from(nav.querySelectorAll('a')).map((a) => a.textContent?.trim());
        expect(labels).toEqual(['Overview', 'Data', 'Train', 'Results', 'Ship', 'Settings']);
    });

    it.each([
        ['/project/1', 'nav-overview'],
        ['/project/1/data?step=cleaning', 'nav-data'],
        ['/project/1/train', 'nav-train'],
        ['/project/1/results', 'nav-results'],
        ['/project/1/ship', 'nav-ship'],
        ['/project/1/settings', 'nav-settings'],
    ])('marks the destination of %s active', (path, testId) => {
        renderSidebar(path);
        expect(screen.getByTestId(testId)).toHaveClass('active');
        expect(screen.getAllByRole('link').filter((a) => a.classList.contains('active'))).toHaveLength(1);
    });

    it('does not mark Overview active on a stage page', () => {
        renderSidebar('/project/1/train');
        expect(screen.getByTestId('nav-overview')).not.toHaveClass('active');
    });

    it('navigates to a destination on click', async () => {
        const user = userEvent.setup();
        renderSidebar('/project/1');
        await user.click(screen.getByTestId('nav-ship'));
        expect(screen.getByTestId('location')).toHaveTextContent('/project/1/ship');
    });

    it('has no beginner-mode block and no stage links', () => {
        renderSidebar('/project/1/data');
        expect(screen.queryByText(/beginner mode/i)).toBeNull();
        expect(screen.queryByRole('link', { name: 'Cleaning' })).toBeNull();
        expect(screen.queryByRole('button', { name: 'Pipeline' })).toBeNull();
    });
});
