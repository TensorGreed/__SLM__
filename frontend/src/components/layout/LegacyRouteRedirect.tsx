/**
 * Redirects a retired workspace URL (``/guide``, ``/wizard``,
 * ``/pipeline/<tab>``) to where it lives now (Epic U phase U1). The search
 * string and hash travel along, so ``?import=auto`` and ``#anchor`` deep
 * links keep working.
 */

import { Navigate, useLocation, useParams } from 'react-router-dom';

import { legacyRedirectPath, projectPath } from '../../utils/workspaceRoutes';

export default function LegacyRouteRedirect() {
    const { id } = useParams<{ id: string }>();
    const location = useLocation();
    const projectId = Number.parseInt(id || '', 10);
    if (!Number.isFinite(projectId)) {
        return <Navigate to="/" replace />;
    }
    const target = legacyRedirectPath(projectId, location.pathname, location.search, location.hash);
    return <Navigate to={target ?? projectPath(projectId)} replace />;
}
