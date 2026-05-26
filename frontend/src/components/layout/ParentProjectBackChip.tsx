/**
 * ParentProjectBackChip — USER-SUCCESS Epic 7 Phase 7d.
 *
 * Provenance chip rendered in the ProjectWorkspaceLayout TopBar
 * actions slot when the current project has a ``parent_project_id``
 * set (today: only RAG-clone projects produced by Phase 7b's
 * ``reroute_to_rag`` endpoint).
 *
 * Fetches the parent project's name once on mount via a lightweight
 * GET; until the fetch resolves the chip shows ``← cloned``. On
 * click it deep-links to the parent project's pipeline so the user
 * can switch between SFT-trained and RAG-first siblings.
 */

import { useEffect, useState } from 'react';
import api from '../../api/client';
import type { Project } from '../../types';
import './ParentProjectBackChip.css';


interface Props {
    parentProjectId: number;
}


export default function ParentProjectBackChip({ parentProjectId }: Props) {
    const [parentName, setParentName] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        api.get<Project>(`/projects/${parentProjectId}`)
            .then((resp) => {
                if (!cancelled) setParentName(resp.data.name);
            })
            .catch(() => {
                // Parent might have been deleted — fall back to id-only chip.
                if (!cancelled) setParentName(null);
            });
        return () => {
            cancelled = true;
        };
    }, [parentProjectId]);

    const label = parentName
        ? `← cloned from ${parentName}`
        : `← cloned from project #${parentProjectId}`;

    return (
        <a
            href={`/project/${parentProjectId}`}
            className="parent-project-chip"
            data-testid="parent-project-chip"
            title="Open the source project this RAG sibling was cloned from"
        >
            {label}
        </a>
    );
}
