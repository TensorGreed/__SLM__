/**
 * PipelineMembership — "Part of pipelines: …" links on a member project's
 * Results page. Renders nothing when the project is in no pipeline.
 */

import { useEffect, useState } from 'react';

import { listProjectPipelines, pipelinePath } from '../../api/pipelines';
import './pipelines.css';

export default function PipelineMembership({ projectId }: { projectId: number }) {
    const [items, setItems] = useState<{ id: string; name: string }[]>([]);
    useEffect(() => {
        let live = true;
        listProjectPipelines(projectId)
            .then((p) => { if (live) setItems(p); })
            .catch(() => { if (live) setItems([]); });
        return () => { live = false; };
    }, [projectId]);
    if (items.length === 0) return null;
    return (
        <p className="pipelines__note" data-testid="pipeline-membership">
            Part of pipelines:{' '}
            {items.map((p, i) => (
                <span key={p.id}>
                    {i > 0 && ', '}
                    <a href={pipelinePath(p.id)}>{p.name}</a>
                </span>
            ))}
        </p>
    );
}
