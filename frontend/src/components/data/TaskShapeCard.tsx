/**
 * Labels step — the task shape (Epic U phase U4): what the rows look like
 * and therefore what gets labelled. Wraps TaskShapeConfirmCard with its own
 * fetch so it can live outside the dataset-prep panel.
 */

import { useCallback, useEffect, useState } from 'react';

import { fetchProjectTaskShape, type ProjectTaskShape } from '../../api/taskShape';
import TaskShapeConfirmCard from './TaskShapeConfirmCard';

interface TaskShapeCardProps {
    projectId: number;
}

export default function TaskShapeCard({ projectId }: TaskShapeCardProps) {
    const [taskShape, setTaskShape] = useState<ProjectTaskShape | null>(null);
    const [loaded, setLoaded] = useState(false);

    const load = useCallback(async () => {
        try {
            const res = await fetchProjectTaskShape(projectId);
            setTaskShape(res && res.detection ? res : null);
        } catch {
            setTaskShape(null);
        } finally {
            setLoaded(true);
        }
    }, [projectId]);

    useEffect(() => {
        void load();
    }, [load]);

    if (!loaded) return null;
    if (!taskShape?.detection) {
        return (
            <section className="card" data-testid="task-shape-card-empty">
                <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 600, marginBottom: 'var(--space-sm)' }}>Task shape</h3>
                <p className="form-hint">Bring data in first; the task shape is detected from the rows.</p>
            </section>
        );
    }
    return (
        <TaskShapeConfirmCard
            key={taskShape.confirmed?.task_profile || 'unconfirmed'}
            projectId={projectId}
            detection={taskShape.detection}
            catalog={taskShape.catalog}
            confirmedLabel={taskShape.confirmed?.label ?? null}
            onConfirmed={() => { void load(); }}
        />
    );
}
