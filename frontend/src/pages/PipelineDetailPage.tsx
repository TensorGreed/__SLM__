/**
 * Pipeline detail — editor, held-out upload, runs and the selected run's
 * results. `/pipelines/new` creates; `/pipelines/:id?run=<run_id>` opens a run.
 */

import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';

import api from '../api/client';
import {
    createPipeline, deletePipeline, getPipeline, getPipelineRun, isFinishedRun, parsePipelineError, runStatusLabel,
    updatePipeline, type FieldError, type PipelineRun,
} from '../api/pipelines';
import TopBar from '../components/layout/TopBar';
import HeldoutCard from '../components/pipelines/HeldoutCard';
import PipelineEditor, {
    draftFromPipeline, draftToDefinition, emptyDraft, type PipelineDraft, type ProjectOption,
} from '../components/pipelines/PipelineEditor';
import PipelineResults from '../components/pipelines/PipelineResults';
import RunsCard from '../components/pipelines/RunsCard';
import { toast } from '../stores/toastStore';
import '../components/pipelines/pipelines.css';

/** A run without results yet (running) or without results at all (failed / cancelled / stale). */
function RunStatusPanel({ run }: { run: PipelineRun }) {
    const status = run.status ?? 'done';
    const text = status === 'running'
        ? 'This run is still going: progress and the outcome appear in the notification bell. Reload the run when it finishes.'
        : status === 'done'
            ? 'This run finished but recorded no per-stage results.'
            : `This run ${runStatusLabel(status)} and has no results.`;
    return (
        <section className="pipelines__section" data-testid="run-status">
            <h3>Run {run.run_id} · {runStatusLabel(status)}</h3>
            <p className={status === 'running' ? 'pipelines__note' : 'pipelines__warn'}>{text}</p>
            {run.error && <p className="pipelines__error" role="alert" data-testid="run-status-error">{run.error}</p>}
            {run.started_at && <p className="pipelines__note">Started {run.started_at}{run.finished_at ? ` · ended ${run.finished_at}` : ''}</p>}
        </section>
    );
}

export default function PipelineDetailPage() {
    const { id = 'new' } = useParams();
    const isNew = id === 'new';
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const runId = searchParams.get('run');

    const [draft, setDraft] = useState<PipelineDraft>(emptyDraft());
    const [projects, setProjects] = useState<ProjectOption[]>([]);
    const [errors, setErrors] = useState<FieldError[]>([]);
    const [message, setMessage] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);
    const [run, setRun] = useState<PipelineRun | null>(null);
    const [runError, setRunError] = useState<string | null>(null);
    const [runsKey, setRunsKey] = useState(0);

    useEffect(() => {
        api.get<{ projects: { id: number; name: string }[] }>('/projects')
            .then((r) => setProjects((r.data.projects ?? []).map((p) => ({ id: p.id, name: p.name }))))
            .catch(() => setProjects([]));
    }, []);

    useEffect(() => {
        if (isNew) { setDraft(emptyDraft()); return; }
        getPipeline(id).then((p) => setDraft(draftFromPipeline(p))).catch((e) => setMessage(parsePipelineError(e).message));
    }, [id, isNew]);

    useEffect(() => {
        setRun(null);
        setRunError(null);
        if (isNew || !runId) return;
        getPipelineRun(id, runId).then(setRun).catch((e) => setRunError(parsePipelineError(e).message));
    }, [id, isNew, runId]);

    const save = async () => {
        setSaving(true);
        setErrors([]);
        setMessage(null);
        try {
            const body = draftToDefinition(draft);
            if (isNew) {
                const created = await createPipeline(body);
                toast.success('Pipeline created');
                navigate(`/pipelines/${created.id}`);
            } else {
                setDraft(draftFromPipeline(await updatePipeline(id, body)));
                toast.success('Pipeline saved');
            }
        } catch (e) {
            const parsed = parsePipelineError(e);
            setMessage(parsed.message);
            setErrors(parsed.errors);
        } finally {
            setSaving(false);
        }
    };

    const remove = async () => {
        if (!window.confirm('Delete this pipeline, its held-out rows and runs?')) return;
        try {
            await deletePipeline(id);
            navigate('/pipelines');
        } catch (e) {
            setMessage(parsePipelineError(e).message);
        }
    };

    return (
        <div className="main-content">
            <TopBar title={isNew ? 'New pipeline' : draft.name || 'Pipeline'} subtitle="Pipelines" />
            <div className="page-container" data-testid="pipeline-detail">
                <div className="pipelines__page-head">
                    <Link to="/pipelines" className="pipelines__back">All pipelines</Link>
                    {!isNew && <button type="button" className="btn btn-ghost" data-testid="pipeline-delete" onClick={() => void remove()}>Delete pipeline</button>}
                </div>
                <PipelineEditor draft={draft} projects={projects} errors={errors} message={message} saving={saving}
                    onChange={setDraft} onSave={() => void save()} />
                {!isNew && (
                    <>
                        <HeldoutCard pipelineId={id} />
                        <RunsCard pipelineId={id} selectedRun={runId} refreshKey={runsKey}
                            onSelect={(r) => { setSearchParams({ run: r }); setRunsKey((k) => k + 1); }} />
                        {runError && <p className="pipelines__error" role="alert">{runError}</p>}
                        {run && (isFinishedRun(run) ? <PipelineResults run={run} /> : <RunStatusPanel run={run} />)}
                    </>
                )}
            </div>
        </div>
    );
}
