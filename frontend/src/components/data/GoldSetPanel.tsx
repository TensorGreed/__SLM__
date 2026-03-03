import { useState, useEffect, useCallback } from 'react';
import api from '../../api/client';
import StepFooter from '../shared/StepFooter';
import './GoldSetPanel.css';

interface GoldSetPanelProps {
    projectId: number;
    onNextStep?: () => void;
}

export default function GoldSetPanel({ projectId, onNextStep }: GoldSetPanelProps) {
    const [entries, setEntries] = useState<any[]>([]);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [difficulty, setDifficulty] = useState('medium');
    const [isHallucTrap, setIsHallucTrap] = useState(false);
    const [datasetType, setDatasetType] = useState('gold_dev');
    const [loaded, setLoaded] = useState(false);

    const fetchEntries = useCallback(async () => {
        const res = await api.get(`/projects/${projectId}/gold/entries?dataset_type=${datasetType}`);
        setEntries(res.data.entries || []);
        setLoaded(true);
    }, [projectId, datasetType]);

    useEffect(() => { fetchEntries(); }, [fetchEntries]);

    const handleAdd = async () => {
        if (!question.trim() || !answer.trim()) return;
        await api.post(`/projects/${projectId}/gold/add`, {
            question, answer, dataset_type: datasetType, difficulty, is_hallucination_trap: isHallucTrap,
        });
        setQuestion(''); setAnswer('');
        fetchEntries();
    };

    const handleLock = async () => {
        if (!confirm('Lock this dataset? No more entries can be added.')) return;
        await api.post(`/projects/${projectId}/gold/lock?dataset_type=${datasetType}`);
    };

    return (
        <div className="gold-panel animate-fade-in">
            <div className="card">
                <div className="gold-header">
                    <h3>Gold Evaluation Dataset</h3>
                    <div className="gold-controls">
                        <select className="input" value={datasetType} onChange={e => setDatasetType(e.target.value)} style={{ width: 'auto' }}>
                            <option value="gold_dev">Dev Set</option>
                            <option value="gold_test">Test Set</option>
                        </select>
                        <button className="btn btn-secondary" onClick={handleLock}>🔒 Lock</button>
                    </div>
                </div>

                <div className="qa-form">
                    <div className="form-group">
                        <label className="form-label">Question</label>
                        <input className="input" placeholder="Enter a question..." value={question} onChange={e => setQuestion(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Expected Answer</label>
                        <textarea className="input gold-textarea" placeholder="Expected answer..." value={answer} onChange={e => setAnswer(e.target.value)} />
                    </div>
                    <div className="form-row">
                        <select className="input" value={difficulty} onChange={e => setDifficulty(e.target.value)} style={{ width: 'auto' }}>
                            <option value="easy">Easy</option>
                            <option value="medium">Medium</option>
                            <option value="hard">Hard</option>
                        </select>
                        <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <input type="checkbox" checked={isHallucTrap} onChange={e => setIsHallucTrap(e.target.checked)} />
                            Hallucination Trap
                        </label>
                        <button className="btn btn-primary" onClick={handleAdd}>+ Add Pair</button>
                    </div>
                </div>
            </div>

            <div className="card">
                <h3>Entries <span className="badge badge-accent">{entries.length}</span></h3>
                <div className="entries-list">
                    {entries.map((e, i) => (
                        <div key={i} className="entry-item">
                            <div className="entry-q"><strong>Q:</strong> {e.question}</div>
                            <div className="entry-a"><strong>A:</strong> {e.answer}</div>
                            <div className="entry-meta">
                                <span className="badge badge-info">{e.difficulty}</span>
                                {e.is_hallucination_trap && <span className="badge badge-warning">Trap</span>}
                            </div>
                        </div>
                    ))}
                    {entries.length === 0 && <div className="empty-state"><div className="empty-state-text">No entries yet. Add Q&A pairs above.</div></div>}
                </div>
            </div>

            {onNextStep && (
                <StepFooter
                    currentStep="Gold Dataset"
                    nextStep="Synthetic Generation"
                    nextStepIcon="🧪"
                    isComplete={entries.length >= 5}
                    hint="Add at least 5 Q&A pairs for evaluation"
                    onNext={onNextStep}
                />
            )}
        </div>
    );
}
