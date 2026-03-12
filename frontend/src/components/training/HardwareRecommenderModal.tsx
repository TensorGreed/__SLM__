import { useEffect, useState } from 'react';
import api from '../../api/client';

export interface HardwareProfile {
    id: string;
    name: string;
    description: string;
    icon: string;
    max_vram_gb: number;
}

export interface RecommendationResult {
    base_model: string;
    compression_bits: number;
    lora_rank: number;
    training_batch_size: number;
    notes: string[];
}

interface HardwareRecommenderModalProps {
    onClose: () => void;
    onApply: (rec: RecommendationResult) => void;
}

export default function HardwareRecommenderModal({ onClose, onApply }: HardwareRecommenderModalProps) {
    const [profiles, setProfiles] = useState<HardwareProfile[]>([]);
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [recommendation, setRecommendation] = useState<RecommendationResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [taskType, setTaskType] = useState('causal_lm');

    useEffect(() => {
        api.get('/hardware/catalog').then((res) => {
            setProfiles(res.data.profiles || []);
            if (res.data.profiles?.length > 0) {
                setSelectedId(res.data.profiles[0].id);
            }
        }).catch(console.error);
    }, []);

    useEffect(() => {
        if (!selectedId) return;
        setLoading(true);
        api.post('/hardware/recommend', {
            hardware_id: selectedId,
            task_type: taskType
        }).then((res) => {
            setRecommendation(res.data);
        }).catch(console.error).finally(() => {
            setLoading(false);
        });
    }, [selectedId, taskType]);

    const handleApply = () => {
        if (recommendation) {
            onApply(recommendation);
        }
    };

    const getIconSrc = (icon: string) => {
        // Simplified icon mapping, could use actual SVGs
        switch (icon) {
            case 'laptop': return '💻';
            case 'gpu': return '🎮';
            case 'server': return '🖥️';
            case 'cpu': return '🧠';
            default: return '⚙️';
        }
    };

    return (
        <div className="modal-backdrop">
            <div className="modal" style={{ maxWidth: 800 }}>
                <div className="modal-header">
                    <h2>Hardware Auto-Tuner</h2>
                    <button className="btn btn-ghost" onClick={onClose}>✕</button>
                </div>

                <div className="modal-body" style={{ display: 'flex', gap: 'var(--space-lg)' }}>

                    {/* Left side: Hardware Selection */}
                    <div style={{ flex: 1, borderRight: '1px solid var(--border-color)', paddingRight: 'var(--space-lg)' }}>
                        <h4 style={{ marginTop: 0 }}>Target Deployment Hardware</h4>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-md)' }}>
                            Select where this model will run. We'll optimize the training parameters and target quantization.
                        </p>

                        <div style={{ marginBottom: 'var(--space-md)' }}>
                            <label className="form-label">Task Type</label>
                            <select
                                className="form-select"
                                value={taskType}
                                onChange={e => setTaskType(e.target.value)}
                            >
                                <option value="causal_lm">Text Generation (Causal LM)</option>
                                <option value="classification">Text Classification</option>
                            </select>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                            {profiles.map(p => (
                                <div
                                    key={p.id}
                                    style={{
                                        padding: 'var(--space-md)',
                                        border: `1px solid ${selectedId === p.id ? 'var(--primary-color)' : 'var(--border-color)'}`,
                                        borderRadius: 'var(--radius-md)',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 'var(--space-md)',
                                        backgroundColor: selectedId === p.id ? 'rgba(59, 130, 246, 0.05)' : 'transparent'
                                    }}
                                    onClick={() => setSelectedId(p.id)}
                                >
                                    <div style={{ fontSize: '1.5rem' }}>{getIconSrc(p.icon)}</div>
                                    <div>
                                        <div style={{ fontWeight: 500 }}>{p.name}</div>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{p.description}</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Right side: Recommendation results */}
                    <div style={{ flex: 1 }}>
                        <h4 style={{ marginTop: 0 }}>Recommended Configuration</h4>

                        {loading ? (
                            <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}>Analyzing hardware profile...</div>
                        ) : recommendation ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                                <div style={{ background: 'var(--bg-secondary)', padding: 'var(--space-md)', borderRadius: 'var(--radius-md)' }}>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        Target Base Model
                                    </div>
                                    <div style={{ fontWeight: 600, fontSize: '1.1rem', marginTop: 4 }}>
                                        {recommendation.base_model}
                                    </div>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-sm)' }}>
                                    <div style={{ background: 'var(--bg-secondary)', padding: 'var(--space-md)', borderRadius: 'var(--radius-md)' }}>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                            Compression
                                        </div>
                                        <div style={{ fontWeight: 600, fontSize: '1.1rem', marginTop: 4 }}>
                                            {recommendation.compression_bits}-bit
                                        </div>
                                    </div>
                                    <div style={{ background: 'var(--bg-secondary)', padding: 'var(--space-md)', borderRadius: 'var(--radius-md)' }}>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                            LoRA Rank
                                        </div>
                                        <div style={{ fontWeight: 600, fontSize: '1.1rem', marginTop: 4 }}>
                                            {recommendation.lora_rank}
                                        </div>
                                    </div>
                                </div>

                                <div style={{ padding: 'var(--space-md)', color: 'var(--primary-color)', background: 'rgba(59, 130, 246, 0.1)', borderRadius: 'var(--radius-md)', fontSize: '0.9rem' }}>
                                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                                        {recommendation.notes.map((note, i) => (
                                            <li key={i} style={{ marginBottom: 4 }}>{note}</li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        ) : (
                            <div style={{ color: 'var(--text-secondary)' }}>Select a hardware profile to see recommendations.</div>
                        )}
                    </div>
                </div>

                <div className="modal-footer">
                    <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
                    <button
                        className="btn btn-primary"
                        disabled={!recommendation || loading}
                        onClick={handleApply}
                    >
                        Apply Configuration
                    </button>
                </div>
            </div>
        </div>
    );
}
