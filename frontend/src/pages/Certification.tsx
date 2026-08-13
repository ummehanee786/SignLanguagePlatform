import { useEffect, useState } from 'react';
import api from '../api';
import { Award, CheckCircle, XCircle, RefreshCw } from 'lucide-react';

const LEVEL_COLORS: Record<string, string> = {
    Beginner: '#10b981',
    Intermediate: '#3b82f6',
    Advanced: '#f59e0b',
};

export default function Certification({ studentId }: { studentId: string }) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [evaluating, setEvaluating] = useState(false);

    const fetchCerts = async () => {
        try {
            const res = await api.get(`/certifications/${studentId}`);
            setData(res.data);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    useEffect(() => { fetchCerts(); }, [studentId]);

    const handleEvaluate = async () => {
        setEvaluating(true);
        try {
            const res = await api.post(`/certifications/${studentId}/evaluate`);
            setData(res.data);
            if (res.data.newly_awarded?.length > 0) {
                alert(`🎉 Congratulations! You earned: ${res.data.newly_awarded.join(', ')} certification!`);
            } else {
                alert('No new certifications at this time. Keep practicing!');
            }
        } catch (e) { console.error(e); }
        finally { setEvaluating(false); }
    };

    if (loading) return <div className="card fade-in"><h2>Loading Certifications...</h2></div>;

    const earned: string[] = (data?.earned_certifications || []).map((c: any) => c.level);
    const levels = data?.level_status || {};

    return (
        <div className="fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <div>
                    <h2>Skill Certifications</h2>
                    <p style={{ color: 'var(--text-muted)' }}>Evaluate your progress and earn achievement badges.</p>
                </div>
                <button className="primary" onClick={handleEvaluate} disabled={evaluating}
                    style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <RefreshCw size={16} className={evaluating ? 'spin' : ''} />
                    {evaluating ? 'Evaluating...' : 'Evaluate Now'}
                </button>
            </div>

            {/* Earned badge strip */}
            {earned.length > 0 && (
                <div className="card" style={{ marginBottom: '2rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    {earned.map((lvl) => (
                        <div key={lvl} style={{
                            display: 'flex', alignItems: 'center', gap: '0.5rem',
                            padding: '0.75rem 1.5rem', borderRadius: '999px',
                            background: LEVEL_COLORS[lvl] + '22', border: `2px solid ${LEVEL_COLORS[lvl]}`,
                            color: LEVEL_COLORS[lvl], fontWeight: 'bold'
                        }}>
                            <Award size={20} /> {lvl} Certified
                        </div>
                    ))}
                </div>
            )}

            {/* Level cards */}
            <div className="grid-3">
                {Object.entries(LEVEL_COLORS).map(([level, color]) => {
                    const info = levels[level];
                    const isEarned = earned.includes(level);
                    const eligible = info?.eligible;
                    const breakdown = info?.letter_breakdown || {};
                    const passCount = Object.values(breakdown).filter((v: any) => v.meets_criteria).length;
                    const total = Object.keys(breakdown).length;

                    return (
                        <div key={level} className="card" style={{
                            borderTop: `4px solid ${color}`, position: 'relative'
                        }}>
                            {isEarned && (
                                <div style={{
                                    position: 'absolute', top: '1rem', right: '1rem',
                                    background: color + '22', borderRadius: '999px',
                                    padding: '0.25rem 0.75rem', color, fontSize: '0.8rem', fontWeight: 'bold'
                                }}>Earned ✓</div>
                            )}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                                <Award size={28} color={color} />
                                <div>
                                    <div style={{ fontWeight: 'bold', fontSize: '1.1rem' }}>{level}</div>
                                    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                                        ≥{info?.required_accuracy}% acc · {info?.required_attempts_per_letter}+ attempts/letter
                                    </div>
                                </div>
                            </div>

                            <div style={{
                                display: 'flex', alignItems: 'center', gap: '0.5rem',
                                marginBottom: '1rem', padding: '0.5rem 1rem', borderRadius: '8px',
                                background: eligible ? '#10b98122' : '#ef444422'
                            }}>
                                {eligible
                                    ? <CheckCircle size={16} color="#10b981" />
                                    : <XCircle size={16} color="#ef4444" />}
                                <span style={{ color: eligible ? '#10b981' : '#ef4444', fontWeight: 'bold', fontSize: '0.9rem' }}>
                                    {eligible ? 'Eligible' : `${passCount}/${total} letters pass`}
                                </span>
                            </div>

                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                                {Object.entries(breakdown).map(([letter, v]: [string, any]) => (
                                    <div key={letter} title={`${letter}: ${v.accuracy}% (${v.attempts} attempts)`}
                                        style={{
                                            width: '2rem', height: '2rem', borderRadius: '6px',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            fontWeight: 'bold', fontSize: '0.8rem',
                                            background: v.meets_criteria ? color + '33' : 'var(--surface-hover)',
                                            color: v.meets_criteria ? color : 'var(--text-muted)',
                                            border: `1px solid ${v.meets_criteria ? color : 'transparent'}`
                                        }}>
                                        {letter}
                                    </div>
                                ))}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
