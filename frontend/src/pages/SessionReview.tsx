import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getSessionReview } from '../api';
import { CheckCircle, XCircle, ChevronRight, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function SessionReview({ studentId }: { studentId: string }) {
    const { id: sessionId } = useParams();
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchReview = async () => {
            if (!sessionId) return;
            try {
                const review = await getSessionReview(sessionId, studentId);
                setData(review);
            } catch (e: any) {
                setError(e.response?.data?.detail || "Session not found or no attempts made.");
            } finally {
                setLoading(false);
            }
        };
        fetchReview();
    }, [sessionId, studentId]);

    if (loading) return <div className="card fade-in"><h2>Loading Review...</h2></div>;
    if (error || !data) return <div className="card danger fade-in"><p>{error || "No data"}</p><Link to="/"><button className="secondary" style={{ marginTop: '1rem' }}>Back to Home</button></Link></div>;

    const {
        overall_score, correct_attempts, incorrect_attempts,
        confidence_trend, most_common_mistakes, gesture_feedback, recommended_gestures
    } = data;

    return (
        <div className="fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <div>
                    <h2>Session Summary</h2>
                    <p style={{ color: 'var(--text-muted)' }}>Review your performance in the last practice session.</p>
                </div>
                <Link to="/">
                    <button className="primary" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        Go to Dashboard <ChevronRight size={18} />
                    </button>
                </Link>
            </div>

            <div className="grid-3" style={{ marginBottom: '2rem' }}>
                <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Session Score</div>
                    <div style={{ fontSize: '3rem', fontWeight: 'bold' }}>{overall_score.toFixed(0)}%</div>
                </div>
                <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Correct</div>
                    <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--success-color)' }}>{correct_attempts}</div>
                </div>
                <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Incorrect</div>
                    <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--danger-color)' }}>{incorrect_attempts}</div>
                </div>
            </div>

            <div className="grid-2" style={{ marginBottom: '2rem' }}>
                <div className="card">
                    <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Activity size={20} /> Confidence Trend
                    </h3>
                    <div style={{ height: 250 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={confidence_trend} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-hover)" vertical={false} />
                                <XAxis dataKey="attempt_number" stroke="var(--text-muted)" fontSize={12} />
                                <YAxis dataKey="confidence" stroke="var(--text-muted)" fontSize={12} />
                                <Tooltip contentStyle={{ backgroundColor: 'var(--surface-color)', borderColor: 'var(--surface-hover)' }} />
                                <Line type="monotone" dataKey="confidence" stroke="var(--primary-color)" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                    <h3 className="card-title">Recommendations for Next Session</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1, justifyContent: 'center' }}>
                        {recommended_gestures?.length > 0 ? (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
                                {recommended_gestures.map((letter: string) => (
                                    <div key={letter} style={{ padding: '1rem 1.5rem', background: 'var(--surface-hover)', borderRadius: '8px', fontSize: '1.5rem', fontWeight: 'bold', border: '1px solid rgba(255,255,255,0.1)' }}>
                                        {letter}
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p style={{ color: 'var(--text-muted)' }}>Awesome job! No specific letters to re-practice.</p>
                        )}
                    </div>
                </div>
            </div>

            <div className="card">
                <h3 className="card-title">Gesture Feedback</h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--surface-hover)', color: 'var(--text-muted)' }}>
                                <th style={{ padding: '1rem' }}>Alphabet</th>
                                <th style={{ padding: '1rem' }}>Accuracy</th>
                                <th style={{ padding: '1rem' }}>Latest Tip</th>
                            </tr>
                        </thead>
                        <tbody>
                            {gesture_feedback?.map((gf: any) => (
                                <tr key={gf.gesture} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                                    <td style={{ padding: '1rem', fontWeight: 'bold' }}>
                                        {gf.accuracy_percentage === 100 ? <CheckCircle size={16} color="var(--success-color)" style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'text-bottom' }} /> :
                                            gf.accuracy_percentage === 0 ? <XCircle size={16} color="var(--danger-color)" style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'text-bottom' }} /> : null}
                                        {gf.gesture}
                                    </td>
                                    <td style={{ padding: '1rem' }}>{gf.accuracy_percentage}% ({gf.correct}/{gf.attempts})</td>
                                    <td style={{ padding: '1rem', color: 'var(--text-muted)' }}>{gf.last_feedback_tip || '-'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
