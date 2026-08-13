import { useEffect, useState } from 'react';
import { getDashboard } from '../api';
import { Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { BookOpen, Target, ShieldCheck, Award, Activity } from 'lucide-react';

export default function Dashboard({ studentId }: { studentId: string }) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const dashboardData = await getDashboard(studentId);
                setData(dashboardData);
            } catch (e) {
                console.error('Failed to load dashboard', e);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [studentId]);

    if (loading) return <div className="card fade-in"><h2>Loading Dashboard...</h2></div>;
    if (!data) return <div className="card danger fade-in"><p>No data available. Proceed to practice first.</p></div>;

    const {
        total_attempts, accuracy_percentage, average_confidence,
        daily_practice_streak, strongest_alphabets, weakest_alphabets,
        most_mistaken_alphabets, recommendations, learner_profile
    } = data;

    // Transform data for charts
    const masteryData = learner_profile?.alphabet_mastery
        ? Object.keys(learner_profile.alphabet_mastery).map(key => ({
            alphabet: key,
            mastery: learner_profile.alphabet_mastery[key] * 100
        }))
        : [];

    return (
        <div className="fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <div>
                    <h1 style={{ marginBottom: '0.25rem' }}>Welcome back, <span className="nav-brand">{studentId}</span>!</h1>
                    <p style={{ color: 'var(--text-muted)' }}>Here's your learning progress today.</p>
                </div>
                <Link to="/practice">
                    <button className="primary" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        <Activity size={18} /> Start Practice
                    </button>
                </Link>
            </div>

            <div className="grid-4" style={{ marginBottom: '2rem' }}>
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Accuracy</div>
                        <Target size={20} color="var(--primary-color)" />
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{accuracy_percentage.toFixed(1)}%</div>
                </div>
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Total Attempts</div>
                        <BookOpen size={20} color="var(--secondary-color)" />
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{total_attempts}</div>
                </div>
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Avg Confidence</div>
                        <ShieldCheck size={20} color="var(--success-color)" />
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{(average_confidence * 100).toFixed(1)}%</div>
                </div>
                <div className="card">
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Practice Streak</div>
                        <Award size={20} color="var(--warning-color)" />
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{daily_practice_streak} Days</div>
                </div>
            </div>

            <div className="grid-2" style={{ marginBottom: '2rem' }}>
                <div className="card">
                    <h3 className="card-title">Alphabet Mastery Overview</h3>
                    {masteryData.length > 0 ? (
                        <div style={{ height: 300 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={masteryData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                    <XAxis dataKey="alphabet" stroke="var(--text-muted)" fontSize={12} />
                                    <YAxis stroke="var(--text-muted)" fontSize={12} />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        contentStyle={{ backgroundColor: 'var(--surface-color)', border: '1px solid var(--surface-hover)', borderRadius: '8px' }}
                                    />
                                    <Bar dataKey="mastery" fill="url(#colorMastery)" radius={[4, 4, 0, 0]} />
                                    <defs>
                                        <linearGradient id="colorMastery" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--primary-color)" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="var(--primary-color)" stopOpacity={0.2} />
                                        </linearGradient>
                                    </defs>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <p style={{ color: 'var(--text-muted)' }}>Not enough data to display mastery yet.</p>
                    )}
                </div>

                <div className="card">
                    <h3 className="card-title">Personalized Recommendations</h3>
                    {recommendations?.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            {recommendations.map((rec: any, index: number) => (
                                <div key={index} style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', borderLeft: '4px solid var(--secondary-color)' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                                        <strong>Alphabet {rec.alphabet}</strong>
                                        <span className="badge warning">Needs Practice</span>
                                    </div>
                                    <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-muted)' }}>{rec.reason}</p>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p style={{ color: 'var(--text-muted)' }}>You're doing great! No specific recommendations right now.</p>
                    )}
                </div>
            </div>

            <div className="grid-3">
                <div className="card">
                    <h3 className="card-title">Strongest</h3>
                    {strongest_alphabets?.length > 0 ? (
                        <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-muted)' }}>
                            {strongest_alphabets.map((stat: any) => (
                                <li key={stat.alphabet}>
                                    <strong>{stat.alphabet}</strong>: {stat.accuracy_percentage}% ({stat.attempts} attempts)
                                </li>
                            ))}
                        </ul>
                    ) : <p style={{ color: 'var(--text-muted)' }}>-</p>}
                </div>

                <div className="card">
                    <h3 className="card-title">Weakest</h3>
                    {weakest_alphabets?.length > 0 ? (
                        <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-muted)' }}>
                            {weakest_alphabets.map((stat: any) => (
                                <li key={stat.alphabet}>
                                    <strong>{stat.alphabet}</strong>: {stat.accuracy_percentage}% ({stat.attempts} attempts)
                                </li>
                            ))}
                        </ul>
                    ) : <p style={{ color: 'var(--text-muted)' }}>-</p>}
                </div>

                <div className="card">
                    <h3 className="card-title">Most Mistaken</h3>
                    {most_mistaken_alphabets?.length > 0 ? (
                        <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-muted)' }}>
                            {most_mistaken_alphabets.map((stat: any) => (
                                <li key={stat.alphabet}>
                                    <strong>{stat.alphabet}</strong>: {stat.mistake_count} / {stat.attempts} mistakes
                                </li>
                            ))}
                        </ul>
                    ) : <p style={{ color: 'var(--text-muted)' }}>-</p>}
                </div>
            </div>
        </div>
    );
}
