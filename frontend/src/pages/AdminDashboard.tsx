import { useEffect, useState } from 'react';
import { getAdminOverview } from '../api';
import { Users, Activity, BarChart2 } from 'lucide-react';

export default function AdminDashboard() {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchAdminData = async () => {
            try {
                const overview = await getAdminOverview();
                setData(overview);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        fetchAdminData();
    }, []);

    if (loading) return <div className="card fade-in"><h2>Loading Admin View...</h2></div>;

    return (
        <div className="fade-in">
            <div style={{ marginBottom: '2rem' }}>
                <h2>Trainer & Instructor Dashboard</h2>
                <p style={{ color: 'var(--text-muted)' }}>Aggregate analytics for all active cohort learners.</p>
            </div>

            <div className="grid-3" style={{ marginBottom: '2rem' }}>
                <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Users size={48} color="var(--primary-color)" />
                    <div>
                        <div style={{ color: 'var(--text-muted)' }}>Total Enrolled</div>
                        <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{data?.total_students || 0}</div>
                    </div>
                </div>

                <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Activity size={48} color="var(--success-color)" />
                    <div>
                        <div style={{ color: 'var(--text-muted)' }}>Active Practitioners</div>
                        <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{data?.active_students || 0}</div>
                    </div>
                </div>

                <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <BarChart2 size={48} color="#f59e0b" />
                    <div>
                        <div style={{ color: 'var(--text-muted)' }}>Cohort Avg Accuracy</div>
                        <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{data?.average_accuracy || 0}%</div>
                    </div>
                </div>
            </div>

            <div className="card">
                <h3>Recent Cohort Alert Notifications</h3>
                <p style={{ color: 'var(--text-muted)' }}>Task 6 feature flag pending.</p>
            </div>
        </div>
    );
}
