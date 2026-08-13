import { useEffect, useState } from 'react';
import { getLessons, startPracticeSession } from '../api';
import { useNavigate } from 'react-router-dom';

export default function PracticeSetup({ studentId }: { studentId: string }) {
    const [lessons, setLessons] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchLessons = async () => {
            try {
                const data = await getLessons();
                setLessons(data);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        fetchLessons();
    }, []);

    const handleStart = async (lessonId: number) => {
        try {
            const result = await startPracticeSession(lessonId, studentId, true);
            navigate(`/practice/session/${result.session_id}`, { state: { reference: result.reference_sign, expected: result.lesson_sign } });
        } catch (e) {
            alert("Failed to start session.");
        }
    };

    if (loading) return <div className="card fade-in"><h2>Loading practice modules...</h2></div>;

    return (
        <div className="fade-in">
            <h1 style={{ marginBottom: '2rem' }}>Select Alphabet to Practice</h1>
            <div className="grid-4">
                {lessons.map(lesson => (
                    <div key={lesson.id} className="card" style={{ textAlign: 'center', cursor: 'pointer' }} onClick={() => handleStart(lesson.id)}>
                        <div style={{ fontSize: '4rem', fontWeight: 'bold', color: 'var(--primary-color)', marginBottom: '1rem' }}>
                            {lesson.sign}
                        </div>
                        <button className="primary" style={{ width: '100%' }}>Practice</button>
                    </div>
                ))}
            </div>
        </div>
    );
}
