import { useEffect, useRef, useState } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { submitAttempt, streamFrame, endPracticeSession } from '../api';

export default function PracticeSession({ studentId }: { studentId: string }) {
    const { id: sessionId } = useParams();
    const location = useLocation();
    const navigate = useNavigate();
    const videoRef = useRef<HTMLVideoElement>(null);

    const [feedback, setFeedback] = useState<any>(null);
    const [submitting, setSubmitting] = useState(false);
    const [streamInfo, setStreamInfo] = useState<any>(null);

    const referenceSign = location.state?.reference;
    const expectedSign = location.state?.expected;

    useEffect(() => {
        let stream: MediaStream | null = null;
        let interval: number;

        const startCamera = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }

                // Setup streaming interval
                interval = window.setInterval(async () => {
                    if (!videoRef.current) return;
                    const canvas = document.createElement('canvas');
                    canvas.width = videoRef.current.videoWidth;
                    canvas.height = videoRef.current.videoHeight;
                    const ctx = canvas.getContext('2d');
                    if (!ctx) return;
                    ctx.drawImage(videoRef.current, 0, 0);

                    canvas.toBlob(async (blob) => {
                        if (blob && sessionId) {
                            const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
                            try {
                                const info = await streamFrame(sessionId, file);
                                setStreamInfo(info);
                            } catch (e) {
                                // Silently drop frame errors unless critical
                            }
                        }
                    }, 'image/jpeg', 0.8);
                }, 300);

            } catch (err) {
                console.error("Camera error:", err);
            }
        };

        startCamera();

        return () => {
            if (stream) stream.getTracks().forEach(track => track.stop());
            clearInterval(interval);
        };
    }, [sessionId]);

    const handlePredict = () => {
        if (!videoRef.current || !sessionId) return;
        setSubmitting(true);

        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob(async (blob) => {
            if (blob) {
                const file = new File([blob], 'attempt.jpg', { type: 'image/jpeg' });
                try {
                    const result = await submitAttempt(sessionId, file);
                    setFeedback(result);

                    // If auto-next triggered
                    if (result.is_correct && result.next_lesson) {
                        setTimeout(() => {
                            navigate(`/practice/session/${sessionId}`, { replace: true, state: { reference: null, expected: result.next_lesson } });
                            setFeedback(null);
                        }, 2000);
                    }
                } catch (e) {
                    console.error(e);
                } finally {
                    setSubmitting(false);
                }
            }
        }, 'image/jpeg', 0.9);
    };

    const handleEnd = async () => {
        if (!sessionId) return;
        await endPracticeSession(sessionId);
        navigate(`/practice/review/${sessionId}`);
    };

    return (
        <div className="fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <h2>Practice: Alphabet "{expectedSign}"</h2>
                <button className="danger" onClick={handleEnd}>End Session</button>
            </div>

            <div className="grid-2">
                <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{ position: 'relative', width: '100%', paddingBottom: '75%', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden' }}>
                        <video ref={videoRef} autoPlay playsInline muted style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
                    </div>

                    <button className="primary" onClick={handlePredict} disabled={submitting} style={{ padding: '1rem', fontSize: '1.25rem' }}>
                        {submitting ? 'Analyzing...' : 'Predict Gesture'}
                    </button>

                    {streamInfo && (
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            Camera Active | {streamInfo.buffered_frames} buffered |
                            Hands: {streamInfo.hand_count} |
                            Body Visible: {streamInfo.upper_body_visible ? 'Yes' : 'No'}
                        </div>
                    )}
                </div>

                <div className="card">
                    <h3 className="card-title">Feedback & Reference</h3>

                    {referenceSign && (
                        <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '1rem' }}>
                            <strong>Expected Sign Description</strong> (Implementation missing from backend to serve actual image bytes, assuming text/base64 placeholder)
                        </div>
                    )}

                    {feedback && (
                        <div className={`slide-down`} style={{
                            padding: '1.5rem',
                            borderRadius: '8px',
                            border: `2px solid ${feedback.is_correct ? 'var(--success-color)' : 'var(--danger-color)'}`,
                            background: feedback.is_correct ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'
                        }}>
                            <h2 style={{ color: feedback.is_correct ? 'var(--success-color)' : 'var(--danger-color)', marginBottom: '0.5rem' }}>
                                {feedback.is_correct ? 'Correct! Great Job!' : 'Incorrect'}
                            </h2>

                            <div style={{ marginBottom: '1rem' }}>
                                <p><strong>Predicted:</strong> {feedback.predicted_class} ({(feedback.confidence * 100).toFixed(1)}%)</p>
                                {feedback.motion_metrics && (
                                    <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                                        Stability: {feedback.motion_metrics.stable_streak} | Latency: {(feedback.processing_time * 1000).toFixed(0)}ms
                                    </p>
                                )}
                            </div>

                            {feedback.feedback && feedback.feedback.length > 0 && (
                                <div style={{ background: 'var(--surface-color)', padding: '1rem', borderRadius: '8px' }}>
                                    <h4 style={{ marginBottom: '0.5rem' }}>Advice</h4>
                                    <ul style={{ margin: 0, paddingLeft: '1.25rem' }}>
                                        {feedback.feedback.map((fb: any, i: number) => (
                                            <li key={i} style={{ marginBottom: '0.25rem' }}>{fb.message}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
