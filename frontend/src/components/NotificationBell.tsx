import { useEffect, useRef, useState } from 'react';
import { Bell, X, CheckCheck } from 'lucide-react';
import api from '../api';

interface Notification {
    id: string;
    type: string;
    title: string;
    body: string;
    severity: 'info' | 'warning' | 'success';
    read: boolean;
    created_at: string;
}

const SEVERITY_COLORS: Record<string, string> = {
    info: '#3b82f6',
    warning: '#f59e0b',
    success: '#10b981',
};

export default function NotificationBell({ studentId }: { studentId: string }) {
    const [notifs, setNotifs] = useState<Notification[]>([]);
    const [open, setOpen] = useState(false);
    const panelRef = useRef<HTMLDivElement>(null);

    const unread = notifs.filter(n => !n.read).length;

    const fetchAndGenerate = async () => {
        try {
            // Trigger generation then fetch all
            await api.post(`/notifications/${studentId}/generate`);
            const res = await api.get(`/notifications/${studentId}`);
            setNotifs(res.data);
        } catch { /* ignore */ }
    };

    useEffect(() => {
        fetchAndGenerate();
        // Poll every 2 minutes
        const id = setInterval(fetchAndGenerate, 120_000);
        return () => clearInterval(id);
    }, [studentId]);

    // Close on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const markRead = async (id: string) => {
        await api.patch(`/notifications/${studentId}/${id}/read`);
        setNotifs(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
    };

    const markAllRead = async () => {
        await api.patch(`/notifications/${studentId}/read-all`);
        setNotifs(prev => prev.map(n => ({ ...n, read: true })));
    };

    return (
        <div ref={panelRef} style={{ position: 'relative' }}>
            {/* Bell button */}
            <button
                onClick={() => setOpen(o => !o)}
                style={{
                    position: 'relative', background: 'none', border: 'none',
                    cursor: 'pointer', padding: '0.4rem', color: 'var(--text-muted)',
                    display: 'flex', alignItems: 'center'
                }}
                title="Notifications"
            >
                <Bell size={20} />
                {unread > 0 && (
                    <span style={{
                        position: 'absolute', top: '0', right: '0',
                        background: '#ef4444', color: '#fff',
                        borderRadius: '999px', fontSize: '0.65rem',
                        padding: '0 0.35rem', fontWeight: 'bold', lineHeight: '1.5',
                        minWidth: '1.1rem', textAlign: 'center'
                    }}>{unread}</span>
                )}
            </button>

            {/* Dropdown panel */}
            {open && (
                <div style={{
                    position: 'absolute', right: 0, top: 'calc(100% + 0.5rem)',
                    width: '340px', maxHeight: '420px', overflowY: 'auto',
                    background: 'var(--surface-color)', border: '1px solid var(--surface-hover)',
                    borderRadius: '12px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                    zIndex: 1000,
                }}>
                    {/* Header */}
                    <div style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '1rem 1.25rem', borderBottom: '1px solid var(--surface-hover)'
                    }}>
                        <span style={{ fontWeight: 'bold' }}>Notifications {unread > 0 && `(${unread})`}</span>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            {unread > 0 && (
                                <button onClick={markAllRead} title="Mark all read"
                                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                                    <CheckCheck size={16} />
                                </button>
                            )}
                            <button onClick={() => setOpen(false)}
                                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                                <X size={16} />
                            </button>
                        </div>
                    </div>

                    {/* List */}
                    {notifs.length === 0 ? (
                        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                            No notifications yet
                        </div>
                    ) : (
                        notifs.map(n => (
                            <div key={n.id} onClick={() => !n.read && markRead(n.id)}
                                style={{
                                    padding: '0.85rem 1.25rem',
                                    borderBottom: '1px solid var(--surface-hover)',
                                    cursor: n.read ? 'default' : 'pointer',
                                    background: n.read ? 'transparent' : SEVERITY_COLORS[n.severity] + '0d',
                                    borderLeft: `3px solid ${n.read ? 'transparent' : SEVERITY_COLORS[n.severity]}`,
                                    transition: 'background 0.2s',
                                }}>
                                <div style={{ fontWeight: n.read ? 400 : 600, fontSize: '0.9rem', marginBottom: '0.25rem' }}>
                                    {n.title}
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                    {n.body}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
}
