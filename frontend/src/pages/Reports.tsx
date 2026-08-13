import { useState } from 'react';
import { Download, FileText } from 'lucide-react';

export default function Reports({ studentId }: { studentId: string }) {
    const [downloading, setDownloading] = useState(false);

    // We are creating a link because Report endpoints are not fully available yet,
    // but wait! Let me build a UI that fetches reports when the endpoint is ready (Task 3).
    // For now, this is a placeholder.

    const handleDownload = async (type: 'pdf' | 'csv') => {
        setDownloading(true);
        try {
            window.open(`http://localhost:8000/reports/${studentId}/export?format=${type}`, '_blank');
        } catch (e) {
            console.error(e);
        } finally {
            setTimeout(() => setDownloading(false), 1000);
        }
    };

    return (
        <div className="fade-in">
            <h2 style={{ marginBottom: '2rem' }}>Export Performance Reports</h2>
            <div className="grid-2">
                <div className="card" style={{ textAlign: 'center', padding: '3rem 2rem' }}>
                    <FileText size={48} color="var(--primary-color)" style={{ marginBottom: '1rem' }} />
                    <h3>PDF Summary Report</h3>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Detailed visually rich report of all metrics and recommendations.</p>
                    <button className="primary" onClick={() => handleDownload('pdf')} disabled={downloading}>
                        <Download size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} /> Download PDF
                    </button>
                </div>
                <div className="card" style={{ textAlign: 'center', padding: '3rem 2rem' }}>
                    <FileText size={48} color="var(--success-color)" style={{ marginBottom: '1rem' }} />
                    <h3>CSV Raw Data</h3>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Export all raw attempts for Excel and Spreadsheet analysis.</p>
                    <button className="secondary" onClick={() => handleDownload('csv')} disabled={downloading}>
                        <Download size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} /> Download CSV
                    </button>
                </div>
            </div>
        </div>
    );
}
