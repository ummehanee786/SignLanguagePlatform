import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Make environment based later (Task 10)

const api = axios.create({
    baseURL: API_BASE_URL,
});

export default api;

export const getDashboard = async (studentId: string) => {
    const { data } = await api.get(`/progress/${studentId}/dashboard`);
    return data;
};

export const getLessons = async () => {
    const { data } = await api.get('/lessons');
    return data;
};

export const startPracticeSession = async (lessonId: number, studentId: string, autoNext: boolean = true) => {
    const { data } = await api.post(`/practice/start/${lessonId}?student_id=${studentId}&auto_next=${autoNext}`);
    return data;
};

export const submitAttempt = async (sessionId: string, studentId: string, predictedClass: string, isCorrect: boolean, confidence: number, processingTime: number) => {
    const res = await api.post(`/practice/${sessionId}/attempt`, {
        student_id: studentId,
        predicted_class: predictedClass,
        is_correct: isCorrect,
        confidence: confidence,
        inference_time_ms: processingTime
    });
    return res.data;
};

export const getAdminOverview = async () => {
    const res = await api.get('/admin/overview');
    return res.data;
};

export const endPracticeSession = async (sessionId: string) => {
    const { data } = await api.post(`/practice/${sessionId}/end`);
    return data;
};

export const getSessionReview = async (sessionId: string, studentId: string) => {
    const { data } = await api.get(`/practice/${sessionId}/review?student_id=${studentId}`);
    return data;
};

export const streamFrame = async (sessionId: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post(`/practice/${sessionId}/stream-frame`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    return data;
};
