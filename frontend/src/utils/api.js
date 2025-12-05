import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

// Create axios instance with default config
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

/**
 * Health check endpoint
 */
export const checkHealth = async () => {
    try {
        const response = await api.get('/health');
        return response.data;
    } catch (error) {
        console.error('Health check failed:', error);
        throw error;
    }
};

/**
 * Get list of available symptoms
 */
export const getSymptomsList = async () => {
    try {
        const response = await api.get('/symptoms/list');
        return response.data;
    } catch (error) {
        console.error('Failed to fetch symptoms:', error);
        throw error;
    }
};

/**
 * Predict disease based on symptoms
 * @param {Array<string>} symptoms - Array of symptom strings
 */
export const predictDisease = async (symptoms) => {
    try {
        const response = await api.post('/predict/symptoms', { symptoms });
        return response.data;
    } catch (error) {
        console.error('Disease prediction failed:', error);
        throw error;
    }
};

/**
 * Predict condition from medical image
 * @param {File} imageFile - Image file to analyze
 */
export const predictFromImage = async (imageFile) => {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);

        const response = await axios.post(`${API_BASE_URL}/predict/image`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        return response.data;
    } catch (error) {
        console.error('Image prediction failed:', error);
        throw error;
    }
};

export default api;
