import React, { useState } from 'react';
import SymptomForm from '../components/SymptomForm';
import ResultCard from '../components/ResultCard';
import { predictDisease } from '../utils/api';
import { FaStethoscope } from 'react-icons/fa';
import './DiseasePredictor.css';

const DiseasePredictor = () => {
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (symptoms) => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const data = await predictDisease(symptoms);
            setResults(data);
        } catch (err) {
            setError('Failed to get prediction. Please try again.');
            console.error('Prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="disease-predictor-page">
            <div className="page-header">
                <div className="header-icon">
                    <FaStethoscope />
                </div>
                <div className="header-content">
                    <h1>Symptom-Based Disease Predictor</h1>
                    <p>
                        Select your symptoms and get AI-powered disease predictions with confidence scores and recommendations.
                    </p>
                </div>
            </div>

            <div className="predictor-content">
                <div className="predictor-main">
                    <SymptomForm onSubmit={handleSubmit} loading={loading} />

                    {error && (
                        <div className="error-message fade-in">
                            <p>{error}</p>
                        </div>
                    )}

                    {results && <ResultCard results={results} />}
                </div>

                <aside className="predictor-sidebar">
                    <div className="info-card">
                        <h3>How It Works</h3>
                        <ol className="steps-list">
                            <li>
                                <strong>Select Symptoms</strong>
                                <p>Search and select all symptoms you're experiencing</p>
                            </li>
                            <li>
                                <strong>AI Analysis</strong>
                                <p>Our ML model analyzes your symptom combination</p>
                            </li>
                            <li>
                                <strong>Get Results</strong>
                                <p>Receive predictions with confidence scores</p>
                            </li>
                            <li>
                                <strong>Consult Doctor</strong>
                                <p>Use results to inform your healthcare professional</p>
                            </li>
                        </ol>
                    </div>

                    <div className="info-card tips-card">
                        <h3>Tips for Best Results</h3>
                        <ul className="tips-list">
                            <li>Select all symptoms you're experiencing</li>
                            <li>Be as specific as possible</li>
                            <li>Include both major and minor symptoms</li>
                            <li>Update if symptoms change</li>
                        </ul>
                    </div>
                </aside>
            </div>
        </div>
    );
};

export default DiseasePredictor;
