import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import ResultCard from '../components/ResultCard';
import { predictFromImage } from '../utils/api';
import { FaXRay } from 'react-icons/fa';
import './ImagePredictor.css';

const ImagePredictor = () => {
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (imageFile) => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const data = await predictFromImage(imageFile);
            setResults(data);
        } catch (err) {
            setError('Failed to analyze image. Please try again.');
            console.error('Image prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="image-predictor-page">
            <div className="page-header">
                <div className="header-icon">
                    <FaXRay />
                </div>
                <div className="header-content">
                    <h1>Pneumonia Image Detector</h1>
                    <p>
                        Upload chest X-ray images to detect signs of pneumonia using AI-powered analysis.
                    </p>
                </div>
            </div>

            <div className="predictor-content">
                <div className="predictor-main">
                    <ImageUpload onSubmit={handleSubmit} loading={loading} />

                    {error && (
                        <div className="error-message fade-in">
                            <p>{error}</p>
                        </div>
                    )}

                    {results && <ResultCard results={results} />}
                </div>

                <aside className="predictor-sidebar">
                    <div className="info-card">
                        <h3>Supported Images</h3>
                        <ul className="supported-list">
                            <li>
                                <strong>Chest X-Ray Images</strong>
                                <p>Chest X-rays for pneumonia detection</p>
                            </li>
                            <li>
                                <strong>File Formats</strong>
                                <p>JPG, PNG, JPEG</p>
                            </li>
                            <li>
                                <strong>Image Quality</strong>
                                <p>Clear, high-resolution images work best</p>
                            </li>
                            <li>
                                <strong>File Size</strong>
                                <p>Maximum 10MB per image</p>
                            </li>
                        </ul>
                    </div>

                    <div className="info-card analysis-card">
                        <h3>Analysis Process</h3>
                        <div className="process-steps">
                            <div className="process-step">
                                <div className="step-number">1</div>
                                <div className="step-content">
                                    <strong>Upload</strong>
                                    <p>Select or drag X-ray image</p>
                                </div>
                            </div>
                            <div className="process-step">
                                <div className="step-number">2</div>
                                <div className="step-content">
                                    <strong>Process</strong>
                                    <p>AI analyzes the image</p>
                                </div>
                            </div>
                            <div className="process-step">
                                <div className="step-number">3</div>
                                <div className="step-content">
                                    <strong>Results</strong>
                                    <p>Get detailed predictions</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="info-card warning-card">
                        <h3>⚠️ Important Note</h3>
                        <p>
                            This tool is for educational purposes only. Always consult with qualified
                            radiologists and healthcare professionals for accurate medical diagnosis.
                        </p>
                    </div>
                </aside>
            </div>
        </div>
    );
};

export default ImagePredictor;
