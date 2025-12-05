import React from 'react';
import { FaExclamationTriangle, FaCheckCircle, FaInfoCircle, FaLightbulb } from 'react-icons/fa';
import './ResultCard.css';

const ResultCard = ({ results }) => {
    if (!results || !results.predictions || results.predictions.length === 0) {
        return null;
    }

    const getSeverityColor = (severity) => {
        const severityLower = severity?.toLowerCase() || '';
        if (severityLower.includes('severe')) return 'var(--error)';
        if (severityLower.includes('moderate')) return 'var(--warning)';
        if (severityLower.includes('mild')) return 'var(--info)';
        if (severityLower.includes('none') || severityLower.includes('normal')) return 'var(--success)';
        return 'var(--text-muted)';
    };

    const getSeverityIcon = (severity) => {
        const severityLower = severity?.toLowerCase() || '';
        if (severityLower.includes('severe')) return <FaExclamationTriangle />;
        if (severityLower.includes('none') || severityLower.includes('normal')) return <FaCheckCircle />;
        return <FaInfoCircle />;
    };

    return (
        <div className="results-container fade-in">
            <div className="results-header">
                <h2>Analysis Results</h2>
                {results.model_type === 'demo' && (
                    <div className="demo-badge">
                        <FaInfoCircle />
                        <span>Demo Mode</span>
                    </div>
                )}
            </div>

            {results.note && (
                <div className="info-banner">
                    <FaInfoCircle />
                    <p>{results.note}</p>
                </div>
            )}

            {results.warning && (
                <div className="warning-banner">
                    <FaExclamationTriangle />
                    <p>{results.warning}</p>
                </div>
            )}

            <div className="predictions-grid">
                {results.predictions.map((prediction, index) => (
                    <div key={index} className="prediction-card" style={{ animationDelay: `${index * 0.1}s` }}>
                        <div className="prediction-header">
                            <div className="prediction-rank">#{index + 1}</div>
                            <div className="prediction-title">
                                <h3>{prediction.disease || prediction.condition}</h3>
                                <div
                                    className="severity-badge"
                                    style={{
                                        backgroundColor: `${getSeverityColor(prediction.severity)}20`,
                                        color: getSeverityColor(prediction.severity)
                                    }}
                                >
                                    {getSeverityIcon(prediction.severity)}
                                    <span>{prediction.severity}</span>
                                </div>
                            </div>
                        </div>

                        <div className="confidence-section">
                            <div className="confidence-label">
                                <span>Confidence</span>
                                <span className="confidence-value">{(prediction.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="confidence-bar">
                                <div
                                    className="confidence-fill"
                                    style={{
                                        width: `${prediction.confidence * 100}%`,
                                        background: `linear-gradient(90deg, ${getSeverityColor(prediction.severity)}, ${getSeverityColor(prediction.severity)}aa)`
                                    }}
                                ></div>
                            </div>
                        </div>

                        {prediction.description && (
                            <div className="prediction-description">
                                <p>{prediction.description}</p>
                            </div>
                        )}

                        {prediction.recommendations && prediction.recommendations.length > 0 && (
                            <div className="recommendations-section">
                                <h4>
                                    <FaLightbulb />
                                    <span>Recommendations</span>
                                </h4>
                                <ul className="recommendations-list">
                                    {prediction.recommendations.map((rec, idx) => (
                                        <li key={idx}>{rec}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <div className="disclaimer-section">
                <FaExclamationTriangle className="disclaimer-icon" />
                <p>
                    <strong>Important:</strong> These results are AI-generated predictions for educational purposes only.
                    Always consult with qualified healthcare professionals for accurate diagnosis and treatment.
                </p>
            </div>
        </div>
    );
};

export default ResultCard;
