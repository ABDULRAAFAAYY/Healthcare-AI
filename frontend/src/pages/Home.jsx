import React from 'react';
import { Link } from 'react-router-dom';
import { FaStethoscope, FaXRay, FaBrain, FaChartLine, FaShieldAlt, FaRocket } from 'react-icons/fa';
import './Home.css';

const Home = () => {
    const features = [
        {
            icon: <FaStethoscope />,
            title: 'Symptom Analysis',
            description: 'Advanced AI-powered disease prediction based on your symptoms with high accuracy.',
            color: '#2563eb'
        },
        {
            icon: <FaXRay />,
            title: 'Medical Imaging',
            description: 'X-ray image analysis using deep learning for detecting various conditions.',
            color: '#8b5cf6'
        },
        {
            icon: <FaBrain />,
            title: 'Machine Learning',
            description: 'State-of-the-art ML models trained on extensive medical datasets.',
            color: '#10b981'
        },
        {
            icon: <FaChartLine />,
            title: 'Detailed Reports',
            description: 'Comprehensive analysis with confidence scores and recommendations.',
            color: '#f59e0b'
        },
        {
            icon: <FaShieldAlt />,
            title: 'Secure & Private',
            description: 'Your medical data is processed securely with privacy protection.',
            color: '#ef4444'
        },
        {
            icon: <FaRocket />,
            title: 'Fast Results',
            description: 'Get instant predictions and analysis in seconds.',
            color: '#06b6d4'
        }
    ];

    return (
        <div className="home-page">
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <span className="badge-dot"></span>
                        <span>AI-Powered Healthcare</span>
                    </div>
                    <h1 className="hero-title">
                        Advanced Disease Prediction & Medical Analysis
                    </h1>
                    <p className="hero-description">
                        Harness the power of artificial intelligence for accurate disease prediction and medical image analysis.
                        Get instant insights powered by state-of-the-art machine learning models.
                    </p>
                    <div className="hero-actions">
                        <Link to="/disease-predictor" className="btn btn-primary btn-large">
                            <FaStethoscope />
                            <span>Check Symptoms</span>
                        </Link>
                        <Link to="/image-predictor" className="btn btn-secondary btn-large">
                            <FaXRay />
                            <span>Analyze X-Ray</span>
                        </Link>
                    </div>
                </div>
                <div className="hero-visual">
                    <div className="floating-card card-1">
                        <div className="card-icon" style={{ background: 'linear-gradient(135deg, #2563eb, #3b82f6)' }}>
                            <FaBrain />
                        </div>
                        <div className="card-content">
                            <h4>AI Powered</h4>
                            <p>Advanced ML Models</p>
                        </div>
                    </div>
                    <div className="floating-card card-2">
                        <div className="card-icon" style={{ background: 'linear-gradient(135deg, #10b981, #34d399)' }}>
                            <FaChartLine />
                        </div>
                        <div className="card-content">
                            <h4>High Accuracy</h4>
                            <p>Reliable Predictions</p>
                        </div>
                    </div>
                    <div className="floating-card card-3">
                        <div className="card-icon" style={{ background: 'linear-gradient(135deg, #8b5cf6, #a78bfa)' }}>
                            <FaRocket />
                        </div>
                        <div className="card-content">
                            <h4>Instant Results</h4>
                            <p>Real-time Analysis</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features-section">
                <div className="section-header">
                    <h2>Powerful Features</h2>
                    <p>Everything you need for AI-powered medical analysis</p>
                </div>
                <div className="features-grid">
                    {features.map((feature, index) => (
                        <div
                            key={index}
                            className="feature-card"
                            style={{ animationDelay: `${index * 0.1}s` }}
                        >
                            <div className="feature-icon" style={{ color: feature.color }}>
                                {feature.icon}
                            </div>
                            <h3>{feature.title}</h3>
                            <p>{feature.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <div className="cta-content">
                    <h2>Ready to Get Started?</h2>
                    <p>Experience the future of healthcare with AI-powered diagnostics</p>
                    <div className="cta-actions">
                        <Link to="/disease-predictor" className="btn btn-primary btn-large">
                            Start Symptom Analysis
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Home;
