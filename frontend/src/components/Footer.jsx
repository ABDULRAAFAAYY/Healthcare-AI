import React from 'react';
import { FaGithub, FaLinkedin, FaTwitter, FaHeart } from 'react-icons/fa';
import './Footer.css';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="footer">
            <div className="footer-container">
                <div className="footer-content">
                    <div className="footer-section">
                        <h3 className="footer-title">Healthcare AI</h3>
                        <p className="footer-description">
                            Advanced disease prediction and medical image analysis powered by machine learning.
                        </p>
                        <div className="footer-social">
                            <a href="#" className="social-link" aria-label="GitHub">
                                <FaGithub />
                            </a>
                            <a href="#" className="social-link" aria-label="LinkedIn">
                                <FaLinkedin />
                            </a>
                            <a href="#" className="social-link" aria-label="Twitter">
                                <FaTwitter />
                            </a>
                        </div>
                    </div>

                    <div className="footer-section">
                        <h4 className="footer-subtitle">Quick Links</h4>
                        <ul className="footer-links">
                            <li><a href="/">Home</a></li>
                            <li><a href="/disease-predictor">Symptom Checker</a></li>
                            <li><a href="/image-predictor">X-Ray Analysis</a></li>
                        </ul>
                    </div>

                    <div className="footer-section">
                        <h4 className="footer-subtitle">Resources</h4>
                        <ul className="footer-links">
                            <li><a href="#">Documentation</a></li>
                            <li><a href="#">API Reference</a></li>
                            <li><a href="#">Privacy Policy</a></li>
                            <li><a href="#">Terms of Service</a></li>
                        </ul>
                    </div>
                </div>

                <div className="footer-disclaimer">
                    <div className="disclaimer-box">
                        <strong>⚠️ Medical Disclaimer:</strong> This application is for educational and demonstration purposes only.
                        It should NOT be used for actual medical diagnosis. Always consult with qualified healthcare professionals
                        for medical advice, diagnosis, or treatment.
                    </div>
                </div>

                <div className="footer-bottom">
                    <p className="footer-copyright">
                        © {currentYear} Healthcare AI. Made with <FaHeart className="heart-icon" /> for better healthcare.
                    </p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
