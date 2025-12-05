import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaHeartbeat, FaStethoscope, FaXRay, FaHome } from 'react-icons/fa';
import './Navbar.css';

const Navbar = () => {
    const location = useLocation();

    const isActive = (path) => location.pathname === path;

    return (
        <nav className="navbar">
            <div className="navbar-container">
                <Link to="/" className="navbar-logo">
                    <FaHeartbeat className="logo-icon" />
                    <span className="logo-text">Healthcare<span className="logo-accent">AI</span></span>
                </Link>

                <ul className="navbar-menu">
                    <li className="navbar-item">
                        <Link
                            to="/"
                            className={`navbar-link ${isActive('/') ? 'active' : ''}`}
                        >
                            <FaHome />
                            <span>Home</span>
                        </Link>
                    </li>
                    <li className="navbar-item">
                        <Link
                            to="/disease-predictor"
                            className={`navbar-link ${isActive('/disease-predictor') ? 'active' : ''}`}
                        >
                            <FaStethoscope />
                            <span>Symptom Checker</span>
                        </Link>
                    </li>
                    <li className="navbar-item">
                        <Link
                            to="/image-predictor"
                            className={`navbar-link ${isActive('/image-predictor') ? 'active' : ''}`}
                        >
                            <FaXRay />
                            <span>X-Ray Analysis</span>
                        </Link>
                    </li>
                </ul>
            </div>
        </nav>
    );
};

export default Navbar;
