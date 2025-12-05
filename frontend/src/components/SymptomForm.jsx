import React, { useState, useEffect } from 'react';
import { FaSearch, FaTimes, FaCheckCircle } from 'react-icons/fa';
import { getSymptomsList } from '../utils/api';
import './SymptomForm.css';

const SymptomForm = ({ onSubmit, loading }) => {
    const [availableSymptoms, setAvailableSymptoms] = useState([]);
    const [selectedSymptoms, setSelectedSymptoms] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [showDropdown, setShowDropdown] = useState(false);

    useEffect(() => {
        loadSymptoms();
    }, []);

    const loadSymptoms = async () => {
        try {
            const data = await getSymptomsList();
            setAvailableSymptoms(data.symptoms || []);
        } catch (error) {
            console.error('Failed to load symptoms:', error);
            // Fallback symptoms if API fails
            setAvailableSymptoms([
                'fever', 'cough', 'fatigue', 'difficulty_breathing', 'headache',
                'sore_throat', 'runny_nose', 'body_ache', 'nausea', 'vomiting',
                'diarrhea', 'loss_of_taste', 'loss_of_smell', 'chest_pain',
                'chills', 'muscle_pain', 'joint_pain', 'dizziness', 'weakness'
            ]);
        }
    };

    const filteredSymptoms = availableSymptoms.filter(symptom =>
        symptom.toLowerCase().includes(searchTerm.toLowerCase()) &&
        !selectedSymptoms.includes(symptom)
    );

    const handleSymptomSelect = (symptom) => {
        if (!selectedSymptoms.includes(symptom)) {
            setSelectedSymptoms([...selectedSymptoms, symptom]);
            setSearchTerm('');
            setShowDropdown(false);
        }
    };

    const handleSymptomRemove = (symptom) => {
        setSelectedSymptoms(selectedSymptoms.filter(s => s !== symptom));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (selectedSymptoms.length > 0) {
            onSubmit(selectedSymptoms);
        }
    };

    const formatSymptomName = (symptom) => {
        return symptom.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };

    return (
        <form className="symptom-form" onSubmit={handleSubmit}>
            <div className="form-header">
                <h3>Select Your Symptoms</h3>
                <p className="form-description">
                    Search and select symptoms you're experiencing. Select at least one symptom to get predictions.
                </p>
            </div>

            <div className="search-container">
                <div className="search-input-wrapper">
                    <FaSearch className="search-icon" />
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Search symptoms (e.g., fever, cough)..."
                        value={searchTerm}
                        onChange={(e) => {
                            setSearchTerm(e.target.value);
                            setShowDropdown(true);
                        }}
                        onFocus={() => setShowDropdown(true)}
                    />
                </div>

                {showDropdown && searchTerm && filteredSymptoms.length > 0 && (
                    <div className="symptoms-dropdown">
                        {filteredSymptoms.slice(0, 10).map((symptom, index) => (
                            <div
                                key={index}
                                className="symptom-option"
                                onClick={() => handleSymptomSelect(symptom)}
                            >
                                <FaCheckCircle className="option-icon" />
                                <span>{formatSymptomName(symptom)}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {selectedSymptoms.length > 0 && (
                <div className="selected-symptoms">
                    <h4 className="selected-title">Selected Symptoms ({selectedSymptoms.length})</h4>
                    <div className="symptoms-grid">
                        {selectedSymptoms.map((symptom, index) => (
                            <div key={index} className="symptom-chip">
                                <span>{formatSymptomName(symptom)}</span>
                                <button
                                    type="button"
                                    className="remove-btn"
                                    onClick={() => handleSymptomRemove(symptom)}
                                    aria-label={`Remove ${symptom}`}
                                >
                                    <FaTimes />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <button
                type="submit"
                className="btn btn-primary submit-btn"
                disabled={selectedSymptoms.length === 0 || loading}
            >
                {loading ? (
                    <>
                        <div className="spinner"></div>
                        <span>Analyzing...</span>
                    </>
                ) : (
                    <>
                        <FaCheckCircle />
                        <span>Predict Disease</span>
                    </>
                )}
            </button>
        </form>
    );
};

export default SymptomForm;
