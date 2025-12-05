import React, { useState, useRef } from 'react';
import { FaCloudUploadAlt, FaImage, FaTimes } from 'react-icons/fa';
import './ImageUpload.css';

const ImageUpload = ({ onSubmit, loading }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);

            // Create preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please select a valid image file');
        }
    };

    const handleFileInput = (e) => {
        const file = e.target.files[0];
        handleFileSelect(file);
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleRemove = () => {
        setSelectedFile(null);
        setPreview(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (selectedFile) {
            onSubmit(selectedFile);
        }
    };

    return (
        <form className="image-upload-form" onSubmit={handleSubmit}>
            <div className="form-header">
                <h3>Upload Medical Image</h3>
                <p className="form-description">
                    Upload an X-ray image for AI-powered analysis. Supported formats: JPG, PNG, JPEG
                </p>
            </div>

            {!preview ? (
                <div
                    className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <FaCloudUploadAlt className="upload-icon" />
                    <h4>Drag & Drop Image Here</h4>
                    <p>or click to browse</p>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleFileInput}
                        style={{ display: 'none' }}
                    />
                </div>
            ) : (
                <div className="preview-container">
                    <div className="preview-header">
                        <div className="preview-info">
                            <FaImage className="file-icon" />
                            <div>
                                <h4>{selectedFile.name}</h4>
                                <p>{(selectedFile.size / 1024).toFixed(2)} KB</p>
                            </div>
                        </div>
                        <button
                            type="button"
                            className="remove-preview-btn"
                            onClick={handleRemove}
                            aria-label="Remove image"
                        >
                            <FaTimes />
                        </button>
                    </div>
                    <div className="preview-image-wrapper">
                        <img src={preview} alt="Preview" className="preview-image" />
                    </div>
                </div>
            )}

            <button
                type="submit"
                className="btn btn-primary submit-btn"
                disabled={!selectedFile || loading}
            >
                {loading ? (
                    <>
                        <div className="spinner"></div>
                        <span>Analyzing Image...</span>
                    </>
                ) : (
                    <>
                        <FaImage />
                        <span>Analyze Image</span>
                    </>
                )}
            </button>
        </form>
    );
};

export default ImageUpload;
