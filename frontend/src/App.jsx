import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import DiseasePredictor from './pages/DiseasePredictor';
import ImagePredictor from './pages/ImagePredictor';

function App() {
    return (
        <Router>
            <div className="app">
                <Navbar />
                <main className="main-content container">
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/disease-predictor" element={<DiseasePredictor />} />
                        <Route path="/image-predictor" element={<ImagePredictor />} />
                    </Routes>
                </main>
                <Footer />
            </div>
        </Router>
    );
}

export default App;
