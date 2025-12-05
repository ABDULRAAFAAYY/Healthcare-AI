import os
import numpy as np
from .model_loader import ModelLoader

class SymptomPredictor:
    """Handles symptom-based disease prediction"""
    
    def __init__(self):
        """Initialize the predictor and load models"""
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_path = os.path.join(self.base_path, 'models')
        
        # Load models
        self.model = ModelLoader.load_pickle_model(
            os.path.join(self.models_path, 'symptom_model.pkl')
        )
        self.symptom_encoder = ModelLoader.load_pickle_model(
            os.path.join(self.models_path, 'symptom_encoder.pkl')
        )
        self.label_encoder = ModelLoader.load_pickle_model(
            os.path.join(self.models_path, 'label_encoder.pkl')
        )
        
        # Fallback symptom list for demo purposes
        self.demo_symptoms = [
            'fever', 'cough', 'fatigue', 'difficulty_breathing', 'headache',
            'sore_throat', 'runny_nose', 'body_ache', 'nausea', 'vomiting',
            'diarrhea', 'loss_of_taste', 'loss_of_smell', 'chest_pain',
            'chills', 'muscle_pain', 'joint_pain', 'dizziness', 'weakness',
            'sweating', 'rash', 'abdominal_pain', 'back_pain', 'sneezing',
            'watery_eyes', 'congestion', 'shortness_of_breath', 'wheezing',
            'rapid_heartbeat', 'confusion', 'anxiety', 'insomnia', 'appetite_loss',
            'high_fever', 'severe_headache', 'pain_behind_eyes', 'severe_body_ache',
            'bleeding_gums', 'night_sweats', 'weight_loss', 'persistent_cough',
            'jaundice', 'dark_urine', 'pale_stool', 'stiff_neck', 'sensitivity_to_light',
            'swollen_lymph_nodes', 'swollen_glands', 'swollen_tonsils', 'difficulty_swallowing',
            'conjunctivitis', 'red_eyes', 'itchy_eyes', 'itchy_nose', 'dehydration',
            'stomach_pain', 'constipation', 'loss_of_appetite', 'coughing_blood',
            'mild_fever', 'mild_headache', 'severe_joint_pain'
        ]
        
        # Demo diseases with descriptions
        self.demo_diseases = {
            'Common Cold': {
                'description': 'A viral infection of the upper respiratory tract',
                'severity': 'Mild',
                'recommendations': [
                    'Get plenty of rest',
                    'Stay hydrated',
                    'Use over-the-counter cold medications',
                    'Consult a doctor if symptoms persist beyond 10 days'
                ]
            },
            'Influenza (Flu)': {
                'description': 'A contagious respiratory illness caused by influenza viruses',
                'severity': 'Moderate',
                'recommendations': [
                    'Rest and stay home',
                    'Drink plenty of fluids',
                    'Consider antiviral medications if prescribed',
                    'Seek medical attention if symptoms worsen'
                ]
            },
            'COVID-19': {
                'description': 'Respiratory illness caused by SARS-CoV-2 virus',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Isolate immediately',
                    'Get tested for COVID-19',
                    'Monitor oxygen levels',
                    'Seek immediate medical care if breathing difficulty occurs'
                ]
            },
            'Allergic Rhinitis': {
                'description': 'Inflammation of the nasal passages due to allergens',
                'severity': 'Mild',
                'recommendations': [
                    'Avoid known allergens',
                    'Use antihistamines',
                    'Consider nasal sprays',
                    'Consult an allergist for long-term management'
                ]
            },
            'Bronchitis': {
                'description': 'Inflammation of the bronchial tubes',
                'severity': 'Moderate',
                'recommendations': [
                    'Rest and drink fluids',
                    'Use a humidifier',
                    'Avoid smoke and irritants',
                    'See a doctor if symptoms persist or worsen'
                ]
            },
            'Dengue Fever': {
                'description': 'Mosquito-borne viral infection causing high fever and severe body pain',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Seek immediate medical attention',
                    'Stay hydrated with oral rehydration solutions',
                    'Take paracetamol for fever (avoid aspirin)',
                    'Monitor for warning signs: severe abdominal pain, bleeding',
                    'Rest in mosquito-free environment'
                ]
            },
            'Dengue Hemorrhagic Fever': {
                'description': 'Severe form of dengue with bleeding complications',
                'severity': 'Severe',
                'recommendations': [
                    'URGENT: Seek emergency medical care immediately',
                    'Hospitalization required',
                    'Monitor platelet count and vital signs',
                    'IV fluids may be necessary',
                    'Watch for signs of shock'
                ]
            },
            'Malaria': {
                'description': 'Parasitic infection transmitted by mosquitoes',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Seek medical attention immediately',
                    'Get blood test for malaria parasites',
                    'Start antimalarial medication as prescribed',
                    'Stay hydrated',
                    'Use mosquito nets and repellents'
                ]
            },
            'Typhoid': {
                'description': 'Bacterial infection causing prolonged fever',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Consult doctor for antibiotic treatment',
                    'Maintain strict hygiene',
                    'Drink only boiled or purified water',
                    'Complete full course of antibiotics',
                    'Get adequate rest and nutrition'
                ]
            },
            'Meningitis': {
                'description': 'Inflammation of membranes surrounding brain and spinal cord',
                'severity': 'Severe',
                'recommendations': [
                    'URGENT: Seek emergency medical care immediately',
                    'This is a medical emergency',
                    'Hospitalization required',
                    'IV antibiotics or antivirals needed',
                    'Isolate to prevent spread'
                ]
            },
            'Pneumonia': {
                'description': 'Infection causing inflammation of air sacs in lungs',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'See a doctor for proper diagnosis',
                    'Take prescribed antibiotics if bacterial',
                    'Get plenty of rest',
                    'Stay hydrated',
                    'Seek emergency care if breathing worsens'
                ]
            },
            'Tuberculosis': {
                'description': 'Bacterial infection primarily affecting the lungs',
                'severity': 'Severe',
                'recommendations': [
                    'Seek immediate medical consultation',
                    'Get chest X-ray and sputum test',
                    'Start TB treatment regimen (6-9 months)',
                    'Isolate to prevent transmission',
                    'Take all medications as prescribed'
                ]
            },
            'Hepatitis': {
                'description': 'Inflammation of the liver, often caused by viral infection',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Consult a hepatologist or gastroenterologist',
                    'Get liver function tests',
                    'Avoid alcohol completely',
                    'Rest and maintain good nutrition',
                    'Get vaccinated if not already done'
                ]
            },
            'Gastroenteritis': {
                'description': 'Inflammation of stomach and intestines',
                'severity': 'Mild to Moderate',
                'recommendations': [
                    'Stay hydrated with ORS or clear fluids',
                    'Eat bland, easy-to-digest foods',
                    'Avoid dairy and fatty foods',
                    'Maintain hand hygiene',
                    'See doctor if symptoms persist beyond 3 days'
                ]
            },
            'Strep Throat': {
                'description': 'Bacterial infection of the throat',
                'severity': 'Mild to Moderate',
                'recommendations': [
                    'See doctor for throat culture',
                    'Take prescribed antibiotics',
                    'Gargle with warm salt water',
                    'Rest your voice',
                    'Complete full antibiotic course'
                ]
            },
            'Measles': {
                'description': 'Highly contagious viral infection',
                'severity': 'Moderate',
                'recommendations': [
                    'Isolate to prevent spread',
                    'Get plenty of rest',
                    'Stay hydrated',
                    'Use fever reducers as needed',
                    'Seek medical care for complications'
                ]
            },
            'Mononucleosis': {
                'description': 'Viral infection causing extreme fatigue',
                'severity': 'Moderate',
                'recommendations': [
                    'Get adequate rest (may need weeks)',
                    'Stay hydrated',
                    'Avoid contact sports (risk of spleen rupture)',
                    'Take pain relievers for sore throat',
                    'Avoid sharing utensils or drinks'
                ]
            },
            'Chikungunya': {
                'description': 'Mosquito-borne viral disease causing severe joint pain',
                'severity': 'Moderate',
                'recommendations': [
                    'Rest and stay hydrated',
                    'Take pain relievers for joint pain',
                    'Use mosquito protection',
                    'Joint pain may persist for months',
                    'Consult doctor for persistent symptoms'
                ]
            },
            'Zika Virus': {
                'description': 'Mosquito-borne viral infection',
                'severity': 'Mild to Moderate',
                'recommendations': [
                    'Rest and stay hydrated',
                    'Use mosquito repellent',
                    'Pregnant women should seek immediate medical care',
                    'Avoid mosquito bites',
                    'Most cases are mild and self-limiting'
                ]
            },
            'Tonsillitis': {
                'description': 'Inflammation of the tonsils',
                'severity': 'Mild to Moderate',
                'recommendations': [
                    'See doctor to determine if bacterial or viral',
                    'Take antibiotics if bacterial',
                    'Gargle with warm salt water',
                    'Rest and stay hydrated',
                    'Use throat lozenges for relief'
                ]
            }
        }
    
    def is_ready(self):
        """Check if the predictor is ready to make predictions"""
        return True
    
    def get_available_symptoms(self):
        """Get list of available symptoms"""
        if self.symptom_encoder is not None:
            try:
                return list(self.symptom_encoder.classes_)
            except:
                pass
        return self.demo_symptoms
    
    def predict(self, symptoms):
        """Predict disease based on symptoms"""
        try:
            if self.model is not None and self.symptom_encoder is not None:
                return self._predict_with_model(symptoms)
            else:
                return self._demo_predict(symptoms)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return self._demo_predict(symptoms)
    
    def _predict_with_model(self, symptoms):
        """Make prediction using trained model"""
        feature_vector = np.zeros(len(self.symptom_encoder.classes_))
        
        for symptom in symptoms:
            symptom_clean = symptom.lower().strip().replace(' ', '_')
            try:
                idx = list(self.symptom_encoder.classes_).index(symptom_clean)
                feature_vector[idx] = 1
            except ValueError:
                print(f"Warning: Symptom '{symptom}' not found in encoder")
        
        feature_vector = feature_vector.reshape(1, -1)
        probabilities = self.model.predict_proba(feature_vector)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        predictions = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            
            disease_info = self.demo_diseases.get(disease, {
                'description': 'Medical condition',
                'severity': 'Unknown',
                'recommendations': ['Consult a healthcare professional']
            })
            
            predictions.append({
                'disease': disease,
                'confidence': confidence,
                'description': disease_info['description'],
                'severity': disease_info['severity'],
                'recommendations': disease_info['recommendations']
            })
        
        return {
            'success': True,
            'predictions': predictions,
            'input_symptoms': symptoms,
            'model_type': 'trained'
        }
    
    def _demo_predict(self, symptoms):
        """Fallback demo prediction when models aren't available"""
        symptom_set = set(s.lower().replace(' ', '_') for s in symptoms)
        
        # Define symptom patterns for each disease
        disease_patterns = {
            'COVID-19': {'fever', 'cough', 'loss_of_taste', 'loss_of_smell', 'difficulty_breathing'},
            'Influenza (Flu)': {'fever', 'body_ache', 'fatigue', 'cough', 'headache', 'chills'},
            'Common Cold': {'runny_nose', 'sneezing', 'sore_throat', 'cough', 'congestion'},
            'Allergic Rhinitis': {'sneezing', 'watery_eyes', 'runny_nose', 'congestion', 'itchy_eyes'},
            'Bronchitis': {'cough', 'chest_pain', 'difficulty_breathing', 'fatigue', 'wheezing'},
            'Dengue Fever': {'high_fever', 'severe_headache', 'pain_behind_eyes', 'joint_pain', 'muscle_pain', 'rash'},
            'Dengue Hemorrhagic Fever': {'high_fever', 'severe_body_ache', 'bleeding_gums', 'severe_headache', 'rash'},
            'Malaria': {'high_fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting'},
            'Typhoid': {'high_fever', 'weakness', 'stomach_pain', 'headache', 'loss_of_appetite', 'abdominal_pain'},
            'Meningitis': {'fever', 'severe_headache', 'stiff_neck', 'nausea', 'vomiting', 'confusion', 'sensitivity_to_light'},
            'Pneumonia': {'fever', 'cough', 'chest_pain', 'difficulty_breathing', 'chills'},
            'Tuberculosis': {'persistent_cough', 'chest_pain', 'weight_loss', 'night_sweats', 'fever', 'coughing_blood'},
            'Hepatitis': {'fever', 'jaundice', 'dark_urine', 'fatigue', 'abdominal_pain', 'nausea'},
            'Gastroenteritis': {'abdominal_pain', 'diarrhea', 'nausea', 'vomiting', 'fever'},
            'Strep Throat': {'sore_throat', 'fever', 'swollen_lymph_nodes', 'difficulty_swallowing'},
            'Measles': {'fever', 'rash', 'cough', 'runny_nose', 'watery_eyes', 'conjunctivitis'},
            'Mononucleosis': {'fever', 'swollen_glands', 'fatigue', 'sore_throat'},
            'Chikungunya': {'high_fever', 'severe_joint_pain', 'muscle_pain', 'rash', 'headache'},
            'Zika Virus': {'fever', 'headache', 'muscle_pain', 'joint_pain', 'rash', 'conjunctivitis'},
            'Tonsillitis': {'fever', 'sore_throat', 'swollen_glands', 'difficulty_swallowing', 'swollen_tonsils'}
        }
        
        # Calculate scores
        scores = []
        for disease, pattern in disease_patterns.items():
            if len(pattern) > 0:
                score = len(symptom_set & pattern) / len(pattern)
                scores.append((disease, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        predictions = []
        for disease, score in scores[:3]:
            if score > 0:
                disease_info = self.demo_diseases.get(disease, {
                    'description': 'Medical condition',
                    'severity': 'Unknown',
                    'recommendations': ['Consult a healthcare professional']
                })
                predictions.append({
                    'disease': disease,
                    'confidence': min(0.95, score + 0.2),
                    'description': disease_info['description'],
                    'severity': disease_info['severity'],
                    'recommendations': disease_info['recommendations']
                })
        
        if not predictions:
            predictions.append({
                'disease': 'Unknown Condition',
                'confidence': 0.3,
                'description': 'Unable to determine specific condition',
                'severity': 'Unknown',
                'recommendations': [
                    'Consult a healthcare professional for proper diagnosis',
                    'Monitor your symptoms',
                    'Seek immediate care if symptoms worsen'
                ]
            })
        
        return {
            'success': True,
            'predictions': predictions,
            'input_symptoms': symptoms,
            'model_type': 'demo',
            'note': 'Using demo prediction logic. Train models for accurate results.'
        }