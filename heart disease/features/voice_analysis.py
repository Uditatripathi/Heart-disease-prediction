"""
Voice Stress Analysis Module
Analyzes voice patterns to detect stress levels
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class VoiceStressAnalyzer:
    """Analyze voice to detect stress levels"""
    
    def __init__(self):
        self.stress_indicators = {}
        
    def analyze_audio_file(self, audio_path):
        """Analyze audio file for stress indicators"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # Calculate stress score
            stress_score = self._calculate_stress_score(features)
            
            return {
                'stress_score': stress_score,
                'stress_level': self._categorize_stress(stress_score),
                'features': features,
                'recommendations': self._get_stress_recommendations(stress_score)
            }
        except Exception as e:
            return {
                'error': f"Error analyzing audio: {str(e)}",
                'stress_score': None
            }
    
    def analyze_audio_array(self, audio_array, sample_rate=22050):
        """Analyze audio array for stress indicators"""
        try:
            y = np.array(audio_array)
            sr = sample_rate
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # Calculate stress score
            stress_score = self._calculate_stress_score(features)
            
            return {
                'stress_score': stress_score,
                'stress_level': self._categorize_stress(stress_score),
                'features': features,
                'recommendations': self._get_stress_recommendations(stress_score)
            }
        except Exception as e:
            return {
                'error': f"Error analyzing audio: {str(e)}",
                'stress_score': None
            }
    
    def _extract_features(self, y, sr):
        """Extract audio features"""
        features = {}
        
        # Fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['mean_pitch'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['mean_pitch'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Zero crossing rate (indicator of voice quality)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['mean_zcr'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['mean_spectral_centroid'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mean_mfcc'] = np.mean(mfccs, axis=1).tolist()
        
        # Energy
        rms = librosa.feature.rms(y=y)[0]
        features['mean_energy'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        
        # Tempo (speech rate)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Jitter (pitch variation)
        if len(pitch_values) > 1:
            jitter = np.mean(np.abs(np.diff(pitch_values)))
            features['jitter'] = jitter
        else:
            features['jitter'] = 0
        
        return features
    
    def _calculate_stress_score(self, features):
        """Calculate stress score from features"""
        score = 0
        
        # High pitch variation indicates stress
        if features['pitch_std'] > 50:
            score += 20
        elif features['pitch_std'] > 30:
            score += 10
        
        # High pitch range indicates stress
        if features['pitch_range'] > 200:
            score += 15
        elif features['pitch_range'] > 100:
            score += 8
        
        # High zero crossing rate indicates stress
        if features['mean_zcr'] > 0.1:
            score += 15
        elif features['mean_zcr'] > 0.05:
            score += 8
        
        # High spectral centroid variation indicates stress
        if features['spectral_centroid_std'] > 500:
            score += 15
        elif features['spectral_centroid_std'] > 300:
            score += 8
        
        # High energy variation indicates stress
        if features['energy_std'] > 0.1:
            score += 10
        elif features['energy_std'] > 0.05:
            score += 5
        
        # High jitter indicates stress
        if features['jitter'] > 5:
            score += 15
        elif features['jitter'] > 2:
            score += 8
        
        # Normalize to 0-100
        stress_score = min(score, 100)
        
        return stress_score
    
    def _categorize_stress(self, stress_score):
        """Categorize stress level"""
        if stress_score >= 70:
            return "High Stress"
        elif stress_score >= 40:
            return "Moderate Stress"
        else:
            return "Low Stress"
    
    def _get_stress_recommendations(self, stress_score):
        """Get recommendations based on stress level"""
        if stress_score >= 70:
            return [
                "High stress detected. Consider stress management techniques.",
                "Practice deep breathing exercises daily.",
                "Consider consulting with a mental health professional.",
                "Engage in regular physical activity to reduce stress.",
                "Ensure adequate sleep (7-9 hours per night)."
            ]
        elif stress_score >= 40:
            return [
                "Moderate stress detected. Monitor stress levels regularly.",
                "Practice mindfulness or meditation.",
                "Take regular breaks during the day.",
                "Engage in activities you enjoy.",
                "Maintain a healthy work-life balance."
            ]
        else:
            return [
                "Low stress levels detected. Maintain current stress management practices.",
                "Continue healthy lifestyle habits.",
                "Regular exercise helps maintain low stress levels."
            ]
    
    def correlate_stress_with_heart_risk(self, stress_score, base_heart_risk):
        """Correlate stress level with heart disease risk"""
        # Stress can increase heart disease risk
        stress_multiplier = 1.0
        
        if stress_score >= 70:
            stress_multiplier = 1.15  # 15% increase
        elif stress_score >= 40:
            stress_multiplier = 1.08  # 8% increase
        
        adjusted_risk = base_heart_risk * stress_multiplier
        adjusted_risk = min(adjusted_risk, 100)  # Cap at 100%
        
        return {
            'base_risk': base_heart_risk,
            'stress_score': stress_score,
            'adjusted_risk': adjusted_risk,
            'risk_increase': adjusted_risk - base_heart_risk,
            'message': f"Stress analysis indicates a {stress_multiplier*100-100:.1f}% increase in heart disease risk."
        }

