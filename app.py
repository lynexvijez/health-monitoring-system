"""
AI-Powered Health Monitoring System - Python Backend
Flask REST API with Machine Learning Anomaly Detection

Requirements:
pip install flask flask-cors numpy pandas scikit-learn

Run: python app.py
API will be available at http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import threading
import time

app = Flask(__name__)
CORS(app)

# ============================================
# HEALTH DATA MODELS & STORAGE
# ============================================

class HealthDataStore:
    """In-memory storage for health data (replace with database in production)"""
    def __init__(self):
        self.data = []
        self.alerts = []
        self.user_profile = {
            'id': 'user_001',
            'name': 'John Doe',
            'age': 32,
            'height_cm': 178,
            'weight_kg': 75,
            'blood_type': 'O+',
            'resting_hr': 68,
            'device': 'Fitbit Sense 2'
        }
    
    def add_reading(self, reading):
        reading['id'] = len(self.data) + 1
        reading['timestamp'] = datetime.now().isoformat()
        self.data.append(reading)
        # Keep last 1000 readings
        if len(self.data) > 1000:
            self.data = self.data[-1000:]
        return reading
    
    def get_recent(self, count=50):
        return self.data[-count:] if self.data else []
    
    def add_alert(self, alert):
        alert['id'] = len(self.alerts) + 1
        alert['timestamp'] = datetime.now().isoformat()
        self.alerts.insert(0, alert)
        self.alerts = self.alerts[:100]  # Keep last 100 alerts
        return alert

store = HealthDataStore()

# ============================================
# AI ANOMALY DETECTION ENGINE
# ============================================

class HealthAnomalyDetector:
    """
    AI-powered anomaly detection using Isolation Forest algorithm
    and threshold-based rules for health metrics
    """
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Medical thresholds for rule-based detection
        self.thresholds = {
            'heart_rate': {
                'normal_low': 55, 'normal_high': 100,
                'critical_low': 45, 'critical_high': 120,
                'unit': 'BPM'
            },
            'blood_oxygen': {
                'normal_low': 94, 'normal_high': 100,
                'critical_low': 90, 'critical_high': 100,
                'unit': '%'
            },
            'sleep_quality': {
                'normal_low': 40, 'normal_high': 100,
                'critical_low': 25, 'critical_high': 100,
                'unit': '%'
            },
            'activity_level': {
                'normal_low': 20, 'normal_high': 100,
                'critical_low': 0, 'critical_high': 100,
                'unit': '%'
            }
        }
    
    def train(self, data):
        """Train the Isolation Forest model on historical data"""
        if len(data) < 50:
            return False
        
        df = pd.DataFrame(data)
        features = df[['heart_rate', 'blood_oxygen']].values
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        self.model.fit(scaled_features)
        self.is_trained = True
        return True
    
    def detect_anomaly_ml(self, heart_rate, blood_oxygen):
        """Use ML model for anomaly detection"""
        if not self.is_trained:
            return None
        
        features = np.array([[heart_rate, blood_oxygen]])
        scaled = self.scaler.transform(features)
        prediction = self.model.predict(scaled)[0]
        score = self.model.score_samples(scaled)[0]
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(score),
            'confidence': min(abs(score) * 100, 100)
        }
    
    def detect_anomaly_rules(self, metric_name, value):
        """Rule-based anomaly detection using medical thresholds"""
        if metric_name not in self.thresholds:
            return {'status': 'unknown', 'severity': 0, 'message': 'Unknown metric'}
        
        t = self.thresholds[metric_name]
        
        if value <= t['critical_low']:
            return {
                'status': 'critical',
                'severity': 3,
                'message': f'Critically low {metric_name.replace("_", " ")}: {value}{t["unit"]}'
            }
        elif value >= t['critical_high'] and metric_name != 'blood_oxygen':
            return {
                'status': 'critical',
                'severity': 3,
                'message': f'Critically high {metric_name.replace("_", " ")}: {value}{t["unit"]}'
            }
        elif value < t['normal_low']:
            return {
                'status': 'warning',
                'severity': 2,
                'message': f'Low {metric_name.replace("_", " ")}: {value}{t["unit"]}'
            }
        elif value > t['normal_high'] and metric_name != 'blood_oxygen':
            return {
                'status': 'warning',
                'severity': 2,
                'message': f'High {metric_name.replace("_", " ")}: {value}{t["unit"]}'
            }
        
        return {
            'status': 'normal',
            'severity': 1,
            'message': f'{metric_name.replace("_", " ").title()} is within normal range'
        }
    
    def analyze_reading(self, reading):
        """Complete analysis of a health reading"""
        results = {
            'heart_rate': self.detect_anomaly_rules('heart_rate', reading['heart_rate']),
            'blood_oxygen': self.detect_anomaly_rules('blood_oxygen', reading['blood_oxygen']),
            'sleep_quality': self.detect_anomaly_rules('sleep_quality', reading.get('sleep_quality', 70)),
            'activity_level': self.detect_anomaly_rules('activity_level', reading.get('activity_level', 50))
        }
        
        # Add ML-based detection if model is trained
        ml_result = self.detect_anomaly_ml(reading['heart_rate'], reading['blood_oxygen'])
        if ml_result:
            results['ml_detection'] = ml_result
        
        # Overall status
        severities = [r['severity'] for r in results.values() if isinstance(r, dict) and 'severity' in r]
        max_severity = max(severities) if severities else 1
        
        results['overall_status'] = {
            1: 'healthy',
            2: 'attention_needed',
            3: 'critical'
        }.get(max_severity, 'unknown')
        
        return results

detector = HealthAnomalyDetector()

# ============================================
# AI RECOMMENDATION ENGINE
# ============================================

class RecommendationEngine:
    """AI-powered personalized health recommendations"""
    
    def __init__(self):
        self.recommendations_db = {
            'heart_rate_high': {
                'type': 'warning',
                'icon': '❤️',
                'text': 'Elevated heart rate detected. Consider: deep breathing exercises, '
                       'reducing caffeine intake, staying hydrated, or taking a rest break.',
                'priority': 1
            },
            'heart_rate_low': {
                'type': 'warning',
                'icon': '❤️',
                'text': 'Low heart rate detected. If experiencing dizziness, fatigue, or '
                       'shortness of breath, please consult a healthcare provider.',
                'priority': 1
            },
            'heart_rate_normal': {
                'type': 'success',
                'icon': '❤️',
                'text': 'Heart rate is within healthy range. Your cardiovascular system '
                       'is functioning well. Keep up your healthy lifestyle!',
                'priority': 4
            },
            'blood_oxygen_low': {
                'type': 'critical',
                'icon': '🫁',
                'text': 'Blood oxygen is below normal range. Practice deep breathing exercises. '
                       'If levels persist below 94% or you feel unwell, seek medical attention.',
                'priority': 1
            },
            'blood_oxygen_normal': {
                'type': 'success',
                'icon': '🫁',
                'text': 'Blood oxygen levels are excellent. Your respiratory function is optimal.',
                'priority': 4
            },
            'sleep_quality_low': {
                'type': 'warning',
                'icon': '😴',
                'text': 'Sleep quality needs improvement. Tips: maintain consistent sleep schedule, '
                       'limit screen time 1 hour before bed, keep room cool and dark.',
                'priority': 2
            },
            'sleep_quality_normal': {
                'type': 'success',
                'icon': '😴',
                'text': 'Good sleep quality detected. Quality rest is essential for recovery '
                       'and overall health. Keep maintaining your sleep routine!',
                'priority': 4
            },
            'activity_low': {
                'type': 'info',
                'icon': '🏃',
                'text': 'Activity level is low today. Aim for at least 30 minutes of moderate '
                       'exercise. Even a short walk can improve your health metrics.',
                'priority': 3
            },
            'activity_high': {
                'type': 'success',
                'icon': '🏃',
                'text': 'Excellent activity level! Remember to stay hydrated, stretch regularly, '
                       'and allow adequate recovery time between intense workouts.',
                'priority': 4
            }
        }
    
    def generate_recommendations(self, reading, analysis):
        """Generate personalized recommendations based on health data"""
        recommendations = []
        
        # Heart rate recommendations
        hr = reading['heart_rate']
        if hr > 100:
            recommendations.append(self.recommendations_db['heart_rate_high'])
        elif hr < 55:
            recommendations.append(self.recommendations_db['heart_rate_low'])
        else:
            recommendations.append(self.recommendations_db['heart_rate_normal'])
        
        # Blood oxygen recommendations
        bo = reading['blood_oxygen']
        if bo < 94:
            recommendations.append(self.recommendations_db['blood_oxygen_low'])
        else:
            recommendations.append(self.recommendations_db['blood_oxygen_normal'])
        
        # Sleep quality recommendations
        sq = reading.get('sleep_quality', 70)
        if sq < 50:
            recommendations.append(self.recommendations_db['sleep_quality_low'])
        else:
            recommendations.append(self.recommendations_db['sleep_quality_normal'])
        
        # Activity level recommendations
        al = reading.get('activity_level', 50)
        if al < 30:
            recommendations.append(self.recommendations_db['activity_low'])
        elif al > 70:
            recommendations.append(self.recommendations_db['activity_high'])
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations

recommender = RecommendationEngine()

# ============================================
# SIMULATED DATA GENERATOR
# ============================================

def generate_health_reading(previous=None):
    """Generate realistic simulated health data"""
    def vary(base, range_val, min_val, max_val):
        return max(min_val, min(max_val, base + (np.random.random() - 0.5) * range_val))
    
    if previous:
        hr = vary(previous['heart_rate'], 8, 50, 130)
        bo = vary(previous['blood_oxygen'], 2, 88, 100)
        sq = previous.get('sleep_quality', 70)
        al = vary(previous.get('activity_level', 50), 15, 10, 100)
    else:
        hr = 70 + np.random.random() * 20
        bo = 95 + np.random.random() * 4
        sq = 60 + np.random.random() * 30
        al = 40 + np.random.random() * 40
    
    return {
        'heart_rate': round(hr),
        'blood_oxygen': round(bo, 1),
        'sleep_quality': round(sq),
        'activity_level': round(al),
        'steps': round(al * 100),
        'calories_burned': round(al * 25)
    }

# ============================================
# API ROUTES
# ============================================

@app.route('/api/health', methods=['GET'])
def get_health_data():
    """Get current health reading with analysis"""
    recent = store.get_recent(1)
    previous = recent[0] if recent else None
    
    reading = generate_health_reading(previous)
    stored = store.add_reading(reading)
    
    analysis = detector.analyze_reading(reading)
    recommendations = recommender.generate_recommendations(reading, analysis)
    
    # Create alerts for anomalies
    for metric, result in analysis.items():
        if isinstance(result, dict) and result.get('severity', 0) >= 2:
            store.add_alert({
                'metric': metric,
                'message': result.get('message', 'Anomaly detected'),
                'severity': result['severity'],
                'value': reading.get(metric.replace('_', ''))
            })
    
    return jsonify({
        'success': True,
        'data': stored,
        'analysis': analysis,
        'recommendations': recommendations
    })

@app.route('/api/health/history', methods=['GET'])
def get_health_history():
    """Get historical health data"""
    count = request.args.get('count', 50, type=int)
    data = store.get_recent(count)
    return jsonify({'success': True, 'data': data, 'count': len(data)})

@app.route('/api/health/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    return jsonify({'success': True, 'alerts': store.alerts[:20]})

@app.route('/api/health/stats', methods=['GET'])
def get_stats():
    """Get health statistics summary"""
    data = store.get_recent(100)
    if not data:
        return jsonify({'success': True, 'stats': None, 'message': 'No data available'})
    
    df = pd.DataFrame(data)
    stats = {
        'heart_rate': {
            'avg': round(df['heart_rate'].mean(), 1),
            'min': int(df['heart_rate'].min()),
            'max': int(df['heart_rate'].max()),
            'std': round(df['heart_rate'].std(), 2)
        },
        'blood_oxygen': {
            'avg': round(df['blood_oxygen'].mean(), 1),
            'min': round(df['blood_oxygen'].min(), 1),
            'max': round(df['blood_oxygen'].max(), 1)
        },
        'readings_count': len(data),
        'anomalies_count': len([a for a in store.alerts if a['severity'] >= 2])
    }
    
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/user/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    return jsonify({'success': True, 'profile': store.user_profile})

@app.route('/api/user/profile', methods=['PUT'])
def update_profile():
    """Update user profile"""
    data = request.get_json()
    store.user_profile.update(data)
    return jsonify({'success': True, 'profile': store.user_profile})

@app.route('/api/health/train', methods=['POST'])
def train_model():
    """Train the ML model on collected data"""
    data = store.get_recent(200)
    success = detector.train(data)
    return jsonify({
        'success': success,
        'message': 'Model trained successfully' if success else 'Not enough data (need 50+ readings)'
    })

@app.route('/api/health/report', methods=['GET'])
def get_report():
    """Generate health report"""
    data = store.get_recent(100)
    if len(data) < 10:
        return jsonify({'success': False, 'message': 'Not enough data for report'})
    
    df = pd.DataFrame(data)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'period': {
            'start': data[0]['timestamp'],
            'end': data[-1]['timestamp'],
            'readings': len(data)
        },
        'heart_rate': {
            'average': round(df['heart_rate'].mean(), 1),
            'range': f"{int(df['heart_rate'].min())} - {int(df['heart_rate'].max())} BPM",
            'status': 'Normal' if 55 <= df['heart_rate'].mean() <= 100 else 'Attention Needed'
        },
        'blood_oxygen': {
            'average': round(df['blood_oxygen'].mean(), 1),
            'range': f"{df['blood_oxygen'].min():.1f} - {df['blood_oxygen'].max():.1f}%",
            'status': 'Normal' if df['blood_oxygen'].mean() >= 94 else 'Attention Needed'
        },
        'alerts_summary': {
            'total': len(store.alerts),
            'critical': len([a for a in store.alerts if a['severity'] == 3]),
            'warnings': len([a for a in store.alerts if a['severity'] == 2])
        },
        'overall_health_score': calculate_health_score(df)
    }
    
    return jsonify({'success': True, 'report': report})

def calculate_health_score(df):
    """Calculate overall health score (0-100)"""
    score = 100
    
    avg_hr = df['heart_rate'].mean()
    if avg_hr < 55 or avg_hr > 100:
        score -= 15
    elif avg_hr < 60 or avg_hr > 90:
        score -= 5
    
    avg_bo = df['blood_oxygen'].mean()
    if avg_bo < 94:
        score -= 20
    elif avg_bo < 96:
        score -= 10
    
    return max(0, min(100, round(score)))

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 50)
    print("AI Health Monitoring System - Backend Server")
    print("=" * 50)
    print("Starting Flask server...")
    print("API Endpoints:")
    print("  GET  /api/health          - Get current reading")
    print("  GET  /api/health/history  - Get historical data")
    print("  GET  /api/health/alerts   - Get alerts")
    print("  GET  /api/health/stats    - Get statistics")
    print("  GET  /api/health/report   - Generate report")
    print("  GET  /api/user/profile    - Get user profile")
    print("  POST /api/health/train    - Train ML model")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)