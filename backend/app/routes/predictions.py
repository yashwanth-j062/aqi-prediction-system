"""
Predictions API Routes
"""
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from app.database import ScopedSession
from app.models.db_models import City, AirQualityData, Prediction
from app.services.predictor import Predictor
from app.services.data_processor import DataProcessor
from app.config import Config
import logging

logger = logging.getLogger(__name__)

predictions_bp = Blueprint('predictions', __name__)


@predictions_bp.route('/predictions/<city_name>', methods=['GET'])
def get_predictions(city_name: str):
    """Get AQI predictions for a city"""
    try:
        # Get query parameters
        model_type = request.args.get('model', 'xgboost')
        hours_ahead = int(request.args.get('hours', 24))
        
        # Validate parameters
        if hours_ahead < 1 or hours_ahead > 168:  # Max 7 days
            return jsonify({
                'success': False,
                'error': 'hours must be between 1 and 168'
            }), 400
        
        # Make predictions
        predictor = Predictor()
        predictions = predictor.predict_city(city_name, model_type, hours_ahead)
        predictor.close()
        
        return jsonify({
            'success': True,
            'city': city_name,
            'model': model_type,
            'predictions': predictions
        }), 200
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@predictions_bp.route('/current/<city_name>', methods=['GET'])
def get_current_data(city_name: str):
    """Get current AQI data and prediction for a city"""
    try:
        db = ScopedSession()
        
        # Get city
        city = db.query(City).filter(City.name == city_name).first()
        if not city:
            return jsonify({
                'success': False,
                'error': f'City {city_name} not found'
            }), 404
        
        # Get latest air quality data
        latest_data = db.query(AirQualityData).filter(
            AirQualityData.city_id == city.id
        ).order_by(AirQualityData.timestamp.desc()).first()
        
        if not latest_data:
            return jsonify({
                'success': False,
                'error': f'No data available for {city_name}'
            }), 404
        
        # Get latest prediction
        latest_prediction = db.query(Prediction).filter(
            Prediction.city_id == city.id
        ).order_by(Prediction.prediction_timestamp.desc()).first()
        
        # Prepare response
        current_aqi = latest_data.aqi
        current_category = Config.get_aqi_category(current_aqi) if current_aqi else None
        health_message = Config.get_health_message(current_category) if current_category else None
        
        response_data = {
            'success': True,
            'city': {
                'name': city.name,
                'state': city.state,
                'latitude': city.latitude,
                'longitude': city.longitude
            },
            'current': {
                'timestamp': latest_data.timestamp.isoformat(),
                'aqi': round(current_aqi, 2) if current_aqi else None,
                'category': current_category,
                'health_message': health_message,
                'pollutants': {
                    'pm25': round(latest_data.pm25, 2) if latest_data.pm25 else None,
                    'pm10': round(latest_data.pm10, 2) if latest_data.pm10 else None,
                    'no2': round(latest_data.no2, 2) if latest_data.no2 else None,
                    'so2': round(latest_data.so2, 2) if latest_data.so2 else None,
                    'co': round(latest_data.co, 2) if latest_data.co else None,
                    'o3': round(latest_data.o3, 2) if latest_data.o3 else None
                },
                'weather': {
                    'temperature': round(latest_data.temperature, 1) if latest_data.temperature else None,
                    'humidity': round(latest_data.humidity, 1) if latest_data.humidity else None,
                    'pressure': round(latest_data.pressure, 1) if latest_data.pressure else None,
                    'wind_speed': round(latest_data.wind_speed, 1) if latest_data.wind_speed else None
                }
            }
        }
        
        # Add prediction if available
        if latest_prediction:
            response_data['prediction'] = {
                'predicted_aqi': round(latest_prediction.predicted_aqi, 2),
                'predicted_category': latest_prediction.predicted_category,
                'target_timestamp': latest_prediction.target_timestamp.isoformat(),
                'model_type': latest_prediction.model_type,
                'confidence_score': latest_prediction.confidence_score
            }
        
        db.close()
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error getting current data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@predictions_bp.route('/historical/<city_name>', methods=['GET'])
def get_historical_data(city_name: str):
    """Get historical AQI data for a city"""
    try:
        # Get query parameters
        days = int(request.args.get('days', 7))
        
        if days < 1 or days > 90:
            return jsonify({
                'success': False,
                'error': 'days must be between 1 and 90'
            }), 400
        
        db = ScopedSession()
        
        # Get city
        city = db.query(City).filter(City.name == city_name).first()
        if not city:
            return jsonify({
                'success': False,
                'error': f'City {city_name} not found'
            }), 404
        
        # Get historical data
        start_date = datetime.utcnow() - timedelta(days=days)
        historical_data = db.query(AirQualityData).filter(
            AirQualityData.city_id == city.id,
            AirQualityData.timestamp >= start_date
        ).order_by(AirQualityData.timestamp).all()
        
        # Format data
        data_list = []
        for record in historical_data:
            data_list.append({
                'timestamp': record.timestamp.isoformat(),
                'aqi': round(record.aqi, 2) if record.aqi else None,
                'pm25': round(record.pm25, 2) if record.pm25 else None,
                'pm10': round(record.pm10, 2) if record.pm10 else None,
                'temperature': round(record.temperature, 1) if record.temperature else None,
                'humidity': round(record.humidity, 1) if record.humidity else None
            })
        
        db.close()
        
        return jsonify({
            'success': True,
            'city': city_name,
            'days': days,
            'count': len(data_list),
            'data': data_list
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
