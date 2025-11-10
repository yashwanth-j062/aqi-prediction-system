"""
Health Check & System Status Routes
"""
from flask import Blueprint, jsonify
from datetime import datetime
from app.database import ScopedSession
from app.models.db_models import City, AirQualityData, ModelMetrics
from sqlalchemy import func
import logging

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    try:
        db = ScopedSession()
        
        # Check database connection
        total_cities = db.query(func.count(City.id)).scalar()
        total_records = db.query(func.count(AirQualityData.id)).scalar()
        
        # Get latest data timestamp
        latest_data = db.query(func.max(AirQualityData.timestamp)).scalar()
        
        db.close()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'total_cities': total_cities,
            'total_records': total_records,
            'latest_data': latest_data.isoformat() if latest_data else None
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500


@health_bp.route('/models/status', methods=['GET'])
def models_status():
    """Get status of trained models"""
    try:
        db = ScopedSession()
        
        # Get latest metrics for each model
        models = db.query(ModelMetrics).filter(
            ModelMetrics.is_active == True
        ).order_by(ModelMetrics.training_date.desc()).all()
        
        models_info = []
        for model in models:
            models_info.append({
                'model_type': model.model_type,
                'training_date': model.training_date.isoformat(),
                'rmse': round(model.rmse, 2) if model.rmse else None,
                'mae': round(model.mae, 2) if model.mae else None,
                'r2_score': round(model.r2_score, 4) if model.r2_score else None,
                'training_samples': model.training_samples
            })
        
        db.close()
        
        return jsonify({
            'success': True,
            'models_count': len(models_info),
            'models': models_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
