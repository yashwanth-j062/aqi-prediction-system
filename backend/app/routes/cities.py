"""
Cities API Routes
"""
from flask import Blueprint, jsonify
from app.database import ScopedSession
from app.models.db_models import City
from app.models.schemas import CityResponse
import logging

logger = logging.getLogger(__name__)

cities_bp = Blueprint('cities', __name__)


@cities_bp.route('/cities', methods=['GET'])
def get_all_cities():
    """Get list of all cities"""
    try:
        db = ScopedSession()
        cities = db.query(City).filter(City.is_active == True).order_by(City.name).all()
        
        cities_list = [
            CityResponse.model_validate(city).model_dump()
            for city in cities
        ]
        
        db.close()
        
        return jsonify({
            'success': True,
            'count': len(cities_list),
            'cities': cities_list
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching cities: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cities_bp.route('/cities/<city_name>', methods=['GET'])
def get_city(city_name: str):
    """Get specific city details"""
    try:
        db = ScopedSession()
        city = db.query(City).filter(City.name == city_name).first()
        
        if not city:
            return jsonify({
                'success': False,
                'error': f'City {city_name} not found'
            }), 404
        
        city_data = CityResponse.model_validate(city).model_dump()
        db.close()
        
        return jsonify({
            'success': True,
            'city': city_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching city: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
