"""
Data Fetcher Service - Collects air quality data from OpenWeather and IQAir APIs
"""
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from app.config import Config
from app.database import ScopedSession
from app.models.db_models import City, AirQualityData, DataFetchLog
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Service for fetching air quality data from external APIs"""
    
    def __init__(self):
        self.openweather_api_key = Config.OPENWEATHER_API_KEY
        self.iqair_api_key = Config.IQAIR_API_KEY
        self.openweather_base_url = Config.OPENWEATHER_BASE_URL
        self.iqair_base_url = Config.IQAIR_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AQI-Prediction-System/1.0'})
    
    def fetch_openweather_data(self, city: City) -> Optional[Dict]:
        """
        Fetch air quality and weather data from OpenWeather API
        
        API Docs: https://openweathermap.org/api/air-pollution
        """
        try:
            # Air Pollution API endpoint
            air_pollution_url = f"{self.openweather_base_url}/air_pollution"
            weather_url = f"{self.openweather_base_url}/weather"
            
            # Fetch air pollution data
            air_params = {
                'lat': city.latitude,
                'lon': city.longitude,
                'appid': self.openweather_api_key
            }
            
            air_response = self.session.get(air_pollution_url, params=air_params, timeout=10)
            air_response.raise_for_status()
            air_data = air_response.json()
            
            # Fetch weather data
            weather_params = {
                'lat': city.latitude,
                'lon': city.longitude,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            weather_response = self.session.get(weather_url, params=weather_params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Parse data
            if air_data.get('list') and weather_data:
                air_info = air_data['list'][0]
                components = air_info.get('components', {})
                weather_main = weather_data.get('main', {})
                wind = weather_data.get('wind', {})
                
                return {
                    'city_id': city.id,
                    'timestamp': datetime.utcfromtimestamp(air_info.get('dt', time.time())),
                    'aqi': air_info.get('main', {}).get('aqi'),
                    'pm25': components.get('pm2_5'),
                    'pm10': components.get('pm10'),
                    'no2': components.get('no2'),
                    'so2': components.get('so2'),
                    'co': components.get('co'),
                    'o3': components.get('o3'),
                    'temperature': weather_main.get('temp'),
                    'humidity': weather_main.get('humidity'),
                    'pressure': weather_main.get('pressure'),
                    'wind_speed': wind.get('speed'),
                    'wind_direction': wind.get('deg'),
                    'data_source': 'openweather'
                }
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenWeather API error for {city.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing OpenWeather data for {city.name}: {e}")
            return None
    
    def fetch_iqair_data(self, city: City) -> Optional[Dict]:
        """
        Fetch air quality data from IQAir API
        
        API Docs: https://www.iqair.com/air-pollution-data-api
        """
        try:
            # Nearest city endpoint
            url = f"{self.iqair_base_url}/nearest_city"
            
            params = {
                'lat': city.latitude,
                'lon': city.longitude,
                'key': self.iqair_api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success' and data.get('data'):
                current = data['data'].get('current', {})
                pollution = current.get('pollution', {})
                weather = current.get('weather', {})
                
                # IQAir uses US AQI, we'll store it directly
                return {
                    'city_id': city.id,
                    'timestamp': datetime.fromisoformat(pollution.get('ts', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
                    'aqi': pollution.get('aqius'),  # US AQI
                    'pm25': pollution.get('p2', {}).get('conc') if isinstance(pollution.get('p2'), dict) else None,
                    'pm10': pollution.get('p1', {}).get('conc') if isinstance(pollution.get('p1'), dict) else None,
                    'temperature': weather.get('tp'),
                    'humidity': weather.get('hu'),
                    'pressure': weather.get('pr'),
                    'wind_speed': weather.get('ws'),
                    'wind_direction': weather.get('wd'),
                    'data_source': 'iqair'
                }
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"IQAir API error for {city.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing IQAir data for {city.name}: {e}")
            return None
    
    def convert_aqi_to_indian_standard(self, aqi: Optional[float], source: str) -> Optional[float]:
        """
        Convert AQI from different standards to Indian AQI
        OpenWeather uses European AQI (1-5 scale)
        IQAir uses US AQI (0-500 scale)
        Indian AQI is also 0-500 scale
        """
        if aqi is None:
            return None
        
        try:
            if source == 'openweather':
                # European AQI to Indian AQI conversion (approximate)
                conversion_map = {1: 50, 2: 100, 3: 200, 4: 300, 5: 400}
                return conversion_map.get(int(aqi), aqi * 100)
            elif source == 'iqair':
                # US AQI is similar to Indian AQI, minimal conversion needed
                return aqi
            else:
                return aqi
        except Exception as e:
            logger.error(f"Error converting AQI: {e}")
            return aqi
    
    def fetch_all_cities(self) -> Dict[str, int]:
        """
        Fetch data for all active cities from both APIs
        
        Returns:
            Dict with statistics: {'total_cities', 'successful', 'failed', 'records_created'}
        """
        db = ScopedSession()
        start_time = time.time()
        
        stats = {
            'total_cities': 0,
            'successful': 0,
            'failed': 0,
            'records_created': 0,
            'openweather_success': 0,
            'iqair_success': 0
        }
        
        try:
            # Get all active cities
            cities = db.query(City).filter(City.is_active == True).all()
            stats['total_cities'] = len(cities)
            
            logger.info(f"Starting data fetch for {stats['total_cities']} cities")
            
            for city in cities:
                city_success = False
                
                # Try OpenWeather API first
                openweather_data = self.fetch_openweather_data(city)
                if openweather_data:
                    # Convert AQI to Indian standard
                    openweather_data['aqi'] = self.convert_aqi_to_indian_standard(
                        openweather_data['aqi'], 
                        'openweather'
                    )
                    
                    # Save to database
                    air_quality_record = AirQualityData(**openweather_data)
                    db.add(air_quality_record)
                    stats['records_created'] += 1
                    stats['openweather_success'] += 1
                    city_success = True
                    logger.info(f"✓ OpenWeather data saved for {city.name}")
                
                # Try IQAir API (with rate limiting delay)
                time.sleep(1)  # Respect API rate limits
                
                iqair_data = self.fetch_iqair_data(city)
                if iqair_data:
                    # Convert AQI to Indian standard
                    iqair_data['aqi'] = self.convert_aqi_to_indian_standard(
                        iqair_data['aqi'], 
                        'iqair'
                    )
                    
                    # Save to database
                    air_quality_record = AirQualityData(**iqair_data)
                    db.add(air_quality_record)
                    stats['records_created'] += 1
                    stats['iqair_success'] += 1
                    city_success = True
                    logger.info(f"✓ IQAir data saved for {city.name}")
                
                # Update stats
                if city_success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                    logger.warning(f"✗ No data fetched for {city.name}")
                
                # Commit every 10 cities to avoid memory issues
                if stats['successful'] % 10 == 0:
                    db.commit()
            
            # Final commit
            db.commit()
            
            # Log fetch operation
            duration = time.time() - start_time
            fetch_log = DataFetchLog(
                fetch_timestamp=datetime.utcnow(),
                data_source='openweather+iqair',
                cities_fetched=stats['successful'],
                records_created=stats['records_created'],
                status='success' if stats['failed'] == 0 else 'partial',
                duration_seconds=duration
            )
            db.add(fetch_log)
            db.commit()
            
            logger.info(f"Data fetch complete: {stats['records_created']} records created in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in fetch_all_cities: {e}")
            db.rollback()
        finally:
            db.close()
        
        return stats
    
    def fetch_single_city(self, city_name: str) -> bool:
        """
        Fetch data for a single city
        
        Args:
            city_name: Name of the city
            
        Returns:
            bool: True if successful
        """
        db = ScopedSession()
        
        try:
            city = db.query(City).filter(City.name == city_name).first()
            
            if not city:
                logger.error(f"City {city_name} not found")
                return False
            
            success = False
            
            # Fetch from OpenWeather
            openweather_data = self.fetch_openweather_data(city)
            if openweather_data:
                openweather_data['aqi'] = self.convert_aqi_to_indian_standard(
                    openweather_data['aqi'], 
                    'openweather'
                )
                air_quality_record = AirQualityData(**openweather_data)
                db.add(air_quality_record)
                success = True
            
            # Fetch from IQAir
            time.sleep(1)
            iqair_data = self.fetch_iqair_data(city)
            if iqair_data:
                iqair_data['aqi'] = self.convert_aqi_to_indian_standard(
                    iqair_data['aqi'], 
                    'iqair'
                )
                air_quality_record = AirQualityData(**iqair_data)
                db.add(air_quality_record)
                success = True
            
            if success:
                db.commit()
                logger.info(f"✓ Data fetched successfully for {city_name}")
            else:
                logger.warning(f"✗ No data fetched for {city_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error fetching data for {city_name}: {e}")
            db.rollback()
            return False
        finally:
            db.close()
