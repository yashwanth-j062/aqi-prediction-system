"""
Data Processor Service - Processes and prepares data for ML models
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from app.config import Config
from app.database import ScopedSession
from app.models.db_models import City, AirQualityData
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Service for processing air quality data"""
    
    def __init__(self):
        self.db = ScopedSession()
    
    def get_city_data(self, city_name: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical data for a city
        
        Args:
            city_name: Name of the city
            days: Number of days of historical data
            
        Returns:
            DataFrame with processed data
        """
        try:
            city = self.db.query(City).filter(City.name == city_name).first()
            
            if not city:
                logger.error(f"City {city_name} not found")
                return None
            
            # Get data from last N days
            start_date = datetime.utcnow() - timedelta(days=days)
            
            data = self.db.query(AirQualityData).filter(
                AirQualityData.city_id == city.id,
                AirQualityData.timestamp >= start_date
            ).order_by(AirQualityData.timestamp).all()
            
            if not data:
                logger.warning(f"No data found for {city_name}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': record.timestamp,
                'aqi': record.aqi,
                'pm25': record.pm25,
                'pm10': record.pm10,
                'no2': record.no2,
                'so2': record.so2,
                'co': record.co,
                'o3': record.o3,
                'temperature': record.temperature,
                'humidity': record.humidity,
                'pressure': record.pressure,
                'wind_speed': record.wind_speed,
                'wind_direction': record.wind_direction,
                'data_source': record.data_source
            } for record in data])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting city data: {e}")
            return None
    
    def get_all_cities_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get data for all cities combined
        
        Args:
            days: Number of days of historical data
            
        Returns:
            DataFrame with all cities data
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            data = self.db.query(
                AirQualityData,
                City.name.label('city_name'),
                City.state.label('state')
            ).join(City).filter(
                AirQualityData.timestamp >= start_date
            ).order_by(AirQualityData.timestamp).all()
            
            if not data:
                logger.warning("No data found")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'city_name': record.city_name,
                'state': record.state,
                'timestamp': record.AirQualityData.timestamp,
                'aqi': record.AirQualityData.aqi,
                'pm25': record.AirQualityData.pm25,
                'pm10': record.AirQualityData.pm10,
                'no2': record.AirQualityData.no2,
                'so2': record.AirQualityData.so2,
                'co': record.AirQualityData.co,
                'o3': record.AirQualityData.o3,
                'temperature': record.AirQualityData.temperature,
                'humidity': record.AirQualityData.humidity,
                'pressure': record.AirQualityData.pressure,
                'wind_speed': record.AirQualityData.wind_speed,
                'wind_direction': record.AirQualityData.wind_direction
            } for record in data])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting all cities data: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Make a copy
            df_clean = df.copy()
            
            # Handle missing values
            numeric_columns = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
                             'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
            
            # Fill missing values with forward fill, then backward fill
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            
            # Remove rows with critical missing values (AQI)
            df_clean = df_clean.dropna(subset=['aqi'])
            
            # Remove outliers (optional - using IQR method)
            for col in ['aqi', 'pm25', 'pm10']:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for loose outlier detection
                    upper_bound = Q3 + 3 * IQR
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Data cleaned: {len(df)} -> {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df
    
    def aggregate_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to hourly intervals
        
        Args:
            df: Input DataFrame
            
        Returns:
            Hourly aggregated DataFrame
        """
        try:
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy = df_copy.set_index('timestamp')
            
            # Resample to hourly and take mean
            numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
            df_hourly = df_copy[numeric_cols].resample('H').mean()
            
            # Reset index
            df_hourly = df_hourly.reset_index()
            
            logger.info(f"Data aggregated to hourly: {len(df_hourly)} records")
            return df_hourly
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return df
    
    def get_data_statistics(self, city_name: Optional[str] = None) -> dict:
        """
        Get statistics about available data
        
        Args:
            city_name: Optional city name to filter
            
        Returns:
            Dict with statistics
        """
        try:
            query = self.db.query(
                func.count(AirQualityData.id).label('total_records'),
                func.min(AirQualityData.timestamp).label('oldest_record'),
                func.max(AirQualityData.timestamp).label('latest_record'),
                func.avg(AirQualityData.aqi).label('avg_aqi')
            )
            
            if city_name:
                city = self.db.query(City).filter(City.name == city_name).first()
                if city:
                    query = query.filter(AirQualityData.city_id == city.id)
            
            result = query.first()
            
            stats = {
                'total_records': result.total_records or 0,
                'oldest_record': result.oldest_record.isoformat() if result.oldest_record else None,
                'latest_record': result.latest_record.isoformat() if result.latest_record else None,
                'avg_aqi': round(result.avg_aqi, 2) if result.avg_aqi else None,
                'city': city_name or 'all'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def prepare_training_data(self, city_name: str, days: int = 30) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare data for model training
        
        Args:
            city_name: Name of the city
            days: Number of days of historical data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Get data
            df = self.get_city_data(city_name, days)
            
            if df is None or len(df) < Config.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient data for {city_name}: {len(df) if df is not None else 0} samples")
                return None
            
            # Clean data
            df = self.clean_data(df)
            
            # Aggregate to hourly
            df = self.aggregate_hourly(df)
            
            # Check if we still have enough data
            if len(df) < Config.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient data after processing for {city_name}")
                return None
            
            # Feature columns
            feature_columns = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
                             'temperature', 'humidity', 'pressure', 'wind_speed']
            
            # Remove rows with missing features
            df = df.dropna(subset=feature_columns + ['aqi'])
            
            X = df[feature_columns]
            y = df['aqi']
            
            logger.info(f"Training data prepared: {len(X)} samples, {len(feature_columns)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        self.db.close()
