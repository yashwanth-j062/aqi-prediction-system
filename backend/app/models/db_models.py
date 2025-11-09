from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class City(Base):
    """Model for storing city information"""
    __tablename__ = 'cities'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    state = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    air_quality_data = relationship("AirQualityData", back_populates="city", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="city", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<City(name='{self.name}', state='{self.state}')>"


class AirQualityData(Base):
    """Model for storing raw air quality data"""
    __tablename__ = 'air_quality_data'
    
    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False, index=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # AQI Data
    aqi = Column(Float)
    
    # Pollutants (in µg/m³)
    pm25 = Column(Float)  # PM2.5
    pm10 = Column(Float)  # PM10
    no2 = Column(Float)   # Nitrogen Dioxide
    so2 = Column(Float)   # Sulfur Dioxide
    co = Column(Float)    # Carbon Monoxide
    o3 = Column(Float)    # Ozone
    
    # Weather Parameters
    temperature = Column(Float)  # in Celsius
    humidity = Column(Float)     # in percentage
    pressure = Column(Float)     # in hPa
    wind_speed = Column(Float)   # in m/s
    wind_direction = Column(Float)  # in degrees
    
    # Data Source
    data_source = Column(String(50))  # 'openweather' or 'iqair'
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    city = relationship("City", back_populates="air_quality_data")
    
    def __repr__(self):
        return f"<AirQualityData(city_id={self.city_id}, aqi={self.aqi}, timestamp={self.timestamp})>"


class Prediction(Base):
    """Model for storing AQI predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False, index=True)
    
    # Prediction Details
    predicted_aqi = Column(Float, nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False)  # When prediction was made
    target_timestamp = Column(DateTime, nullable=False, index=True)  # Time being predicted
    
    # Model Information
    model_type = Column(String(50), nullable=False)  # 'linear_regression', 'random_forest', 'xgboost', 'lstm'
    model_version = Column(String(50))
    confidence_score = Column(Float)  # Model confidence (0-1)
    
    # Additional Predictions
    predicted_pm25 = Column(Float)
    predicted_pm10 = Column(Float)
    predicted_category = Column(String(50))  # 'Good', 'Moderate', etc.
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    city = relationship("City", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(city_id={self.city_id}, aqi={self.predicted_aqi}, model={self.model_type})>"


class ModelMetrics(Base):
    """Model for storing ML model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model Details
    model_type = Column(String(50), nullable=False)
    model_version = Column(String(50))
    training_date = Column(DateTime, nullable=False)
    
    # Performance Metrics
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Squared Error
    r2_score = Column(Float)  # R² Score
    mape = Column(Float)  # Mean Absolute Percentage Error
    
    # Training Details
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    features_used = Column(JSON)  # List of features
    hyperparameters = Column(JSON)  # Model hyperparameters
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_type}, rmse={self.rmse}, r2={self.r2_score})>"


class DataFetchLog(Base):
    """Model for logging data fetch operations"""
    __tablename__ = 'data_fetch_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Fetch Details
    fetch_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    data_source = Column(String(50), nullable=False)  # 'openweather' or 'iqair'
    cities_fetched = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), nullable=False)  # 'success', 'partial', 'failed'
    error_message = Column(String(500))
    
    # Metadata
    duration_seconds = Column(Float)
    
    def __repr__(self):
        return f"<DataFetchLog(source={self.data_source}, status={self.status}, records={self.records_created})>"
