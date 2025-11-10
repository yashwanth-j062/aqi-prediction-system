"""
Predictor Service - Makes predictions using trained models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from app.config import Config
from app.database import ScopedSession
from app.models.db_models import City, Prediction
from app.services.data_processor import DataProcessor
from app.utils.feature_engineering import FeatureEngineer
from app.ml_models.linear_regression import LinearRegressionModel
from app.ml_models.random_forest import RandomForestModel
from app.ml_models.xgboost_model import XGBoostModel
from app.ml_models.lstm_model import LSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Service for making AQI predictions"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.db = ScopedSession()
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Linear Regression
            try:
                lr_model = LinearRegressionModel(Config.MODELS_DIR / 'linear_regression')
                lr_model.load()
                self.models['linear_regression'] = lr_model
                logger.info("✓ Linear Regression model loaded")
            except Exception as e:
                logger.warning(f"Could not load Linear Regression: {e}")
            
            # Random Forest
            try:
                rf_model = RandomForestModel(Config.MODELS_DIR / 'random_forest')
                rf_model.load()
                self.models['random_forest'] = rf_model
                logger.info("✓ Random Forest model loaded")
            except Exception as e:
                logger.warning(f"Could not load Random Forest: {e}")
            
            # XGBoost
            try:
                xgb_model = XGBoostModel(Config.MODELS_DIR / 'xgboost')
                xgb_model.load()
                self.models['xgboost'] = xgb_model
                logger.info("✓ XGBoost model loaded")
            except Exception as e:
                logger.warning(f"Could not load XGBoost: {e}")
            
            # LSTM
            try:
                lstm_model = LSTMModel(Config.MODELS_DIR / 'lstm')
                lstm_model.load()
                self.models['lstm'] = lstm_model
                logger.info("✓ LSTM model loaded")
            except Exception as e:
                logger.warning(f"Could not load LSTM: {e}")
            
            if not self.models:
                raise ValueError("No models could be loaded. Please train models first.")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_city(self, city_name: str, model_type: str = 'xgboost', hours_ahead: int = 24) -> List[Dict]:
        """
        Predict AQI for a city for next N hours
        
        Args:
            city_name: Name of the city
            model_type: Model to use ('linear_regression', 'random_forest', 'xgboost', 'lstm')
            hours_ahead: Number of hours to predict
            
        Returns:
            List of prediction dicts
        """
        try:
            city = self.db.query(City).filter(City.name == city_name).first()
            if not city:
                raise ValueError(f"City {city_name} not found")
            
            # Get model
            model = self.models.get(model_type)
            if not model:
                raise ValueError(f"Model {model_type} not available")
            
            # Get recent data for feature engineering
            df = self.data_processor.get_city_data(city_name, days=7)
            
            if df is None or len(df) < 48:  # Need at least 2 days of data
                raise ValueError(f"Insufficient data for prediction: {len(df) if df is not None else 0} samples")
            
            # Clean and prepare features
            df = self.data_processor.clean_data(df)
            df = self._prepare_prediction_features(df)
            
            # Get latest data point for prediction
            X, _ = self.feature_engineer.select_features(df, target_col='aqi')
            latest_features = X.tail(1).values
            
            # Make prediction for next hour
            prediction = model.predict(latest_features)[0]
            
            # Get current timestamp
            current_time = datetime.utcnow()
            
            # Create predictions list
            predictions = []
            
            for hour in range(1, hours_ahead + 1):
                target_time = current_time + timedelta(hours=hour)
                
                # For simplicity, use same prediction (in real scenario, would use recursive prediction)
                pred_dict = {
                    'city_id': city.id,
                    'city_name': city_name,
                    'predicted_aqi': round(float(prediction), 2),
                    'predicted_category': Config.get_aqi_category(prediction),
                    'target_timestamp': target_time,
                    'model_type': model_type,
                    'confidence_score': 0.85  # Placeholder
                }
                
                predictions.append(pred_dict)
            
            # Save predictions to database
            self._save_predictions(predictions, model_type)
            
            logger.info(f"✓ Generated {len(predictions)} predictions for {city_name}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting for {city_name}: {e}")
            raise
    
    def predict_current_aqi(self, city_name: str, model_type: str = 'xgboost') -> Dict:
        """
        Predict current AQI based on latest available data
        
        Args:
            city_name: Name of the city
            model_type: Model to use
            
        Returns:
            Prediction dict
        """
        try:
            predictions = self.predict_city(city_name, model_type, hours_ahead=1)
            return predictions[0] if predictions else None
            
        except Exception as e:
            logger.error(f"Error predicting current AQI: {e}")
            raise
    
    def _prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same feature engineering as training"""
        df = self.feature_engineer.create_time_features(df)
        
        lag_columns = ['pm25', 'pm10', 'no2', 'aqi']
        df = self.feature_engineer.create_lag_features(df, lag_columns, lags=[1, 3, 6])
        
        rolling_columns = ['pm25', 'pm10', 'temperature', 'humidity']
        df = self.feature_engineer.create_rolling_features(df, rolling_columns, windows=[3, 6, 12])
        
        df = self.feature_engineer.create_interaction_features(df)
        df = df.dropna()
        
        return df
    
    def _save_predictions(self, predictions: List[Dict], model_type: str):
        """Save predictions to database"""
        try:
            for pred in predictions:
                prediction = Prediction(
                    city_id=pred['city_id'],
                    predicted_aqi=pred['predicted_aqi'],
                    prediction_timestamp=datetime.utcnow(),
                    target_timestamp=pred['target_timestamp'],
                    model_type=model_type,
                    predicted_category=pred['predicted_category'],
                    confidence_score=pred['confidence_score']
                )
                self.db.add(prediction)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            self.db.rollback()
    
    def get_best_model(self) -> str:
        """Get name of best performing model"""
        # Could query database for best metrics
        # For now, return xgboost as default
        return 'xgboost' if 'xgboost' in self.models else list(self.models.keys())[0]
    
    def close(self):
        """Close connections"""
        self.db.close()
        self.data_processor.close()
