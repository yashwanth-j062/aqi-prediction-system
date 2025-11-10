"""
Model Trainer Service - Trains and evaluates all ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

from app.config import Config
from app.database import ScopedSession
from app.models.db_models import ModelMetrics
from app.services.data_processor import DataProcessor
from app.utils.feature_engineering import FeatureEngineer
from app.utils.model_selector import ModelSelector
from app.ml_models.linear_regression import LinearRegressionModel
from app.ml_models.random_forest import RandomForestModel
from app.ml_models.xgboost_model import XGBoostModel
# Make LSTM optional
try:
    from app.ml_models.lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Service for training all ML models"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.models_dir = Config.MODELS_DIR
        self.db = ScopedSession()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering pipeline
        
        Args:
            df: Raw dataframe
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Applying feature engineering...")
        
        # Create time features
        df = self.feature_engineer.create_time_features(df)
        
        # Create lag features (only for most important pollutants)
        lag_columns = ['pm25', 'pm10', 'no2', 'aqi']
        df = self.feature_engineer.create_lag_features(df, lag_columns, lags=[1, 3, 6])
        
        # Create rolling features
        rolling_columns = ['pm25', 'pm10', 'temperature', 'humidity']
        df = self.feature_engineer.create_rolling_features(df, rolling_columns, windows=[3, 6, 12])
        
        # Create interaction features
        df = self.feature_engineer.create_interaction_features(df)
        
        # Drop rows with NaN created by lag/rolling
        df = df.dropna()
        
        logger.info(f"Feature engineering complete: {df.shape[1]} features")
        return df
    
    def train_all_models(self, city_name: str, days: int = 30) -> dict:
        """
        Train all models for a specific city
        
        Args:
            city_name: Name of the city
            days: Days of historical data to use
            
        Returns:
            Dict with training results
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING MODELS FOR {city_name.upper()}")
            logger.info(f"{'='*60}\n")
            
            # Get data
            df = self.data_processor.get_city_data(city_name, days)
            
            if df is None or len(df) < Config.MIN_TRAINING_SAMPLES:
                raise ValueError(f"Insufficient data for {city_name}: {len(df) if df is not None else 0} samples")
            
            # Clean data
            df = self.data_processor.clean_data(df)
            
            # Apply feature engineering
            df = self.prepare_features(df)
            
            # Prepare features and target
            X, y = self.feature_engineer.select_features(df, target_col='aqi')
            
            logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=Config.TEST_SIZE, 
                random_state=Config.RANDOM_STATE,
                shuffle=False  # Preserve time order
            )
            
            logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")
            
            # Train all models
            model_results = {}
            
            # 1. Linear Regression
            logger.info("1. Training Linear Regression...")
            lr_model = LinearRegressionModel(Config.MODELS_DIR / 'linear_regression')
            lr_model.train(X_train.values, y_train.values)
            lr_metrics = lr_model.evaluate(X_test.values, y_test.values)
            lr_model.save()
            model_results['linear_regression'] = lr_metrics
            
            # 2. Random Forest
            logger.info("\n2. Training Random Forest...")
            rf_model = RandomForestModel(Config.MODELS_DIR / 'random_forest')
            rf_model.train(X_train.values, y_train.values)
            rf_metrics = rf_model.evaluate(X_test.values, y_test.values)
            rf_model.save()
            model_results['random_forest'] = rf_metrics
            
            # 3. XGBoost
            logger.info("\n3. Training XGBoost...")
            xgb_model = XGBoostModel(Config.MODELS_DIR / 'xgboost')
            xgb_model.train(X_train.values, y_train.values)
            xgb_metrics = xgb_model.evaluate(X_test.values, y_test.values)
            xgb_model.save()
            model_results['xgboost'] = xgb_metrics
            
            # 4. LSTM (if enough sequential data)
            if len(df) >= 100:  # LSTM needs more data
                try:
                    logger.info("\n4. Training LSTM...")
                    
                    # Prepare sequential data for LSTM
                    sequence_length = 24  # 24 hours lookback
                    lstm_data = np.column_stack([X.values, y.values])
                    
                    X_lstm, y_lstm = self.feature_engineer.prepare_lstm_data(
                        lstm_data, 
                        sequence_length=sequence_length
                    )
                    
                    # Split LSTM data
                    split_idx = int(len(X_lstm) * (1 - Config.TEST_SIZE))
                    X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
                    y_lstm_train, y_lstm_test = y_lstm[:split_idx], y_lstm[split_idx:]
                    
                    lstm_model = LSTMModel(
                        Config.MODELS_DIR / 'lstm',
                        sequence_length=sequence_length,
                        n_features=X.shape[1]
                    )
                    
                    lstm_model.train(X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test, epochs=30)
                    lstm_metrics = lstm_model.evaluate(X_lstm_test, y_lstm_test)
                    lstm_model.save()
                    model_results['lstm'] = lstm_metrics
                    
                except Exception as e:
                    logger.warning(f"LSTM training failed: {e}")
                    logger.info("Continuing with other models...")
            else:
                logger.info("\n4. Skipping LSTM (insufficient data)")
            
            # Select best model
            logger.info("\n")
            best_model_name, best_metrics = ModelSelector.select_best_model(model_results)
            
            # Save metrics to database
            self._save_model_metrics(model_results, X.columns.tolist(), len(X_train), len(X_test))
            
            results = {
                'city': city_name,
                'best_model': best_model_name,
                'best_metrics': best_metrics,
                'all_models': model_results,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X.shape[1]
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING COMPLETE FOR {city_name.upper()}")
            logger.info(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
        finally:
            self.data_processor.close()
    
    def _save_model_metrics(self, model_results: dict, features: list, train_samples: int, test_samples: int):
        """Save model metrics to database"""
        try:
            for model_name, metrics in model_results.items():
                model_metric = ModelMetrics(
                    model_type=model_name,
                    model_version='1.0',
                    training_date=datetime.utcnow(),
                    mae=metrics.get('mae'),
                    rmse=metrics.get('rmse'),
                    r2_score=metrics.get('r2'),
                    mape=metrics.get('mape'),
                    training_samples=train_samples,
                    test_samples=test_samples,
                    features_used=features,
                    hyperparameters={},
                    is_active=True
                )
                self.db.add(model_metric)
            
            self.db.commit()
            logger.info("✓ Model metrics saved to database")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            self.db.rollback()
    
    def close(self):
        """Close database connection"""
        self.db.close()
        self.data_processor.close()
