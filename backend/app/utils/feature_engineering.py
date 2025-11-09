"""
Feature Engineering utilities for ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for AQI prediction"""
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with additional time features
        """
        df_copy = df.copy()
        
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            # Extract time components
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
            df_copy['day_of_month'] = df_copy['timestamp'].dt.day
            df_copy['month'] = df_copy['timestamp'].dt.month
            df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding for time features
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
            df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
            df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
            
            logger.info("Time features created")
        
        return df_copy
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods (in hours)
            
        Returns:
            DataFrame with lag features
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                for lag in lags:
                    df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        logger.info(f"Lag features created for {len(columns)} columns")
        return df_copy
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: list, windows: list = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame
            columns: Columns to compute rolling stats for
            windows: List of window sizes (in hours)
            
        Returns:
            DataFrame with rolling features
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                for window in windows:
                    # Rolling mean
                    df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window, min_periods=1).mean()
                    # Rolling std
                    df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window, min_periods=1).std()
                    # Rolling max
                    df_copy[f'{col}_rolling_max_{window}'] = df_copy[col].rolling(window=window, min_periods=1).max()
                    # Rolling min
                    df_copy[f'{col}_rolling_min_{window}'] = df_copy[col].rolling(window=window, min_periods=1).min()
        
        logger.info(f"Rolling features created for {len(columns)} columns")
        return df_copy
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between pollutants and weather
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_copy = df.copy()
        
        # PM2.5 and humidity interaction
        if 'pm25' in df_copy.columns and 'humidity' in df_copy.columns:
            df_copy['pm25_humidity'] = df_copy['pm25'] * df_copy['humidity']
        
        # PM10 and wind speed interaction
        if 'pm10' in df_copy.columns and 'wind_speed' in df_copy.columns:
            df_copy['pm10_wind'] = df_copy['pm10'] * df_copy['wind_speed']
        
        # Temperature and humidity interaction
        if 'temperature' in df_copy.columns and 'humidity' in df_copy.columns:
            df_copy['temp_humidity'] = df_copy['temperature'] * df_copy['humidity']
        
        # Pollutant ratios
        if 'pm25' in df_copy.columns and 'pm10' in df_copy.columns:
            df_copy['pm25_pm10_ratio'] = df_copy['pm25'] / (df_copy['pm10'] + 1)  # +1 to avoid division by zero
        
        logger.info("Interaction features created")
        return df_copy
    
    @staticmethod
    def select_features(df: pd.DataFrame, target_col: str = 'aqi') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select and prepare final feature set
        
        Args:
            df: Input DataFrame with all features
            target_col: Name of target column
            
        Returns:
            Tuple of (X features, y target)
        """
        # Drop non-numeric and target columns
        exclude_cols = ['timestamp', 'city_name', 'state', 'data_source', target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Drop rows with NaN values
        valid_indices = X.notna().all(axis=1) & y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Feature selection: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y
    
    @staticmethod
    def prepare_lstm_data(data: np.ndarray, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model (sequences)
        
        Args:
            data: Input data array (features + target in last column)
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length, :-1])  # All features except last column
            y.append(data[i+sequence_length, -1])  # Target (last column)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"LSTM data prepared: X shape {X.shape}, y shape {y.shape}")
        return X, y
