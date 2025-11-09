"""
Linear Regression Model for AQI Prediction
"""
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegressionModel:
    """Linear Regression model wrapper"""
    
    def __init__(self, model_dir: Path):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.model_dir = model_dir
        self.model_path = model_dir / 'linear_regression_model.pkl'
        self.scaler_path = model_dir / 'linear_regression_scaler.pkl'
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Training metrics dict
        """
        try:
            logger.info("Training Linear Regression model...")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Get training predictions
            y_train_pred = self.model.predict(X_train_scaled)
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'r2': r2_score(y_train, y_train_pred)
            }
            
            self.is_trained = True
            logger.info(f"Linear Regression trained - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training Linear Regression: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dict
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Scale features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Predict
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
            logger.info(f"Linear Regression evaluated - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating Linear Regression: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Ensure predictions are in valid AQI range
            predictions = np.clip(predictions, 0, 500)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save(self):
        """Save model and scaler to disk"""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Linear Regression model saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self):
        """Load model and scaler from disk"""
        try:
            if not self.model_path.exists() or not self.scaler_path.exists():
                raise FileNotFoundError("Model files not found")
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"Linear Regression model loaded from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
