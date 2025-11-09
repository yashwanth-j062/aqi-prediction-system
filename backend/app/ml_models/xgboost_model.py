"""
XGBoost Model for AQI Prediction
"""
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost model wrapper"""
    
    def __init__(self, model_dir: Path, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 6, random_state: int = 42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror'
        )
        self.scaler = StandardScaler()
        self.model_dir = model_dir
        self.model_path = model_dir / 'xgboost_model.pkl'
        self.scaler_path = model_dir / 'xgboost_scaler.pkl'
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training metrics dict
        """
        try:
            logger.info("Training XGBoost model...")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=10,
                    verbose=False
                )
            else:
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
            logger.info(f"XGBoost trained - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
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
            
            logger.info(f"XGBoost evaluated - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating XGBoost: {e}")
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
    
    def get_feature_importance(self, feature_names: list = None) -> dict:
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dict mapping feature names to importance scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained first")
            
            importance_dict = self.model.get_booster().get_score(importance_type='weight')
            
            if feature_names:
                # Map feature indices to names
                mapped_importance = {}
                for key, value in importance_dict.items():
                    idx = int(key.replace('f', ''))
                    if idx < len(feature_names):
                        mapped_importance[feature_names[idx]] = value
                importance_dict = dict(sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def save(self):
        """Save model and scaler to disk"""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"XGBoost model saved to {self.model_dir}")
            
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
            logger.info(f"XGBoost model loaded from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
