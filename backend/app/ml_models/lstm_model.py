"""
LSTM Model for AQI Prediction (Time Series)
"""
import numpy as np
import pickle
from pathlib import Path
import logging

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will not work.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM model wrapper for time series AQI prediction"""
    
    def __init__(self, model_dir: Path, sequence_length: int = 24, n_features: int = 10):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.model_dir = model_dir
        self.model_path = model_dir / 'lstm_model.keras'
        self.scaler_path = model_dir / 'lstm_scaler.pkl'
        self.is_trained = False
    
    def _build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None, 
              epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training metrics dict
        """
        try:
            logger.info("Training LSTM model...")
            
            # Update n_features if needed
            if X_train.ndim == 3:
                self.n_features = X_train.shape[2]
            
            # Build model
            self.model = self._build_model()
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                            patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                                factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Get training predictions
            y_train_pred = self.model.predict(X_train, verbose=0).flatten()
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'r2': r2_score(y_train, y_train_pred),
                'epochs_trained': len(history.history['loss'])
            }
            
            self.is_trained = True
            logger.info(f"LSTM trained - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Evaluation metrics dict
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before evaluation")
            
            # Predict
            y_pred = self.model.predict(X_test, verbose=0).flatten()
            
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
            
            logger.info(f"LSTM evaluated - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating LSTM: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions array
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before prediction")
            
            predictions = self.model.predict(X, verbose=0).flatten()
            
            # Ensure predictions are in valid AQI range
            predictions = np.clip(predictions, 0, 500)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save(self):
        """Save model to disk"""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            if self.model is not None:
                self.model.save(self.model_path)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"LSTM model saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self):
        """Load model from disk"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError("Model file not found")
            
            self.model = keras.models.load_model(self.model_path)
            
            if self.scaler_path.exists():
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"LSTM model loaded from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
