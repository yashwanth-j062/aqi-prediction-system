"""
Model Selector - Selects best performing model
"""
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects best model based on evaluation metrics"""
    
    @staticmethod
    def select_best_model(model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Select best model based on RMSE (lower is better)
        
        Args:
            model_results: Dict mapping model_name -> metrics dict
            
        Returns:
            Tuple of (best_model_name, best_metrics)
        """
        if not model_results:
            raise ValueError("No model results provided")
        
        best_model = None
        best_rmse = float('inf')
        best_metrics = None
        
        logger.info("\nModel Comparison:")
        logger.info("=" * 60)
        
        for model_name, metrics in model_results.items():
            rmse = metrics.get('rmse', float('inf'))
            mae = metrics.get('mae', 0)
            r2 = metrics.get('r2', 0)
            
            logger.info(f"{model_name:20s} - RMSE: {rmse:6.2f}, MAE: {mae:6.2f}, R²: {r2:6.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
                best_metrics = metrics
        
        logger.info("=" * 60)
        logger.info(f"✓ Best Model: {best_model} (RMSE: {best_rmse:.2f})")
        
        return best_model, best_metrics
    
    @staticmethod
    def rank_models(model_results: Dict[str, Dict]) -> list:
        """
        Rank all models by performance
        
        Args:
            model_results: Dict mapping model_name -> metrics dict
            
        Returns:
            List of (model_name, metrics) sorted by RMSE
        """
        ranked = sorted(
            model_results.items(),
            key=lambda x: x[1].get('rmse', float('inf'))
        )
        return ranked
