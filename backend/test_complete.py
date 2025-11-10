"""
Complete backend system test
"""
import time
from app.services.data_fetcher import DataFetcher
from app.services.model_trainer import ModelTrainer
from app.services.predictor import Predictor

def main():
    print("\n" + "="*60)
    print("COMPLETE BACKEND SYSTEM TEST")
    print("="*60)
    
    # Step 1: Fetch data for test city
    print("\n1. FETCHING DATA FOR DELHI")
    print("-"*60)
    fetcher = DataFetcher()
    success = fetcher.fetch_single_city("Delhi")
    
    if not success:
        print("✗ Data fetch failed. Please check API keys.")
        return
    
    print("✓ Data fetch successful")
    time.sleep(2)
    
    # Fetch for a few more cities to have more training data
    print("\n2. FETCHING DATA FOR ADDITIONAL CITIES")
    print("-"*60)
    for city in ["Mumbai", "Bangalore"]:
        print(f"Fetching {city}...")
        fetcher.fetch_single_city(city)
        time.sleep(2)
    
    # Step 2: Train models
    print("\n3. TRAINING ML MODELS")
    print("-"*60)
    trainer = ModelTrainer()
    
    try:
        results = trainer.train_all_models("Delhi", days=7)
        print(f"\n✓ Training complete!")
        print(f"Best model: {results['best_model']}")
        print(f"Test RMSE: {results['best_metrics']['rmse']:.2f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("Note: You may need more historical data. Try fetching data multiple times.")
        return
    finally:
        trainer.close()
    
    # Step 3: Make predictions
    print("\n4. MAKING PREDICTIONS")
    print("-"*60)
    predictor = Predictor()
    
    try:
        predictions = predictor.predict_city("Delhi", model_type='xgboost', hours_ahead=5)
        print(f"\n✓ Generated {len(predictions)} predictions")
        print("\nSample predictions:")
        for pred in predictions[:3]:
            print(f"  {pred['target_timestamp']}: AQI {pred['predicted_aqi']} ({pred['predicted_category']})")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
    finally:
        predictor.close()
    
    print("\n" + "="*60)
    print("SYSTEM TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python app/main.py")
    print("2. Test API: http://localhost:5000/api/health")
    print("3. Get predictions: http://localhost:5000/api/current/Delhi")

if __name__ == "__main__":
    main()
