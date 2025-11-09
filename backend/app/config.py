import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///aqi_data.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    IQAIR_API_KEY = os.getenv('IQAIR_API_KEY')
    
    # API Endpoints
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    IQAIR_BASE_URL = "https://api.airvisual.com/v2"
    
    # Scheduler Settings
    DATA_FETCH_INTERVAL_HOURS = int(os.getenv('DATA_FETCH_INTERVAL_HOURS', 1))
    MODEL_RETRAIN_INTERVAL_DAYS = int(os.getenv('MODEL_RETRAIN_INTERVAL_DAYS', 7))
    
    # Model Configuration
    MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', 1000))
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    
    # File Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    CITIES_FILE = DATA_DIR / 'cities.json'
    MODELS_DIR = BASE_DIR / 'saved_models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # CORS Settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
    
    # AQI Thresholds (Based on Indian AQI standards)
    AQI_CATEGORIES = {
        'Good': (0, 50),
        'Satisfactory': (51, 100),
        'Moderate': (101, 200),
        'Poor': (201, 300),
        'Very Poor': (301, 400),
        'Severe': (401, 500)
    }
    
    # Health Messages
    HEALTH_MESSAGES = {
        'Good': 'Air quality is satisfactory, and air pollution poses little or no risk.',
        'Satisfactory': 'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.',
        'Moderate': 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.',
        'Poor': 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.',
        'Very Poor': 'Health alert: The risk of health effects is increased for everyone.',
        'Severe': 'Health warning of emergency conditions: everyone is more likely to be affected.'
    }
    
    @staticmethod
    def validate_config():
        """Validate required configuration"""
        if not Config.OPENWEATHER_API_KEY:
            raise ValueError("OPENWEATHER_API_KEY is not set in environment variables")
        if not Config.IQAIR_API_KEY:
            raise ValueError("IQAIR_API_KEY is not set in environment variables")
        
        # Create directories if they don't exist
        for directory in [Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR, 
                         Config.MODELS_DIR, Config.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        return True

    @staticmethod
    def get_aqi_category(aqi_value):
        """Get AQI category based on value"""
        for category, (min_val, max_val) in Config.AQI_CATEGORIES.items():
            if min_val <= aqi_value <= max_val:
                return category
        return 'Severe' if aqi_value > 500 else 'Unknown'
    
    @staticmethod
    def get_health_message(category):
        """Get health message for AQI category"""
        return Config.HEALTH_MESSAGES.get(category, 'No data available')
