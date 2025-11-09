"""
Test script to verify configuration and database setup
"""
from app.config import Config
from app.database import init_db, engine
from app.models.db_models import City, AirQualityData, Prediction
import json

def test_config():
    """Test configuration loading"""
    print("=" * 50)
    print("TESTING CONFIGURATION")
    print("=" * 50)
    
    try:
        Config.validate_config()
        print("✓ Configuration validated successfully")
        print(f"✓ Database URL: {Config.DATABASE_URL}")
        print(f"✓ OpenWeather API Key: {'Set' if Config.OPENWEATHER_API_KEY else 'NOT SET'}")
        print(f"✓ IQAir API Key: {'Set' if Config.IQAIR_API_KEY else 'NOT SET'}")
        print(f"✓ Data fetch interval: {Config.DATA_FETCH_INTERVAL_HOURS} hours")
        print(f"✓ Cities file: {Config.CITIES_FILE}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_database():
    """Test database initialization"""
    print("\n" + "=" * 50)
    print("TESTING DATABASE")
    print("=" * 50)
    
    try:
        init_db()
        print("✓ Database tables created successfully")
        
        # Check tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"✓ Created tables: {', '.join(tables)}")
        return True
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False

def load_cities():
    """Load cities from JSON into database"""
    print("\n" + "=" * 50)
    print("LOADING CITIES")
    print("=" * 50)
    
    try:
        from app.database import ScopedSession
        
        # Read cities.json
        with open(Config.CITIES_FILE, 'r') as f:
            cities_data = json.load(f)
        
        db = ScopedSession()
        
        # Check if cities already exist
        existing_count = db.query(City).count()
        if existing_count > 0:
            print(f"✓ Database already has {existing_count} cities")
            db.close()
            return True
        
        # Add cities
        for city_data in cities_data['cities']:
            city = City(
                name=city_data['name'],
                state=city_data['state'],
                latitude=city_data['lat'],
                longitude=city_data['lon']
            )
            db.add(city)
        
        db.commit()
        total_cities = db.query(City).count()
        print(f"✓ Loaded {total_cities} cities into database")
        
        # Show first 5 cities
        sample_cities = db.query(City).limit(5).all()
        print("\nSample cities:")
        for city in sample_cities:
            print(f"  - {city.name}, {city.state} ({city.latitude}, {city.longitude})")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"✗ Error loading cities: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("BACKEND SETUP TEST")
    print("=" * 50)
    
    # Run tests
    config_ok = test_config()
    db_ok = test_database() if config_ok else False
    cities_ok = load_cities() if db_ok else False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Configuration: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"Database: {'✓ PASS' if db_ok else '✗ FAIL'}")
    print(f"Cities: {'✓ PASS' if cities_ok else '✗ FAIL'}")
    
    if config_ok and db_ok and cities_ok:
        print("\n✓ All tests passed! Backend setup is complete.")
    else:
        print("\n✗ Some tests failed. Check errors above.")
