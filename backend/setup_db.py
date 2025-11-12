"""
Database setup script - loads initial data
Run this once after deployment
"""
from app.database import init_db, ScopedSession
from app.models.db_models import City
import json
import sys

def load_cities():
    """Load cities from JSON into database"""
    try:
        print("Initializing database...")
        init_db()
        
        print("Loading cities...")
        with open('data/cities.json', 'r') as f:
            cities_data = json.load(f)
        
        db = ScopedSession()
        
        # Check if already loaded
        existing_count = db.query(City).count()
        if existing_count > 0:
            print(f"Database already has {existing_count} cities. Skipping.")
            db.close()
            return
        
        # Load cities
        for city_data in cities_data['cities']:
            city = City(
                name=city_data['name'],
                state=city_data['state'],
                latitude=city_data['lat'],
                longitude=city_data['lon']
            )
            db.add(city)
        
        db.commit()
        total = db.query(City).count()
        print(f"✓ Successfully loaded {total} cities!")
        
        # Show first 5
        sample = db.query(City).limit(5).all()
        print("\nSample cities:")
        for city in sample:
            print(f"  - {city.name}, {city.state}")
        
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    load_cities()
