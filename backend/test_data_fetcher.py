"""
Test script for data fetcher service
"""
from app.services.data_fetcher import DataFetcher
from app.services.data_processor import DataProcessor
import time

def test_single_city():
    """Test fetching data for a single city"""
    print("=" * 50)
    print("TESTING SINGLE CITY DATA FETCH")
    print("=" * 50)
    
    fetcher = DataFetcher()
    
    # Test with Delhi
    print("\nFetching data for Delhi...")
    success = fetcher.fetch_single_city("Delhi")
    
    if success:
        print("✓ Data fetched successfully for Delhi")
    else:
        print("✗ Failed to fetch data for Delhi")
    
    return success

def test_multiple_cities():
    """Test fetching data for multiple cities"""
    print("\n" + "=" * 50)
    print("TESTING MULTIPLE CITIES DATA FETCH (First 5 cities)")
    print("=" * 50)
    
    fetcher = DataFetcher()
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"]
    
    for city in cities:
        print(f"\nFetching data for {city}...")
        success = fetcher.fetch_single_city(city)
        if success:
            print(f"✓ Success: {city}")
        else:
            print(f"✗ Failed: {city}")
        time.sleep(2)  # Respect API rate limits

def test_data_processor():
    """Test data processor"""
    print("\n" + "=" * 50)
    print("TESTING DATA PROCESSOR")
    print("=" * 50)
    
    processor = DataProcessor()
    
    # Get statistics
    stats = processor.get_data_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Oldest record: {stats.get('oldest_record', 'N/A')}")
    print(f"  Latest record: {stats.get('latest_record', 'N/A')}")
    print(f"  Average AQI: {stats.get('avg_aqi', 'N/A')}")
    
    # Get city-specific stats
    city_stats = processor.get_data_statistics("Delhi")
    print(f"\nDelhi Statistics:")
    print(f"  Total records: {city_stats.get('total_records', 0)}")
    print(f"  Average AQI: {city_stats.get('avg_aqi', 'N/A')}")
    
    processor.close()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("DATA FETCHER TEST")
    print("=" * 50)
    print("\nNote: This test requires valid API keys in .env file")
    print("Tests will fetch live data from OpenWeather and IQAir APIs")
    
    # Run tests
    test_single_city()
    time.sleep(3)
    
    # Uncomment to test multiple cities (will take ~2 minutes)
    # test_multiple_cities()
    
    test_data_processor()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
