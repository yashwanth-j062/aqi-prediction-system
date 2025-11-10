import React, { useState, useEffect } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import CitySelector from './components/CitySelector';
import { getCities } from './services/api';

function App() {
  const [cities, setCities] = useState([]);
  const [selectedCity, setSelectedCity] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCities();
  }, []);

  const fetchCities = async () => {
    try {
      setLoading(true);
      const data = await getCities();
      setCities(data.cities || []);
      
      // Set default city (Delhi)
      const defaultCity = data.cities?.find(c => c.name === 'Delhi') || data.cities?.[0];
      setSelectedCity(defaultCity);
      
      setError(null);
    } catch (err) {
      console.error('Error fetching cities:', err);
      setError('Failed to load cities. Please check if backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleCityChange = (city) => {
    setSelectedCity(city);
  };

  if (loading) {
    return (
      <div className="app">
        <Navbar />
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading cities...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <Navbar />
        <div className="error-container">
          <h2>⚠️ Error</h2>
          <p>{error}</p>
          <button onClick={fetchCities} className="retry-btn">Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <Navbar />
      <div className="app-container">
        <CitySelector 
          cities={cities} 
          selectedCity={selectedCity}
          onCityChange={handleCityChange}
        />
        {selectedCity && (
          <Dashboard city={selectedCity} />
        )}
      </div>
    </div>
  );
}

export default App;
