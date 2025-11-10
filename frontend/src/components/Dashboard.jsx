import React, { useState, useEffect } from 'react';
import MetricsCard from './MetricsCard';
import AQIChart from './AQIChart';
import ForecastChart from './ForecastChart';
import HealthAlert from './HealthAlert';
import { getCurrentData, getPredictions, getHistoricalData } from '../services/api';
import { REFRESH_INTERVAL } from '../utils/constants';

function Dashboard({ city }) {
  const [currentData, setCurrentData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    if (city) {
      fetchAllData();
      
      // Set up auto-refresh
      const interval = setInterval(fetchAllData, REFRESH_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [city]);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch current data, predictions, and historical data in parallel
      const [currentRes, predictionsRes, historicalRes] = await Promise.all([
        getCurrentData(city.name),
        getPredictions(city.name, 24),
        getHistoricalData(city.name, 7)
      ]);

      setCurrentData(currentRes);
      setPredictions(predictionsRes.predictions || []);
      setHistoricalData(historicalRes.data || []);
      setLastUpdate(new Date());
      setError(null);

    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !currentData) {
    return (
      <div className="dashboard-loading">
        <div className="spinner"></div>
        <p>Loading dashboard for {city.name}...</p>
      </div>
    );
  }

  if (error && !currentData) {
    return (
      <div className="dashboard-error">
        <h3>⚠️ {error}</h3>
        <button onClick={fetchAllData} className="retry-btn">Retry</button>
      </div>
    );
  }

  const current = currentData?.current || {};
  const pollutants = current?.pollutants || {};
  const weather = current?.weather || {};

  return (
    <div className="dashboard">
      {/* Header with last update */}
      <div className="dashboard-header">
        <h2>{city.name}, {city.state}</h2>
        {lastUpdate && (
          <p className="last-update">
            Last updated: {lastUpdate.toLocaleTimeString('en-IN')}
          </p>
        )}
      </div>

      {/* Health Alert */}
      {current.category && (
        <HealthAlert 
          category={current.category}
          message={current.health_message}
        />
      )}

      {/* Main Metrics Grid */}
      <div className="metrics-grid">
        <MetricsCard
          title="Current AQI"
          value={current.aqi}
          unit=""
          category={current.category}
          isPrimary={true}
        />
        <MetricsCard
          title="PM2.5"
          value={pollutants.pm25}
          unit="µg/m³"
          description="Fine particulate matter"
        />
        <MetricsCard
          title="PM10"
          value={pollutants.pm10}
          unit="µg/m³"
          description="Coarse particulate matter"
        />
        <MetricsCard
          title="Temperature"
          value={weather.temperature}
          unit="°C"
          description="Current temperature"
        />
        <MetricsCard
          title="Humidity"
          value={weather.humidity}
          unit="%"
          description="Relative humidity"
        />
        <MetricsCard
          title="Wind Speed"
          value={weather.wind_speed}
          unit="m/s"
          description="Current wind speed"
        />
      </div>

      {/* Pollutants Details */}
      <div className="pollutants-section">
        <h3>Other Pollutants</h3>
        <div className="pollutants-grid">
          <div className="pollutant-item">
            <span className="pollutant-name">NO₂</span>
            <span className="pollutant-value">
              {pollutants.no2 !== null ? `${pollutants.no2.toFixed(2)} µg/m³` : 'N/A'}
            </span>
          </div>
          <div className="pollutant-item">
            <span className="pollutant-name">SO₂</span>
            <span className="pollutant-value">
              {pollutants.so2 !== null ? `${pollutants.so2.toFixed(2)} µg/m³` : 'N/A'}
            </span>
          </div>
          <div className="pollutant-item">
            <span className="pollutant-name">CO</span>
            <span className="pollutant-value">
              {pollutants.co !== null ? `${pollutants.co.toFixed(2)} µg/m³` : 'N/A'}
            </span>
          </div>
          <div className="pollutant-item">
            <span className="pollutant-name">O₃</span>
            <span className="pollutant-value">
              {pollutants.o3 !== null ? `${pollutants.o3.toFixed(2)} µg/m³` : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        {/* Historical AQI Chart */}
        {historicalData.length > 0 && (
          <div className="chart-container">
            <h3>7-Day AQI History</h3>
            <AQIChart data={historicalData} />
          </div>
        )}

        {/* Forecast Chart */}
        {predictions.length > 0 && (
          <div className="chart-container">
            <h3>24-Hour AQI Forecast</h3>
            <ForecastChart predictions={predictions} />
          </div>
        )}
      </div>

      {/* Refresh Button */}
      <div className="dashboard-actions">
        <button onClick={fetchAllData} className="refresh-btn" disabled={loading}>
          {loading ? 'Refreshing...' : '🔄 Refresh Data'}
        </button>
      </div>
    </div>
  );
}

export default Dashboard;
