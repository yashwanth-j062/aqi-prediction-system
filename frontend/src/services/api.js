const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

// Helper function for API calls
async function apiCall(endpoint) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
}

// Get all cities
export async function getCities() {
  return apiCall('/cities');
}

// Get current AQI data for a city
export async function getCurrentData(cityName) {
  return apiCall(`/current/${cityName}`);
}

// Get predictions for a city
export async function getPredictions(cityName, hours = 24) {
  return apiCall(`/predictions/${cityName}?hours=${hours}`);
}

// Get historical data for a city
export async function getHistoricalData(cityName, days = 7) {
  return apiCall(`/historical/${cityName}?days=${days}`);
}

// Get system health
export async function getHealth() {
  return apiCall('/health');
}
