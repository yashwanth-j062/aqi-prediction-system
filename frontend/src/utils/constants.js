// AQI Categories with colors
export const AQI_CATEGORIES = {
  'Good': {
    range: [0, 50],
    color: '#00E400',
    bgColor: '#E8F5E9',
    textColor: '#1B5E20'
  },
  'Satisfactory': {
    range: [51, 100],
    color: '#FFFF00',
    bgColor: '#FFFDE7',
    textColor: '#F57F17'
  },
  'Moderate': {
    range: [101, 200],
    color: '#FF7E00',
    bgColor: '#FFF3E0',
    textColor: '#E65100'
  },
  'Poor': {
    range: [201, 300],
    color: '#FF0000',
    bgColor: '#FFEBEE',
    textColor: '#B71C1C'
  },
  'Very Poor': {
    range: [301, 400],
    color: '#8F3F97',
    bgColor: '#F3E5F5',
    textColor: '#4A148C'
  },
  'Severe': {
    range: [401, 500],
    color: '#7E0023',
    bgColor: '#FCE4EC',
    textColor: '#880E4F'
  }
};

// Get category style based on AQI value
export function getAQIStyle(aqi) {
  if (aqi === null || aqi === undefined) return AQI_CATEGORIES['Good'];
  
  for (const [category, data] of Object.entries(AQI_CATEGORIES)) {
    const [min, max] = data.range;
    if (aqi >= min && aqi <= max) {
      return { ...data, category };
    }
  }
  
  return { ...AQI_CATEGORIES['Severe'], category: 'Severe' };
}

// Refresh interval (5 minutes)
export const REFRESH_INTERVAL = 300000;
