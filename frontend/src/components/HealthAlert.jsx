import React from 'react';
import { getAQIStyle } from '../utils/constants';

function HealthAlert({ category, message }) {
  const aqiStyle = getAQIStyle(category);

  const getIcon = () => {
    switch(category) {
      case 'Good':
      case 'Satisfactory':
        return '✅';
      case 'Moderate':
        return '⚠️';
      case 'Poor':
      case 'Very Poor':
        return '🚨';
      case 'Severe':
        return '☠️';
      default:
        return 'ℹ️';
    }
  };

  return (
    <div 
      className="health-alert"
      style={{ 
        borderLeftColor: aqiStyle.color,
        backgroundColor: aqiStyle.bgColor 
      }}
    >
      <div className="alert-icon">{getIcon()}</div>
      <div className="alert-content">
        <h3 style={{ color: aqiStyle.textColor }}>
          Air Quality: {category}
        </h3>
        <p>{message}</p>
      </div>
    </div>
  );
}

export default HealthAlert;
