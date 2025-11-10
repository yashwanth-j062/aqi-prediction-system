import React from 'react';
import { getAQIStyle } from '../utils/constants';
import { roundNumber } from '../utils/helpers';

function MetricsCard({ title, value, unit, category, description, isPrimary }) {
  const aqiStyle = isPrimary && category ? getAQIStyle(value) : null;

  return (
    <div 
      className={`metrics-card ${isPrimary ? 'primary-card' : ''}`}
      style={aqiStyle ? { 
        borderLeft: `4px solid ${aqiStyle.color}`,
        backgroundColor: aqiStyle.bgColor 
      } : {}}
    >
      <div className="card-header">
        <h4>{title}</h4>
        {description && <p className="card-description">{description}</p>}
      </div>
      
      <div className="card-value">
        <span className="value" style={aqiStyle ? { color: aqiStyle.textColor } : {}}>
          {roundNumber(value)}
        </span>
        {unit && <span className="unit">{unit}</span>}
      </div>

      {isPrimary && category && (
        <div 
          className="category-badge"
          style={{ 
            backgroundColor: aqiStyle.color,
            color: '#fff'
          }}
        >
          {category}
        </div>
      )}
    </div>
  );
}

export default MetricsCard;
