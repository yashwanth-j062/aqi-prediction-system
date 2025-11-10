import React, { useState } from 'react';

function CitySelector({ cities, selectedCity, onCityChange }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  const filteredCities = cities.filter(city =>
    city.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    city.state.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCitySelect = (city) => {
    onCityChange(city);
    setIsOpen(false);
    setSearchTerm('');
  };

  return (
    <div className="city-selector">
      <label className="selector-label">Select City</label>
      <div className="selector-dropdown">
        <button 
          className="selector-button"
          onClick={() => setIsOpen(!isOpen)}
        >
          <span className="selected-city">
            <strong>{selectedCity?.name || 'Select a city'}</strong>
            {selectedCity && <span className="city-state">{selectedCity.state}</span>}
          </span>
          <span className="dropdown-arrow">{isOpen ? '▲' : '▼'}</span>
        </button>

        {isOpen && (
          <div className="dropdown-menu">
            <div className="search-box">
              <input
                type="text"
                placeholder="Search city or state..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                autoFocus
              />
            </div>
            <div className="cities-list">
              {filteredCities.length > 0 ? (
                filteredCities.map((city) => (
                  <div
                    key={city.id}
                    className={`city-item ${selectedCity?.id === city.id ? 'active' : ''}`}
                    onClick={() => handleCitySelect(city)}
                  >
                    <strong>{city.name}</strong>
                    <span className="city-state-small">{city.state}</span>
                  </div>
                ))
              ) : (
                <div className="no-results">No cities found</div>
              )}
            </div>
          </div>
        )}
      </div>
      <p className="selector-info">
        Monitoring {cities.length} cities across India
      </p>
    </div>
  );
}

export default CitySelector;
