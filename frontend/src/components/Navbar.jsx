import React from 'react';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <h1>🌍 AQI Prediction System</h1>
          <p>Real-time Air Quality Monitoring</p>
        </div>
        <div className="navbar-info">
          <span className="live-indicator">
            <span className="pulse"></span>
            Live
          </span>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
