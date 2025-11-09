# AQI Prediction System - Backend

Flask-based REST API for AQI prediction system.

## Setup

### 1. Create Virtual Environment

### 2. Install Dependencies

### 3. Configure Environment
Copy `.env.example` to `.env` and add your API keys:
- OpenWeather API Key
- IQAir API Key

### 4. Run Application

## API Endpoints

- `GET /api/cities` - Get list of all cities
- `GET /api/predictions/<city>` - Get AQI predictions for a city
- `GET /api/health` - Health check endpoint
- `POST /api/train` - Trigger model training

## Project Structure
