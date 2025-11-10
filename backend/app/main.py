"""
Main Flask Application with Scheduler
"""
from flask import Flask, jsonify
from flask_cors import CORS
from app.config import Config
from app.database import init_db, close_db
from app.routes.cities import cities_bp
from app.routes.predictions import predictions_bp
from app.routes.health import health_bp
from app.services.scheduler import SchedulerService
import logging
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler_service = None


def create_app():
    """Application factory"""
    global scheduler_service
    
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app, origins=Config.CORS_ORIGINS)
    
    # Initialize database
    with app.app_context():
        try:
            Config.validate_config()
            init_db()
            logger.info("✓ Database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    # Initialize and start scheduler
    try:
        scheduler_service = SchedulerService()
        scheduler_service.start()
        logger.info("✓ Scheduler service started")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        logger.warning("Application will continue without scheduler")
    
    # Register cleanup on shutdown
    atexit.register(lambda: shutdown_scheduler())
    
    # Register blueprints
    app.register_blueprint(health_bp, url_prefix='/api')
    app.register_blueprint(cities_bp, url_prefix='/api')
    app.register_blueprint(predictions_bp, url_prefix='/api')
    
    # Scheduler routes
    @app.route('/api/scheduler/status', methods=['GET'])
    def scheduler_status():
        """Get scheduler status"""
        if scheduler_service and scheduler_service.is_running:
            jobs = scheduler_service.get_jobs()
            return jsonify({
                'status': 'running',
                'jobs': jobs
            }), 200
        else:
            return jsonify({
                'status': 'stopped',
                'jobs': []
            }), 200
    
    @app.route('/api/scheduler/trigger-fetch', methods=['POST'])
    def trigger_data_fetch():
        """Manually trigger data fetch"""
        if scheduler_service:
            try:
                scheduler_service.fetch_all_cities_job()
                return jsonify({
                    'success': True,
                    'message': 'Data fetch triggered successfully'
                }), 200
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Scheduler not available'
            }), 503
    
    # Root route
    @app.route('/')
    def root():
        return jsonify({
            'message': 'AQI Prediction System API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/api/health',
                'cities': '/api/cities',
                'predictions': '/api/predictions/<city_name>',
                'current': '/api/current/<city_name>',
                'historical': '/api/historical/<city_name>',
                'scheduler_status': '/api/scheduler/status'
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app


def shutdown_scheduler():
    """Cleanup function to stop scheduler"""
    global scheduler_service
    if scheduler_service and scheduler_service.is_running:
        logger.info("Shutting down scheduler...")
        scheduler_service.stop()


# Create app instance
app = create_app()


if __name__ == '__main__':
    try:
        logger.info("="*60)
        logger.info("STARTING AQI PREDICTION SYSTEM")
        logger.info("="*60)
        logger.info(f"Server: http://{Config.HOST}:{Config.PORT}")
        logger.info(f"Data fetch interval: Every {Config.DATA_FETCH_INTERVAL_HOURS} hour(s)")
        logger.info(f"Model retrain interval: Every {Config.MODEL_RETRAIN_INTERVAL_DAYS} day(s)")
        logger.info("="*60)
        
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        shutdown_scheduler()
        close_db()
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        shutdown_scheduler()
        close_db()
