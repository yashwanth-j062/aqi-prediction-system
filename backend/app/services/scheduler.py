"""
Scheduler Service - Automates data fetching and model retraining
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging

from app.config import Config
from app.services.data_fetcher import DataFetcher
from app.services.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchedulerService:
    """Service for scheduling periodic tasks"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.data_fetcher = DataFetcher()
        self.is_running = False
    
    def fetch_all_cities_job(self):
        """Job to fetch data for all cities"""
        try:
            logger.info("="*60)
            logger.info("SCHEDULED DATA FETCH STARTED")
            logger.info(f"Time: {datetime.now()}")
            logger.info("="*60)
            
            stats = self.data_fetcher.fetch_all_cities()
            
            logger.info("="*60)
            logger.info("SCHEDULED DATA FETCH COMPLETED")
            logger.info(f"Success: {stats['successful']}/{stats['total_cities']} cities")
            logger.info(f"Records created: {stats['records_created']}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error in scheduled data fetch: {e}")
    
    def retrain_models_job(self):
        """Job to retrain models periodically"""
        try:
            logger.info("="*60)
            logger.info("SCHEDULED MODEL RETRAINING STARTED")
            logger.info(f"Time: {datetime.now()}")
            logger.info("="*60)
            
            # Retrain for major cities (you can expand this list)
            major_cities = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai"]
            
            for city in major_cities:
                try:
                    logger.info(f"\nRetraining models for {city}...")
                    trainer = ModelTrainer()
                    results = trainer.train_all_models(city, days=30)
                    logger.info(f"✓ {city} - Best model: {results['best_model']}")
                    trainer.close()
                except Exception as e:
                    logger.error(f"Failed to retrain for {city}: {e}")
            
            logger.info("="*60)
            logger.info("SCHEDULED MODEL RETRAINING COMPLETED")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error in scheduled model retraining: {e}")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Schedule data fetching (every N hours based on config)
        self.scheduler.add_job(
            func=self.fetch_all_cities_job,
            trigger=IntervalTrigger(hours=Config.DATA_FETCH_INTERVAL_HOURS),
            id='fetch_data_job',
            name='Fetch air quality data for all cities',
            replace_existing=True
        )
        logger.info(f"✓ Scheduled data fetching every {Config.DATA_FETCH_INTERVAL_HOURS} hour(s)")
        
        # Schedule model retraining (every N days based on config)
        self.scheduler.add_job(
            func=self.retrain_models_job,
            trigger=IntervalTrigger(days=Config.MODEL_RETRAIN_INTERVAL_DAYS),
            id='retrain_models_job',
            name='Retrain ML models',
            replace_existing=True
        )
        logger.info(f"✓ Scheduled model retraining every {Config.MODEL_RETRAIN_INTERVAL_DAYS} day(s)")
        
        # Start scheduler
        self.scheduler.start()
        self.is_running = True
        logger.info("✓ Scheduler started successfully")
        
        # Run initial data fetch
        logger.info("\nRunning initial data fetch...")
        self.fetch_all_cities_job()
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.scheduler.shutdown()
        self.is_running = False
        logger.info("✓ Scheduler stopped")
    
    def get_jobs(self):
        """Get list of scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            })
        return jobs
