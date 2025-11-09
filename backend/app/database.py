from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from app.config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    Config.DATABASE_URL,
    echo=Config.DEBUG,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
ScopedSession = scoped_session(SessionLocal)

# Create base class for models
Base = declarative_base()

def get_db():
    """
    Dependency to get database session
    Usage: db = next(get_db())
    """
    db = ScopedSession()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database - create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def close_db():
    """Close database connections"""
    ScopedSession.remove()
    engine.dispose()
    logger.info("Database connections closed")
