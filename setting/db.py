from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from setting.base import DATABASE_URI, SESSION_POOL_SIZE
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

engine = create_engine(
    DATABASE_URI,
    pool_size=SESSION_POOL_SIZE,
    max_overflow=20,
    pool_timeout=60,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 60,
        "read_timeout": 300,
        "write_timeout": 300,
    },
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class DatabaseManager:
    """
    Manages database connections for multi-database support.
    Supports both local mode (backward compatibility) and multi-database mode.
    """

    def __init__(self):
        self.user_connections: Dict[str, sessionmaker] = {}
        self.local_database_uri = DATABASE_URI
        logger.info(
            f"DatabaseManager initialized with local database"
        )

    def get_session_factory(self, database_uri: Optional[str] = None) -> sessionmaker:
        """
        Get database session factory for the given database URI.

        Args:
            database_uri: Database connection string

        Returns:
            sessionmaker: Session factory for the database

        Raises:
            Exception: If database connection fails
        """
        if self.is_local_mode(database_uri):
            logger.debug("Using local database session factory")
            return SessionLocal

        # Multi-database mode
        if database_uri not in self.user_connections:
            logger.info(f"Creating new connection for external database")
            try:
                engine = create_engine(
                    database_uri,
                    pool_size=10,  # Smaller pool for user databases
                    max_overflow=20,
                    pool_timeout=60,
                    pool_recycle=1800,
                    pool_pre_ping=True,
                    connect_args={
                        "connect_timeout": 60,
                        "read_timeout": 300,
                        "write_timeout": 300,
                    },
                    echo=False,
                )

                # Create all tables in user database
                self._create_user_tables(engine)

                self.user_connections[database_uri] = sessionmaker(
                    autocommit=False, autoflush=False, bind=engine
                )
                logger.info(f"Successfully created connection for user database")

            except Exception as e:
                logger.error(f"Failed to create database connection: {e}")
                raise Exception(f"Failed to connect to database: {str(e)}")

        return self.user_connections[database_uri]

    def is_local_mode(self, database_uri: Optional[str] = None) -> bool:
        """
        Check if the given database URI is the same as local database.

        Args:
            database_uri: Database connection string to check

        Returns:
            bool: True if it's local mode, False otherwise
        """
        return (
            database_uri is None
            or database_uri == ""
            or database_uri == self.local_database_uri
        )

    def _create_user_tables(self, engine):
        """
        Create all necessary tables in user database.

        Args:
            engine: SQLAlchemy engine for the user database

        Raises:
            Exception: If table creation fails
        """
        try:
            from knowledge_graph.models import Base

            Base.metadata.create_all(engine)
            logger.info("Successfully created tables in user database")
        except Exception as e:
            logger.error(f"Failed to create tables in user database: {e}")
            raise Exception(f"Failed to initialize database schema: {str(e)}")

    def validate_database_connection(self, database_uri: str) -> bool:
        """
        Validate if database connection is working.

        Args:
            database_uri: Database connection string to validate

        Returns:
            bool: True if connection is valid, False otherwise
        """
        try:
            session_factory = self.get_session_factory(database_uri)
            with session_factory() as session:
                # Simple query to test connection
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection validation failed: {e}")
            return False

    def close_all_connections(self):
        """Close all user database connections."""
        for database_uri, session_factory in self.user_connections.items():
            try:
                session_factory.bind.dispose()
                logger.info(f"Closed connection for database: {database_uri[:50]}...")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        self.user_connections.clear()


# Global database manager instance
db_manager = DatabaseManager()
