"""MongoDB connection manager."""
import asyncio
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError
import structlog

from app.config import settings

logger = structlog.get_logger()


class DatabaseManager:
    """MongoDB connection manager with async support."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        
    async def connect(self) -> None:
        """Establish database connection."""
        try:
            self.client = AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=10
            )
            
            # Test the connection
            await self.client.admin.command('ping')
            
            self.database = self.client[settings.mongodb_database]
            
            # Create indexes
            await self._create_indexes()
            
            logger.info("Successfully connected to MongoDB", 
                       database=settings.mongodb_database)
            
        except ServerSelectionTimeoutError as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        """Create necessary database indexes."""
        if self.database is not None:
            # Create indexes for better query performance
            await self.database.evaluations.create_index("model_name")
            await self.database.evaluations.create_index("created_at")
            await self.database.evaluations.create_index("evaluation_type")
            await self.database.evaluations.create_index([
                ("model_name", 1), 
                ("created_at", -1)
            ])
            
            logger.info("Database indexes created successfully")
    
    def get_collection(self, collection_name: str):
        """Get a collection from the database."""
        if self.database is None:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> AsyncIOMotorDatabase:
    """Dependency to get database instance."""
    if not db_manager.database is None:
        await db_manager.connect()
    return db_manager.database