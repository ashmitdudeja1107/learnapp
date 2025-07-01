import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import redis.asyncio as redis
from redis.asyncio import Redis
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.session_expire_minutes = int(os.getenv("SESSION_EXPIRE_MINUTES", 60))
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    async def store_quiz_session(self, questions: List[Dict[str, Any]]) -> str:
        """
        Store quiz session in Redis
        Returns session_id
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            session_id = str(uuid.uuid4())
            
            # Prepare session data
            session_data = {
                "session_id": session_id,
                "questions": questions,
                "created_at": datetime.utcnow().isoformat(),
                "total_questions": len(questions),
                "completed": False
            }
            
            # Store in Redis with expiration
            key = f"quiz_session:{session_id}"
            await self.redis_client.setex(
                key,
                timedelta(minutes=self.session_expire_minutes),
                json.dumps(session_data, default=str)
            )
            
            logger.info(f"Stored quiz session {session_id} with {len(questions)} questions")
            return session_id
            
        except Exception as e:
            logger.error(f"Error storing quiz session: {str(e)}")
            raise
    
    async def get_quiz_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve quiz session from Redis
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            key = f"quiz_session:{session_id}"
            session_data = await self.redis_client.get(key)
            
            if session_data:
                return json.loads(session_data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving quiz session {session_id}: {str(e)}")
            return None
    
    async def update_quiz_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update quiz session in Redis
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            # Get existing session
            session_data = await self.get_quiz_session(session_id)
            if not session_data:
                return False
            
            # Update session data
            session_data.update(updates)
            session_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Store updated session
            key = f"quiz_session:{session_id}"
            await self.redis_client.setex(
                key,
                timedelta(minutes=self.session_expire_minutes),
                json.dumps(session_data, default=str)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating quiz session {session_id}: {str(e)}")
            return False
    
    async def delete_quiz_session(self, session_id: str) -> bool:
        """
        Delete quiz session from Redis
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            key = f"quiz_session:{session_id}"
            result = await self.redis_client.delete(key)
            
            if result:
                logger.info(f"Deleted quiz session {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting quiz session {session_id}: {str(e)}")
            return False
    
    async def get_all_sessions(self) -> List[str]:
        """
        Get all active quiz session IDs
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            keys = await self.redis_client.keys("quiz_session:*")
            session_ids = [key.split(":")[-1] for key in keys]
            return session_ids
            
        except Exception as e:
            logger.error(f"Error getting all sessions: {str(e)}")
            return []
    
    async def clean_expired_sessions(self):
        """
        Clean up expired sessions (Redis handles this automatically with TTL)
        This method is for manual cleanup if needed
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            # Get all session keys
            keys = await self.redis_client.keys("quiz_session:*")
            
            expired_count = 0
            for key in keys:
                # Check if key exists (Redis auto-expires keys with TTL)
                if not await self.redis_client.exists(key):
                    expired_count += 1
            
            logger.info(f"Cleaned up {expired_count} expired sessions")
            
        except Exception as e:
            logger.error(f"Error cleaning expired sessions: {str(e)}")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about quiz sessions
        """
        try:
            if not self.redis_client:
                await self.connect()
            
            # Get all session keys
            keys = await self.redis_client.keys("quiz_session:*")
            
            stats = {
                "total_sessions": len(keys),
                "active_sessions": len(keys),  # All keys returned are active (non-expired)
                "redis_info": await self.redis_client.info("memory")
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting session stats: {str(e)}")
            return {"error": str(e)}

# Global Redis service instance
redis_service = RedisService()

# Utility functions for backward compatibility
async def store_quiz_session(questions: List[Dict[str, Any]]) -> str:
    """Utility function to store quiz session"""
    return await redis_service.store_quiz_session(questions)

async def get_quiz_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Utility function to get quiz session"""
    return await redis_service.get_quiz_session(session_id)

async def clean_expired_sessions():
    """Utility function to clean expired sessions"""
    await redis_service.clean_expired_sessions()