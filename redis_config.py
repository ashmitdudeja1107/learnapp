import os
import logging
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from typing import Optional
from datetime import datetime, timedelta  # Fixed import

logger = logging.getLogger(__name__)

class RedisConfig:
    """Redis configuration and connection management"""
    
    def __init__(self):
        # Redis connection parameters from your Valkey dashboard
        self.host = os.getenv('REDIS_HOST', 'valkey-291ddeb5-ashmit-650b.f.aivencloud.com')
        self.port = int(os.getenv('REDIS_PORT', '19758'))
        self.username = os.getenv('REDIS_USERNAME', 'default')
        self.password = os.getenv('REDIS_PASSWORD', 'AVNS_SoEtTDbyYLt9D7h143i')  # Replace with actual password
        self.decode_responses = True
        self.socket_timeout = 5
        self.socket_connect_timeout = 5
        self.health_check_interval = 30
        
        # SSL/TLS configuration for cloud Redis
        self.ssl = True
        self.ssl_cert_reqs = None  # For self-signed certificates
        
        self._client = None
    
    def get_client(self) -> Redis:
        """Get Redis client instance with connection pooling"""
        if self._client is None:
            self._client = Redis(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                health_check_interval=self.health_check_interval,
                ssl=self.ssl,
                ssl_cert_reqs=self.ssl_cert_reqs,
                retry_on_timeout=True,
                retry_on_error=[ConnectionError, TimeoutError],
                max_connections=10
            )
        return self._client
    
    def test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            client = self.get_client()
            client.ping()
            logger.info("Redis connection successful!")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            return False

# Initialize Redis configuration
redis_config = RedisConfig()
redis_client = redis_config.get_client()

def is_redis_available() -> bool:
    """
    Check if Redis is available and working
    Returns: bool - True if Redis is working, False otherwise
    """
    try:
        redis_client.ping()
        return True
    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning(f"Redis unavailable: {str(e)}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected Redis error: {str(e)}")
        return False

def safe_redis_get(key: str) -> Optional[str]:
    """
    Safely get value from Redis with fallback
    Returns: cached value or None if Redis is unavailable
    """
    if not is_redis_available():
        return None
    
    try:
        return redis_client.get(key)
    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning(f"Redis get failed for key {key}: {str(e)}")
        return None

def safe_redis_setex(key: str, time: int, value: str) -> bool:
    """
    Safely set value in Redis with expiration, with fallback
    Returns: bool - True if successful, False if failed
    """
    if not is_redis_available():
        logger.info("Redis unavailable, skipping cache set")
        return False
    
    try:
        redis_client.setex(key, time, value)
        return True
    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning(f"Redis setex failed for key {key}: {str(e)}")
        return False

def safe_redis_delete(key: str) -> bool:
    """
    Safely delete key from Redis
    Returns: bool - True if successful, False if failed
    """
    if not is_redis_available():
        logger.info("Redis unavailable, skipping cache delete")
        return False
    
    try:
        redis_client.delete(key)
        return True
    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning(f"Redis delete failed for key {key}: {str(e)}")
        return False

def safe_redis_exists(key: str) -> bool:
    """
    Safely check if key exists in Redis
    Returns: bool - True if exists, False if not exists or Redis unavailable
    """
    if not is_redis_available():
        return False
    
    try:
        return redis_client.exists(key)
    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning(f"Redis exists failed for key {key}: {str(e)}")
        return False

def safe_redis_expire(key: str, time: int) -> bool:
    """
    Safely set expiration time for key in Redis
    Returns: bool - True if successful, False if failed
    """
    if not is_redis_available():
        logger.info("Redis unavailable, skipping expire set")
        return False
    
    try:
        redis_client.expire(key, time)
        return True
    except (ConnectionError, TimeoutError, RedisError) as e:
        logger.warning(f"Redis expire failed for key {key}: {str(e)}")
        return False

# Cache key generators for different data types
def get_quiz_cache_key(file_hash: str, num_questions: int, difficulty: str) -> str:
    """Generate cache key for quiz questions"""
    return f"quiz:{file_hash}:{num_questions}:{difficulty}"

def get_session_cache_key(session_id: str) -> str:
    """Generate cache key for quiz sessions"""
    return f"session:{session_id}"

def get_user_cache_key(user_id: str, action: str) -> str:
    """Generate cache key for user-specific data"""
    return f"user:{user_id}:{action}"

# Enhanced session management with Redis
import json

class QuizJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for quiz-related objects"""
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def serialize_quiz_data(data):
    """
    Recursively serialize quiz data to make it JSON serializable
    Handles QuizOption objects and other custom objects
    """
    if hasattr(data, '__dict__'):
        # If it's a custom object, convert to dict
        return {key: serialize_quiz_data(value) for key, value in data.__dict__.items()}
    elif isinstance(data, dict):
        return {key: serialize_quiz_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_quiz_data(item) for item in data]
    else:
        return data

def store_quiz_session_redis(session_id: str, questions_data: dict, ttl: int = 86400) -> bool:
    """
    Store quiz session in Redis with TTL
    Args:
        session_id: Unique session identifier
        questions_data: Quiz questions and metadata
        ttl: Time to live in seconds (default: 24 hours)
    Returns: bool - True if successful, False if failed
    """
    cache_key = get_session_cache_key(session_id)
    
    # Serialize the questions data to handle custom objects
    try:
        serialized_questions = serialize_quiz_data(questions_data)
        session_data = {
            'questions': serialized_questions,
            'created_at': str(datetime.now()),
            'expires_at': str(datetime.now() + timedelta(seconds=ttl))
        }
        
        # Use custom encoder as fallback
        json_string = json.dumps(session_data, cls=QuizJSONEncoder)
        return safe_redis_setex(cache_key, ttl, json_string)
    except Exception as e:
        logger.error(f"Failed to store session {session_id}: {str(e)}")
        return False

def get_quiz_session_redis(session_id: str) -> Optional[dict]:
    """
    Retrieve quiz session from Redis
    Returns: session data or None if not found
    """
    cache_key = get_session_cache_key(session_id)
    
    try:
        cached_data = safe_redis_get(cache_key)
        if cached_data:
            import json
            return json.loads(cached_data)
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve session {session_id}: {str(e)}")
        return None

def cleanup_expired_sessions_redis() -> int:
    """
    Clean up expired sessions from Redis
    Returns: number of sessions cleaned
    """
    try:
        if not is_redis_available():
            return 0
        
        # Get all session keys
        pattern = "session:*"
        keys = redis_client.keys(pattern)
        
        cleaned_count = 0
        for key in keys:
            try:
                # Check if key has TTL
                ttl = redis_client.ttl(key)
                if ttl == -1:  # Key exists but has no TTL
                    redis_client.expire(key, 86400)  # Set 24 hour expiry
                elif ttl == -2:  # Key doesn't exist
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Error checking TTL for key {key}: {str(e)}")
        
        return cleaned_count
    except Exception as e:
        logger.error(f"Failed to cleanup expired sessions: {str(e)}")
        return 0

# Test the connection on import
if __name__ == "__main__":
    if redis_config.test_connection():
        print("✅ Redis connection successful!")
    else:
        print("❌ Redis connection failed!")