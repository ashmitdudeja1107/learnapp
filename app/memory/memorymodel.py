from langchain.memory import ConversationBufferWindowMemory
from typing import Dict, List, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferWindowMemory] = {}
        
    def get_or_create_session(self, session_id: str = None) -> Tuple[str, ConversationBufferWindowMemory]:
        """Get existing session or create new one"""
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True
            )
            logger.info(f"Created new session: {session_id}")
        else:
            logger.info(f"Retrieved existing session: {session_id}")
        
        return session_id, self.sessions[session_id]
    
    def create_session_only(self, session_id: str = None) -> str:
        """Create a new session and return only the session ID (for API responses)"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True
            )
            logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def add_message(self, session_id: str, human_message: str, ai_message: str):
        """Add human and AI messages to session memory"""
        if session_id in self.sessions:
            memory = self.sessions[session_id]
            memory.chat_memory.add_user_message(human_message)
            memory.chat_memory.add_ai_message(ai_message)
            logger.debug(f"Added messages to session {session_id}")
        else:
            logger.warning(f"Attempted to add message to non-existent session: {session_id}")
    
    def get_chat_history(self, session_id: str) -> List[str]:
        """Get formatted chat history for a session"""
        if session_id not in self.sessions:
            logger.warning(f"Requested history for non-existent session: {session_id}")
            return []
        
        memory = self.sessions[session_id]
        messages = memory.chat_memory.messages
        
        history = []
        for message in messages:
            if hasattr(message, 'content'):
                role = "Human" if message.type == "human" else "Assistant"
                history.append(f"{role}: {message.content}")
        
        return history
    
    def get_memory_object(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get the actual memory object for a session"""
        if session_id not in self.sessions:
            logger.warning(f"Requested memory object for non-existent session: {session_id}")
            return None
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
        else:
            logger.warning(f"Attempted to clear non-existent session: {session_id}")
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        return session_id in self.sessions