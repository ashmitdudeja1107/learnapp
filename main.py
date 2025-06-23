from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

# Import your existing services and models
from services.chat_service import ChatService
from api.routes.quiz_routes import router as quiz_router
from api.routes.chat import router as chat_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for services
chat_service_instance = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True

class AddMessageRequest(BaseModel):
    message: str
    role: Optional[str] = "user"  # user, assistant, system
    generate_response: bool = False  # Whether to generate AI response after adding message

class NewSessionRequest(BaseModel):
    initial_message: Optional[str] = None
    role: Optional[str] = "user"
    generate_response: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global chat_service_instance
    
    try:
        logger.info("Initializing services...")
        
        # Initialize chat service - this was missing in your original code!
        chat_service_instance = ChatService()
        logger.info("ChatService initialized successfully")
        
        # Test the service to make sure it's working
        stats = chat_service_instance.get_system_stats()
        logger.info(f"Chat service stats: {stats}")
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        # Don't raise the exception here, just log it
        # The dependency function will handle the error
        yield
    finally:
        logger.info("Application shutdown")
        # Add any cleanup logic here if needed
        chat_service_instance = None

# Create FastAPI app
app = FastAPI(
    title="AI Tutor API - LearnFlow Quiz System",
    description="Comprehensive API for AI tutoring with quiz generation from uploaded documents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get chat service
async def get_chat_service() -> ChatService:
    """Dependency to provide chat service instance"""
    if chat_service_instance is None:
        logger.error("Chat service not initialized")
        raise RuntimeError("Chat service not initialized")
    return chat_service_instance

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(quiz_router, prefix="/api/v1", tags=["quiz"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Tutor API - LearnFlow Quiz System is running",
        "features": ["chat", "quiz_generation", "document_analysis"],
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True  # Remove in production
    )