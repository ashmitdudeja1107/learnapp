from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from database.models.chat import ChatRequest, ChatResponse
from services.chat_service import ChatService
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
chat_service = ChatService()

# Pydantic models for request/response
class ChatRequestSimple(BaseModel):
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

# Dependency to get chat service
async def get_chat_service() -> ChatService:
    """Dependency to provide chat service instance"""
    return chat_service

# ORIGINAL UPLOAD ROUTE (KEPT FROM ORIGINAL chat.py)
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None)
):
    """
    Upload a document (PDF or text) to the knowledge base
    """
    try:
        # Add debugging info
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}, size: {file.size}")
        
        # Check if file is actually received
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Use provided filename or file's filename
        doc_filename = filename or file.filename
        
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Determine if it's a PDF
        is_pdf = file.content_type == "application/pdf" or doc_filename.lower().endswith('.pdf')
        
        # Process document
        result = await chat_service.upload_document(content, doc_filename, is_pdf)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Document uploaded successfully",
                "details": result,
                "filename": doc_filename,
                "size": len(content),
                "content_type": file.content_type
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

# ALL OTHER ROUTES MOVED FROM main.py

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint to verify service status"""
    try:
        service_status = {
            "chat_service": chat_service is not None
        }
        
        # If chat service is available, get additional stats
        if chat_service:
            try:
                stats = chat_service.get_system_stats()
                service_status.update(stats)
            except Exception as e:
                logger.warning(f"Could not get service stats: {str(e)}")
        
        return {
            "status": "healthy" if chat_service else "unhealthy",
            "features": ["quiz_generation", "document_analysis", "chat"],
            "services": service_status
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {"chat_service": False}
        }
@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,  # Changed from ChatRequestSimple to ChatRequest
    chat_service: ChatService = Depends(get_chat_service)
):
    """Enhanced chat endpoint using ChatService process_chat_message method"""
    try:
        # If no session_id provided, create a new session
        if not request.session_id:
            request.session_id = chat_service.create_new_session()
            logger.info(f"Created new session: {request.session_id}")
            new_session = True
        else:
            new_session = False
        
        # Use the full ChatService method
        chat_response = await chat_service.process_chat_message(request)
        
        # Convert ChatResponse to dict format and add session info
        result = {
            "success": True,
            "response": chat_response.response,
            "session_id": chat_response.session_id,
            "sources": chat_response.sources
        }
        
        # Add session creation flag if it was created
        if new_session:
            result["new_session"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return {
            "success": False,
            "error": f"Chat processing failed: {str(e)}",
            "session_id": request.session_id if hasattr(request, 'session_id') else None
        }
# Legacy chat endpoint (backward compatibility)
# Legacy chat endpoint (backward compatibility)
@router.post("/chat/simple")
async def chat_endpoint_simple(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    chat_service: ChatService = Depends(get_chat_service)
):
    """Simple text-only chat endpoint (no document upload support)"""
    try:
        # Process using simple chat method
        result = await chat_service.process_chat_message_simple(
            message=message,
            session_id=session_id,
            use_rag=False  # Simple chat doesn't use RAG
        )
        
        if result["success"]:
            return {
                "response": result["response"],
                "session_id": result["session_id"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Simple chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
    
    
# Document summarization endpoint
@router.post("/summarize-document")
async def summarize_document_endpoint(
    filename: str = Form(...),
    chat_service: ChatService = Depends(get_chat_service)
):
    """Summarize document using ChatService"""
    try:
        result = await chat_service.summarize_document(filename)
        return result
        
    except Exception as e:
        logger.error(f"Document summarization error: {str(e)}")
        return {
            "success": False,
            "error": f"Document summarization failed: {str(e)}"
        }

# Explain concept endpoint
@router.post("/explain-concept")
async def explain_concept_endpoint(
    concept: str = Form(...),
    use_rag: bool = Form(default=True),
    chat_service: ChatService = Depends(get_chat_service)
):
    """Explain concept using ChatService"""
    try:
        result = await chat_service.explain_concept(concept, use_rag)
        return {
            "success": True,
            "explanation": result
        }
        
    except Exception as e:
        logger.error(f"Concept explanation error: {str(e)}")
        return {
            "success": False,
            "error": f"Concept explanation failed: {str(e)}"
        }

# SESSION MANAGEMENT ENDPOINTS

@router.post("/sessions/new")
async def create_new_session(
    request: NewSessionRequest = None,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Create a new chat session with optional initial message"""
    try:
        # Create new session
        session_id = chat_service.create_new_session()
        
        response_data = {
            "success": True,
            "session_id": session_id,
            "message": "New session created successfully",
            "has_initial_message": bool(request and request.initial_message)
        }
        
        # Add initial message if provided
        if request and request.initial_message:
            try:
                # Add the message to the session
                if hasattr(chat_service, 'add_message_to_session'):
                    # Use dedicated method if available
                    await chat_service.add_message_to_session(
                        session_id=session_id,
                        message=request.initial_message,
                        role=request.role
                    )
                else:
                    # Fallback: use process_chat_message_simple if it's a user message
                    if request.role == "user":
                        result = await chat_service.process_chat_message_simple(
                            message=request.initial_message,
                            session_id=session_id,
                            use_rag=False
                        )
                        if request.generate_response:
                            response_data["ai_response"] = result.get("response", "")
                
                logger.info(f"Added initial message to session {session_id}")
                
            except Exception as content_error:
                logger.warning(f"Failed to add initial message: {str(content_error)}")
                response_data["warning"] = f"Session created but failed to add initial message: {str(content_error)}"
        
        return response_data
        
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to create session: {str(e)}"
        }

@router.post("/sessions/{session_id}/messages")
async def add_message_to_session(
    session_id: str,
    request: AddMessageRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Add a new message to an existing session"""
    try:
        # Verify session exists
        try:
            session_info = chat_service.get_session_info(session_id)
            if not session_info:
                raise HTTPException(status_code=404, detail="Session not found")
        except Exception:
            raise HTTPException(status_code=404, detail="Session not found")
        
        response_data = {
            "success": True,
            "session_id": session_id,
            "message": "Message added to session successfully",
            "role": request.role,
            "message_content": request.message
        }
        
        # Add message to session
        if hasattr(chat_service, 'add_message_to_session'):
            # Use dedicated method if available
            result = await chat_service.add_message_to_session(
                session_id=session_id,
                message=request.message,
                role=request.role
            )
            response_data.update(result)
            
        elif request.role == "user":
            # For user messages, use the chat processing (this will add to history)
            if request.generate_response:
                # Generate AI response
                result = await chat_service.process_chat_message_simple(
                    message=request.message,
                    session_id=session_id,
                    use_rag=True
                )
                response_data["ai_response"] = result.get("response", "")
                response_data["response_generated"] = True
            else:
                # Just add the message without generating response
                # This might need a custom method in your ChatService
                result = await chat_service.process_chat_message_simple(
                    message=request.message,
                    session_id=session_id,
                    use_rag=False
                )
                response_data["message_added"] = True
        else:
            # For assistant/system messages, we need a way to add them to history
            # This would require a method in ChatService to add messages without processing
            logger.warning(f"Cannot add {request.role} message without dedicated method")
            response_data["warning"] = f"Adding {request.role} messages requires ChatService.add_message_to_session method"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add message to session error: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to add message to session: {str(e)}"
        }

@router.post("/sessions/{session_id}/messages/batch")
async def add_multiple_messages_to_session(
    session_id: str,
    messages: List[AddMessageRequest],
    chat_service: ChatService = Depends(get_chat_service)
):
    """Add multiple messages to an existing session at once"""
    try:
        # Verify session exists
        try:
            session_info = chat_service.get_session_info(session_id)
            if not session_info:
                raise HTTPException(status_code=404, detail="Session not found")
        except Exception:
            raise HTTPException(status_code=404, detail="Session not found")
        
        results = []
        for i, message_request in enumerate(messages):
            try:
                # Add each message
                if hasattr(chat_service, 'add_message_to_session'):
                    result = await chat_service.add_message_to_session(
                        session_id=session_id,
                        message=message_request.message,
                        role=message_request.role
                    )
                    results.append({
                        "index": i,
                        "success": True,
                        "role": message_request.role,
                        "message": message_request.message[:50] + "..." if len(message_request.message) > 50 else message_request.message
                    })
                else:
                    # Fallback for user messages
                    if message_request.role == "user":
                        await chat_service.process_chat_message_simple(
                            message=message_request.message,
                            session_id=session_id,
                            use_rag=False
                        )
                        results.append({
                            "index": i,
                            "success": True,
                            "role": message_request.role,
                            "message": message_request.message[:50] + "..." if len(message_request.message) > 50 else message_request.message
                        })
                    else:
                        results.append({
                            "index": i,
                            "success": False,
                            "error": f"Cannot add {message_request.role} message without dedicated method",
                            "role": message_request.role
                        })
                        
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "role": message_request.role
                })
        
        return {
            "success": True,
            "session_id": session_id,
            "total_messages": len(messages),
            "successful_additions": len([r for r in results if r.get("success")]),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch add messages error: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to add messages to session: {str(e)}"
        }

@router.get("/sessions")
async def get_all_sessions(chat_service: ChatService = Depends(get_chat_service)):
    """Get all active sessions using ChatService method"""
    try:
        sessions = chat_service.get_all_sessions()
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get sessions: {str(e)}",
            "sessions": []
        }

@router.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get session information using ChatService method"""
    try:
        result = chat_service.get_session_info(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session_info": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session info error: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get session info: {str(e)}"
        }

@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Clear session using ChatService method"""
    try:
        result = chat_service.clear_session(session_id)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Clear session error: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to clear session: {str(e)}"
        }

# Additional endpoints for debugging
@router.get("/debug/service-status")
async def debug_service_status():
    """Debug endpoint to check service initialization status"""
    return {
        "chat_service_instance": chat_service is not None,
        "service_type": type(chat_service).__name__ if chat_service else None,
        "available_methods": dir(chat_service) if chat_service else []
    }