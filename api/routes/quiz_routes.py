from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import tempfile
import os
import hashlib
import json
import logging
from datetime import datetime, timedelta

from services.quiz_service import QuizService
from services.llm_service import LLMService, create_llm_service
from app.rag.quiz_rag import QuizRAGService
from database.models.quiz_models import (
    QuizGenerationRequest, UserAnswer, QuizSubmission, 
    QuizQuestion, QuizResult, DifficultyLevel
)

# Import the new Redis configuration
from redis_config import (
    redis_client, is_redis_available, safe_redis_get, safe_redis_setex,
    safe_redis_delete, safe_redis_exists, get_quiz_cache_key,
    get_session_cache_key, store_quiz_session_redis, get_quiz_session_redis,
    cleanup_expired_sessions_redis
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quiz", tags=["quiz"])

# Initialize Llama3 service for local development
def initialize_llama3_service():
    """Initialize LLM service with local Llama3 model via Ollama"""
    try:
        llm_service = create_llm_service(
            provider="ollama",
            model_name="llama3",
            temperature=0.7,
            max_tokens=2000
        )
        return llm_service
    except Exception as e:
        raise Exception(f"Could not connect to local Llama3 via Ollama. Make sure Ollama is running: {str(e)}")

# Initialize services
llm_service = initialize_llama3_service()
quiz_service = QuizService(llm_service.model)

# Fallback in-memory storage for when Redis is unavailable
quiz_sessions = {}

class SimpleAnswersRequest(BaseModel):
    session_id: str
    answers: List[str]

class QuizGenerationRequest(BaseModel):
    num_questions: int
    difficulty: str 
    question_type: Optional[str] = "multiple_choice"

class TextQuizRequest(BaseModel):
    text_content: str
    request: QuizGenerationRequest

class QuizEvaluationRequest(BaseModel):
    questions: List[QuizQuestion]
    submission: QuizSubmission

def store_quiz_session(questions: List[QuizQuestion]) -> str:
    """
    Store quiz questions and return session ID
    Uses Redis if available, fallback to in-memory storage
    """
    session_id = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Convert questions to serializable format
    questions_data = []
    for q in questions:
        questions_data.append({
            'id': q.id,
            'question': q.question,
            'options': q.options,
            'correct_answer': q.correct_answer,
            'explanation': q.explanation,
            'difficulty': q.difficulty.value if hasattr(q.difficulty, 'value') else str(q.difficulty),
            'topic': q.topic
        })
    
    # Try Redis first
    if store_quiz_session_redis(session_id, questions_data):
        logger.info(f"Session {session_id} stored in Redis")
        return session_id
    
    # Fallback to in-memory storage
    quiz_sessions[session_id] = {
        'questions': questions,
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(hours=24)
    }
    logger.info(f"Session {session_id} stored in memory (Redis unavailable)")
    return session_id

def get_quiz_session(session_id: str) -> Optional[List[QuizQuestion]]:
    """
    Retrieve quiz session from Redis or in-memory storage
    """
    # Try Redis first
    redis_data = get_quiz_session_redis(session_id)
    if redis_data:
        # Convert back to QuizQuestion objects
        questions = []
        for q_data in redis_data['questions']:
            question = QuizQuestion(
                id=q_data['id'],
                question=q_data['question'],
                options=q_data['options'],
                correct_answer=q_data['correct_answer'],
                explanation=q_data['explanation'],
                difficulty=DifficultyLevel(q_data['difficulty']),
                topic=q_data['topic']
            )
            questions.append(question)
        return questions
    
    # Fallback to in-memory storage
    if session_id in quiz_sessions:
        session_data = quiz_sessions[session_id]
        if session_data['expires_at'] > datetime.now():
            return session_data['questions']
        else:
            del quiz_sessions[session_id]  # Remove expired session
    
    return None

def clean_expired_sessions():
    """Clean up expired sessions from both Redis and in-memory storage"""
    # Clean Redis sessions
    cleanup_expired_sessions_redis()
    
    # Clean in-memory sessions
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, data in quiz_sessions.items()
        if data['expires_at'] < current_time
    ]
    for session_id in expired_sessions:
        del quiz_sessions[session_id]

@router.post("/generate")
async def generate_quiz_from_uploaded_file(
    file: UploadFile = File(...),
    num_questions: int = Form(8),
    difficulty: str = Form("medium")
):
    tmp_file_path = None
    try:
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type.")

        if num_questions < 1 or num_questions > 20:
            raise HTTPException(status_code=400, detail="Invalid num_questions")

        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File empty")

        # Generate cache key using hash of file content + params
        file_hash = hashlib.sha256(content).hexdigest()
        cache_key = get_quiz_cache_key(file_hash, num_questions, difficulty)

        # Check Redis cache
        cached_data = safe_redis_get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for key: {cache_key}")
            return json.loads(cached_data)
        
        # Store file temporarily
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tmp_file_path = tmp_file.name
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()

        documents = quiz_service.process_document_for_quiz(tmp_file_path)
        if not documents:
            raise HTTPException(status_code=422, detail="Document unreadable")

        quiz_response = quiz_service.generate_quiz_questions(
            num_questions=num_questions,
            difficulty=difficulty,
            question_type="multiple_choice"
        )

        if not quiz_response.get('questions'):
            raise HTTPException(status_code=422, detail="Quiz generation failed")

        questions = []
        for q_data in quiz_response['questions']:
            question = QuizQuestion(
                id=q_data['id'],
                question=q_data['question'],
                options=q_data['options'],
                correct_answer=q_data['correct_answer'],
                explanation=q_data.get('explanation', ''),
                difficulty=DifficultyLevel(q_data.get('difficulty', difficulty)),
                topic=q_data.get('topic', 'General')
            )
            questions.append(question)

        session_id = store_quiz_session(questions)
        clean_expired_sessions()

        response = {
            "session_id": session_id,
            "questions": quiz_response['questions']
        }

        # Cache the response with 10 minute TTL
        cache_success = safe_redis_setex(cache_key, 600, json.dumps(response))
        if cache_success:
            logger.info(f"Response cached successfully for key: {cache_key}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.info(f"Deleted temp file: {tmp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {str(cleanup_error)}")

@router.post("/generate-from-text")
async def generate_quiz_from_text(data: TextQuizRequest):
    """
    Generate quiz questions directly from provided text content using Llama3
    """
    try:
        if not data.text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        if len(data.text_content) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Text content too short. Please provide at least 100 characters for meaningful quiz generation."
            )
        
        # Generate cache key for text-based quizzes
        text_hash = hashlib.sha256(data.text_content.encode()).hexdigest()
        cache_key = get_quiz_cache_key(text_hash, data.request.num_questions, data.request.difficulty)
        
        # Check cache first
        cached_data = safe_redis_get(cache_key)
        if cached_data:
            logger.info(f"Text quiz cache hit for key: {cache_key}")
            return json.loads(cached_data)
        
        # Generate questions using Llama3
        questions_data = llm_service.generate_quiz_questions(
            content=data.text_content,
            num_questions=data.request.num_questions,
            difficulty=data.request.difficulty,
            question_type=data.request.question_type
        )
        
        if not questions_data:
            raise HTTPException(
                status_code=422, 
                detail="Could not generate questions from the provided text using Llama3"
            )
        
        # Process questions
        questions = []
        quiz_questions_for_response = []
        
        for i, q_data in enumerate(questions_data):
            try:
                options = []
                for j, opt in enumerate(q_data['options']):
                    if isinstance(opt, dict):
                        options.append({
                            "label": opt.get('label', chr(65 + j)),
                            "text": opt.get('text', str(opt)),
                            "is_correct": opt.get('is_correct', False)
                        })
                    else:
                        options.append({
                            "label": chr(65 + j),
                            "text": str(opt),
                            "is_correct": False
                        })
                
                question_obj = QuizQuestion(
                    id=q_data.get('id', f'q{i+1}'),
                    question=q_data['question'],
                    options=options,
                    correct_answer=q_data['correct_answer'],
                    explanation=q_data.get('explanation', ''),
                    difficulty=DifficultyLevel(q_data.get('difficulty', data.request.difficulty)),
                    topic=q_data.get('topic', 'General')
                )
                questions.append(question_obj)
                
                question_response = {
                    "id": q_data.get('id', f'q{i+1}'),
                    "question": q_data['question'],
                    "options": options,
                    "correct_answer": q_data['correct_answer'],
                    "explanation": q_data.get('explanation', ''),
                    "difficulty": q_data.get('difficulty', data.request.difficulty),
                    "topic": q_data.get('topic', 'General')
                }
                quiz_questions_for_response.append(question_response)
                
            except KeyError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid question format from Llama3: missing {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Error processing question {i+1}: {str(e)}"
                )
        
        session_id = store_quiz_session(questions)
        clean_expired_sessions()
        
        response = {
            "session_id": session_id,
            "questions": quiz_questions_for_response
        }
        
        # Cache the response
        safe_redis_setex(cache_key, 600, json.dumps(response))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz with Llama3: {str(e)}")

@router.post("/evaluate", response_model=QuizResult)
async def evaluate_quiz(request: SimpleAnswersRequest):
    """
    Evaluate quiz with simple answers
    """
    try:
        clean_expired_sessions()
        
        # Get stored questions
        questions = get_quiz_session(request.session_id)
        if not questions:
            raise HTTPException(
                status_code=404, 
                detail="Quiz session not found or expired. Please generate a new quiz."
            )
        
        # Validate answer count
        if len(request.answers) != len(questions):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(questions)} answers, got {len(request.answers)}"
            )
        
        # Validate answer format
        valid_answers = {'A', 'B', 'C', 'D'}
        for i, answer in enumerate(request.answers):
            if answer not in valid_answers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid answer '{answer}' at position {i+1}. Must be A, B, C, or D"
                )
        
        # Create user answers mapping
        user_answers = {}
        for i, answer in enumerate(request.answers):
            question_id = questions[i].id
            user_answers[question_id] = answer
        
        # Evaluate quiz
        results = quiz_service.evaluate_quiz(questions, user_answers)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating quiz: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint that includes Redis status"""
    redis_status = "connected" if is_redis_available() else "disconnected"
    return {
        "status": "healthy",
        "redis_status": redis_status,
        "timestamp": datetime.now().isoformat()
    }

@router.delete("/session/{session_id}")
async def delete_quiz_session(session_id: str):
    """Delete a specific quiz session"""
    try:
        # Try to delete from Redis
        redis_key = get_session_cache_key(session_id)
        redis_deleted = safe_redis_delete(redis_key)
        
        # Also delete from in-memory storage
        memory_deleted = session_id in quiz_sessions
        if memory_deleted:
            del quiz_sessions[session_id]
        
        if redis_deleted or memory_deleted:
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@router.get("/sessions/cleanup")
async def cleanup_sessions():
    """Manually trigger session cleanup"""
    try:
        clean_expired_sessions()
        return {"message": "Session cleanup completed"}
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")