from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import tempfile
import os

import logging
from datetime import datetime, timedelta
import tempfile
from services.quiz_service import QuizService
from services.llm_service import LLMService, create_llm_service
from app.rag.quiz_rag import QuizRAGService
from database.models.quiz_models import (
    QuizGenerationRequest, UserAnswer, QuizSubmission, 
    QuizQuestion, QuizResult, DifficultyLevel
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quiz", tags=["quiz"])

# Initialize Llama3 service for local development
def initialize_llama3_service():
    """Initialize LLM service with local Llama3 model via Ollama"""
    try:
        llm_service = create_llm_service(
            provider="ollama",
            model_name="llama3",  # Using local Llama3 model
            temperature=0.7,
            max_tokens=2000
        )
        return llm_service
    except Exception as e:
        raise Exception(f"Could not connect to local Llama3 via Ollama. Make sure Ollama is running: {str(e)}")

# Initialize services
llm_service = initialize_llama3_service()
quiz_service = QuizService(llm_service.model)

from pydantic import BaseModel
from typing import List, Dict, Any

# Create a response model for the wrapped format
quiz_sessions = {}

class SimpleAnswersRequest(BaseModel):
    session_id: str
    answers: List[str]  # ["B", "A", "B"]

def store_quiz_session(questions: List[QuizQuestion]) -> str:
    """Store quiz questions and return session ID"""
    session_id = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    quiz_sessions[session_id] = {
        'questions': questions,
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(hours=24)
    }
    return session_id

def clean_expired_sessions():
    """Remove expired quiz sessions"""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, data in quiz_sessions.items()
        if data['expires_at'] < current_time
    ]
    for session_id in expired_sessions:
        del quiz_sessions[session_id]

# ROUTE 1: Generate Quiz (Modified to include session)
@router.post("/generate")
async def generate_quiz_from_uploaded_file(
    file: UploadFile = File(...),
    num_questions: int = Form(8),
    difficulty: str = Form("medium")
):
    """
    Generate quiz questions from uploaded document
    Returns: {"session_id": "...", "questions": [...]}
    """
    tmp_file_path = None
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Validate parameters
        if num_questions < 1 or num_questions > 20:
            raise HTTPException(
                status_code=400,
                detail="Number of questions must be between 1 and 20"
            )
        
        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(
                status_code=400,
                detail="Difficulty must be one of: easy, medium, hard"
            )
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Create temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tmp_file_path = tmp_file.name
        
        # Write content and close file properly
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()
        
        # Verify file was created successfully
        if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
            raise HTTPException(
                status_code=500, 
                detail="Failed to create temporary file"
            )
        
        logger.info(f"Created temporary file: {tmp_file_path}, size: {os.path.getsize(tmp_file_path)}")
        
        # Process document
        documents = quiz_service.process_document_for_quiz(tmp_file_path)
        if not documents:
            raise HTTPException(
                status_code=422, 
                detail="Could not process the uploaded document. Please ensure the file is readable and contains text content."
            )
        
        # Generate quiz questions
        quiz_response = quiz_service.generate_quiz_questions(
            num_questions=num_questions,
            difficulty=difficulty,
            question_type="multiple_choice"
        )
        
        # Validate that we have questions
        if not quiz_response.get('questions'):
            raise HTTPException(
                status_code=422, 
                detail="Could not generate questions from the uploaded document. The document may not contain enough content for quiz generation."
            )
        
        # Convert to QuizQuestion objects for storage
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
        
        # Store questions and get session ID
        session_id = store_quiz_session(questions)
        
        # Clean expired sessions
        clean_expired_sessions()
        
        # Return quiz with session ID
        return {
            "session_id": session_id,
            "questions": quiz_response['questions']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.info(f"Cleaned up temporary file: {tmp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {tmp_file_path}: {str(cleanup_error)}")

class QuizGenerationRequest(BaseModel):
    num_questions: int
    difficulty: str 
    question_type: Optional[str] = "multiple_choice"

class TextQuizRequest(BaseModel):
    text_content: str
    request: QuizGenerationRequest

@router.post("/generate-from-text")
async def generate_quiz_from_text(data: TextQuizRequest):
    """
    Generate quiz questions directly from provided text content using Llama3
    Returns: {"session_id": "...", "questions": [...]}
    """
    try:
        if not data.text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        if len(data.text_content) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Text content too short. Please provide at least 100 characters for meaningful quiz generation."
            )
        
        # Use Llama3 service for text-based quiz generation
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
        
        # Convert to QuizQuestion objects
        questions = []
        quiz_questions_for_response = []  # For the response format
        
        for i, q_data in enumerate(questions_data):
            try:
                # Handle different option formats
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
                            "label": chr(65 + j),  # A, B, C, D
                            "text": str(opt),
                            "is_correct": False
                        })
                
                # Create QuizQuestion object for storage
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
                
                # Create question data for response (similar to the /generate endpoint)
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
        
        # Store questions and get session ID (same as /generate endpoint)
        session_id = store_quiz_session(questions)
        
        # Clean expired sessions
        clean_expired_sessions()
        
        # Return quiz with session ID (matching /generate endpoint format)
        return {
            "session_id": session_id,
            "questions": quiz_questions_for_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz with Llama3: {str(e)}")
from pydantic import BaseModel

class QuizEvaluationRequest(BaseModel):
    questions: List[QuizQuestion]
    submission: QuizSubmission


@router.post("/evaluate", response_model=QuizResult)
async def evaluate_quiz(request: SimpleAnswersRequest):
    """
    Evaluate quiz with simple answers
    Input: {"session_id": "quiz_...", "answers": ["B", "A", "B"]}
    """
    try:
        # Clean expired sessions first
        clean_expired_sessions()
        
        # Get stored questions
        if request.session_id not in quiz_sessions:
            raise HTTPException(
                status_code=404, 
                detail="Quiz session not found or expired. Please generate a new quiz."
            )
        
        session_data = quiz_sessions[request.session_id]
        questions = session_data['questions']
        
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
        
        # Evaluate quiz using existing service
        results = quiz_service.evaluate_quiz(questions, user_answers)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating quiz: {str(e)}")