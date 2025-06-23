from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import tempfile
import os

import logging

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
class QuizResponse(BaseModel):
    questions: List[QuizQuestion]

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz_from_uploaded_file(
    file: UploadFile = File(...),
    num_questions: int = Form(5),
    difficulty: str = Form("medium")
):
    """
    Generate quiz questions from uploaded document using Llama3
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
        
        # Read file content first
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Create temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tmp_file_path = tmp_file.name
        
        try:
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
            
            # Generate quiz questions - this returns {"questions": [...]}
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
            
            # Return the full dictionary structure with "questions" wrapper
            return quiz_response
            
        except Exception as e:
            # Re-raise HTTPExceptions as-is
            if isinstance(e, HTTPException):
                raise
            else:
                logger.error(f"Error in quiz generation pipeline: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating quiz with Llama3: {str(e)}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_quiz_from_upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz with Llama3: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.info(f"Cleaned up temporary file: {tmp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {tmp_file_path}: {str(cleanup_error)}")


@router.post("/generate-from-text", response_model=List[QuizQuestion])
async def generate_quiz_from_text(
    text_content: str,
    request: QuizGenerationRequest
):
    """
    Generate quiz questions directly from provided text content using Llama3
    """
    try:
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        if len(text_content) < 100:
            raise HTTPException(
                status_code=400, 
                detail="Text content too short. Please provide at least 100 characters for meaningful quiz generation."
            )
        
        # Use Llama3 service for text-based quiz generation
        questions_data = llm_service.generate_quiz_questions(
            content=text_content,
            num_questions=request.num_questions,
            difficulty=request.difficulty.value,
            question_type=request.question_type.value
        )
        
        if not questions_data:
            raise HTTPException(
                status_code=422, 
                detail="Could not generate questions from the provided text using Llama3"
            )
        
        # Convert to QuizQuestion objects
        questions = []
        for i, q_data in enumerate(questions_data):
            try:
                # Create QuizQuestion from the Llama3 response
                question = QuizQuestion(
                    id=q_data.get('id', f'q{i+1}'),
                    question=q_data['question'],
                    options=[
                        {
                            "label": opt['label'],
                            "text": opt['text'],
                            "is_correct": opt.get('is_correct', False)
                        }
                        for opt in q_data['options']
                    ],
                    correct_answer=q_data['correct_answer'],
                    explanation=q_data.get('explanation', ''),
                    difficulty=DifficultyLevel(q_data.get('difficulty', request.difficulty.value)),
                    topic=q_data.get('topic', 'General')
                )
                questions.append(question)
            except KeyError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid question format from Llama3: missing {str(e)}"
                )
        
        return questions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz with Llama3: {str(e)}")
from pydantic import BaseModel

class QuizEvaluationRequest(BaseModel):
    questions: List[QuizQuestion]
    submission: QuizSubmission

@router.post("/evaluate", response_model=QuizResult)
async def evaluate_quiz(request: QuizEvaluationRequest):
    """
    Evaluate quiz answers and return results using Llama3
    """
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided for evaluation")
        
        if not request.submission.answers:
            raise HTTPException(status_code=400, detail="No answers provided for evaluation")
        
        # Convert answers to dictionary format
        user_answers = {answer.question_id: answer.selected_answer for answer in request.submission.answers}
        
        # Evaluate quiz using the service (powered by Llama3)
        results = quiz_service.evaluate_quiz(request.questions, user_answers)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating quiz with Llama3: {str(e)}")

@router.post("/explain-answer")
async def explain_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
    context: str = ""
):
    """
    Get detailed explanation for a quiz answer using Llama3
    """
    try:
        explanation = llm_service.evaluate_answer_explanation(
            question=question,
            correct_answer=correct_answer,
            user_answer=user_answer,
            context=context
        )
        
        return {
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "explanation": explanation,
            "generated_by": "llama3"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation with Llama3: {str(e)}")

@router.get("/health")
async def quiz_health_check():
    """
    Health check endpoint for quiz service and Llama3
    """
    try:
        # Check quiz service
        quiz_health = {"status": "healthy", "service": "quiz"}
        
        # Check Llama3 service
        llm_health = llm_service.health_check()
        
        return {
            "overall_status": "healthy" if llm_health["status"] == "healthy" else "degraded",
            "quiz_service": quiz_health,
            "llm_service": {
                **llm_health,
                "model": "llama3",
                "provider": "ollama"
            },
            "timestamp": llm_health["timestamp"]
        }
        
    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "quiz_service": {"status": "unknown"},
            "llm_service": {
                "status": "unhealthy",
                "model": "llama3",
                "provider": "ollama"
            }
        }

@router.get("/config")
async def get_quiz_configuration():
    """
    Get quiz service configuration for Llama3
    """
    return {
        "difficulties": [
            {"value": "easy", "label": "Easy", "description": "Basic comprehension questions"},
            {"value": "medium", "label": "Medium", "description": "Moderate analysis questions"},
            {"value": "hard", "label": "Hard", "description": "Complex reasoning questions"}
        ],
        "question_limits": {
            "min_questions": 1,
            "max_questions": 20,
            "default_questions": 5,
            "recommended_questions": [5, 10, 15, 20]
        },
        "supported_formats": [".pdf", ".txt", ".docx", ".doc"],
        "question_types": ["multiple_choice", "true_false", "fill_blank"],
        "llm_provider": "ollama",
        "llm_model": "llama3",
        "environment": "local_development"
    }

@router.get("/model-info")
async def get_model_info():
    """
    Get information about the local Llama3 model
    """
    try:
        return {
            "model_name": "llama3",
            "provider": "ollama",
            "environment": "local",
            "temperature": 0.7,
            "max_tokens": 2000,
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/document-summary")
async def get_document_summary():
    """
    Get summary of the currently processed document (generated by Llama3)
    """
    try:
        summary = quiz_service.get_document_summary()
        return {
            **summary,
            "generated_by": "llama3"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document summary: {str(e)}")

@router.post("/cleanup")
async def cleanup_resources():
    """
    Clean up quiz service resources
    """
    try:
        quiz_service.cleanup_resources()
        return {"status": "success", "message": "Resources cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up resources: {str(e)}")

@router.post("/restart-llama3")
async def restart_llama3_service():
    """
    Restart the local Llama3 service connection
    """
    try:
        global llm_service, quiz_service
        
        # Reinitialize local Llama3 service
        llm_service = initialize_llama3_service()
        quiz_service = QuizService(llm_service.model)
        
        return {
            "status": "success", 
            "message": "Local Llama3 service restarted successfully",
            "model": "llama3",
            "provider": "ollama"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restarting local Llama3 service: {str(e)}")

# Legacy endpoints for backward compatibility
@router.get("/difficulties")
async def get_difficulty_levels():
    """
    Get available difficulty levels (legacy endpoint)
    """
    return {
        "difficulties": [
            {"value": "easy", "label": "Easy"},
            {"value": "medium", "label": "Medium"},
            {"value": "hard", "label": "Hard"}
        ]
    }

@router.get("/question-limits")
async def get_question_limits():
    """
    Get min/max limits for number of questions (legacy endpoint)
    """
    return {
        "min_questions": 1,
        "max_questions": 20,
        "default_questions": 5
    }