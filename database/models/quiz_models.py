from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"

class QuizGenerationRequest(BaseModel):
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Difficulty level")
    question_type: QuestionType = Field(default=QuestionType.MULTIPLE_CHOICE, description="Type of questions")
    topics: Optional[List[str]] = Field(default=None, description="Specific topics to focus on")
    
    @validator('num_questions')
    def validate_num_questions(cls, v):
        if v < 1 or v > 20:
            raise ValueError('Number of questions must be between 1 and 20')
        return v

class QuizOption(BaseModel):
    label: str = Field(..., description="Option label (A, B, C, D)")
    text: str = Field(..., description="Option text")
    is_correct: bool = Field(default=False, description="Whether this option is correct")

class QuizQuestion(BaseModel):
    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="Question text")
    options: List[QuizOption] = Field(..., description="Answer options")
    correct_answer: str = Field(..., description="Correct answer label")
    explanation: str = Field(..., description="Explanation for the correct answer")
    difficulty: DifficultyLevel = Field(..., description="Question difficulty")
    topic: Optional[str] = Field(default=None, description="Question topic/category")
    context: Optional[str] = Field(default=None, description="Source context from document")
    
    @validator('options')
    def validate_options(cls, v):
        if len(v) < 2:
            raise ValueError('Question must have at least 2 options')
        return v
    
    @validator('correct_answer')
    def validate_correct_answer(cls, v, values):
        if 'options' in values:
            option_labels = [opt.label for opt in values['options']]
            if v not in option_labels:
                raise ValueError('Correct answer must match one of the option labels')
        return v

class UserAnswer(BaseModel):
    question_id: str = Field(..., description="Question ID")
    selected_answer: str = Field(..., description="Selected answer label")
    time_taken: Optional[int] = Field(default=None, description="Time taken in seconds")

class QuizSubmission(BaseModel):
    quiz_id: Optional[str] = Field(default=None, description="Quiz identifier")
    answers: List[UserAnswer] = Field(..., description="User answers")
    submission_time: datetime = Field(default_factory=datetime.now, description="Submission timestamp")
    
    @validator('answers')
    def validate_answers(cls, v):
        if not v:
            raise ValueError('At least one answer must be provided')
        return v

class QuestionResult(BaseModel):
    question_id: str
    question: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str
    topic: Optional[str] = None

class QuizResult(BaseModel):
    quiz_id: Optional[str] = None
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    score_percentage: float
    grade: str
    results: List[QuestionResult]
    completion_time: datetime = Field(default_factory=datetime.now)
    
    @validator('score_percentage')
    def validate_score(cls, v):
        return round(v, 2)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.grade = self._calculate_grade(self.score_percentage)
    
    def _calculate_grade(self, percentage: float) -> str:
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

class QuizSession(BaseModel):
    session_id: str
    questions: List[QuizQuestion]
    created_at: datetime = Field(default_factory=datetime.now)
    difficulty: DifficultyLevel
    total_questions: int
    document_name: Optional[str] = None
    
    @validator('total_questions')
    def validate_total_questions(cls, v, values):
        if 'questions' in values and v != len(values['questions']):
            raise ValueError('Total questions must match the number of questions provided')
        return v

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_name: Optional[str] = None
    document_summary: Optional[Dict[str, Any]] = None
    supported_formats: List[str] = Field(default=[".pdf", ".txt", ".docx", ".doc"])

class QuizGenerationResponse(BaseModel):
    success: bool
    message: str
    quiz_session: Optional[QuizSession] = None
    error_details: Optional[str] = None

class QuizEvaluationResponse(BaseModel):
    success: bool
    message: str
    quiz_result: Optional[QuizResult] = None
    error_details: Optional[str] = None

# Additional models for API responses
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.now)

class DifficultyOption(BaseModel):
    value: str
    label: str
    description: Optional[str] = None

class QuestionLimits(BaseModel):
    min_questions: int = 1
    max_questions: int = 20
    default_questions: int = 5
    recommended_questions: List[int] = Field(default=[5, 10, 15, 20])

class ServiceConfiguration(BaseModel):
    difficulties: List[DifficultyOption]
    question_limits: QuestionLimits
    supported_formats: List[str]
    question_types: List[str]

# Error models
class QuizError(BaseModel):
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ValidationError(QuizError):
    field_errors: Optional[Dict[str, str]] = None

# Configuration models
class QuizConfig(BaseModel):
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = Field(default=[".pdf", ".txt", ".docx", ".doc"])
    default_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    default_questions: int = 5
    max_questions_per_quiz: int = 20
    question_generation_timeout: int = 300  # seconds
    
class RAGConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_for_question: int = 3
    similarity_threshold: float = 0.7