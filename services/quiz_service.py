from typing import List, Dict, Any, Optional
import json
import random
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from database.models.quiz_models import QuizQuestion, QuizOption, DifficultyLevel, QuestionResult, QuizResult
from app.rag.quiz_rag import QuizRAGService
from app.chains.quiz_chains import QuizChains, QuizGenerationPipeline

logger = logging.getLogger(__name__)

class QuizService:
    def __init__(self, llm_model):
        """
        Initialize Quiz Service with integrated RAG and chains
        """
        self.llm_model = llm_model
        self.quiz_rag_service = QuizRAGService()
        self.quiz_chains = QuizChains(llm_model)
        self.quiz_pipeline = QuizGenerationPipeline(self.quiz_chains, self.quiz_rag_service)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        logger.info("Quiz Service initialized with RAG and chains")
    
    def process_document_for_quiz(self, file_path: str) -> bool:
        """
        Process uploaded document for quiz generation
        """
        try:
            success = self.quiz_rag_service.setup_quiz_rag(file_path)
            if success:
                logger.info(f"Document {file_path} processed successfully for quiz")
                return True
            else:
                logger.error(f"Failed to process document {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return False
    
    def generate_quiz_questions(self, num_questions: int = 10, difficulty: str = "medium", question_type: str = "multiple_choice") -> Dict[str, Any]:
        """
        Generate quiz questions using the integrated pipeline in formatted JSON structure
        """
        try:
            if not self.quiz_rag_service.documents:
                logger.error("No documents loaded for quiz generation")
                return {"questions": []}
            
            # Get diverse content chunks
            content_chunks = self.quiz_rag_service.get_context_for_question_generation()
            
            if not content_chunks:
                logger.error("No content chunks available for question generation")
                return {"questions": []}
            
            # Generate questions using the pipeline
            questions = self.quiz_pipeline.generate_quiz_questions(
                content_chunks=content_chunks,
                num_questions=num_questions,
                difficulty=difficulty,
                question_type=question_type
            )
            
            # Convert to the exact format with proper structure
            formatted_questions = []
            for i, question in enumerate(questions, 1):
                # Convert options to the required format with label, text, and is_correct
                formatted_options = []
                for opt in question.options:
                    formatted_options.append({
                        "label": opt.label,
                        "text": opt.text,
                        "is_correct": opt.is_correct
                    })
                
                # Get context from RAG service for this question
                context = ""
                try:
                    # Try to get the context that was used to generate this question
                    if hasattr(question, 'context') and question.context:
                        context = question.context
                    else:
                        # Fallback: get relevant context from RAG service
                        if content_chunks and len(content_chunks) > 0:
                            # Use the first available context chunk or try to find relevant one
                            context = content_chunks[0] if isinstance(content_chunks[0], str) else str(content_chunks[0])
                            # Truncate context if too long
                            if len(context) > 500:
                                context = context[:500] + "..."
                except Exception as ctx_error:
                    logger.warning(f"Could not retrieve context for question {i}: {str(ctx_error)}")
                    context = f"Context from processed document for question {i}"
                
                formatted_question = {
                    "id": question.id,
                    "question": question.question,
                    "options": formatted_options,
                    "correct_answer": question.correct_answer,
                    "explanation": question.explanation,
                    "difficulty": difficulty,
                    "topic": getattr(question, 'topic', 'General'),
                    "context": context
                }
                formatted_questions.append(formatted_question)
            
            # Return in the exact format with "questions" wrapper
            result = {
                "questions": formatted_questions
            }
            
            logger.info(f"Generated {len(questions)} quiz questions in formatted structure")
            return result
            
        except Exception as e:
            logger.error(f"Error generating quiz questions: {str(e)}")
            return {"questions": []}
    def evaluate_quiz(self, questions: List[QuizQuestion], user_answers: Dict[str, str]) -> QuizResult:
        """
        Evaluate quiz answers and return comprehensive results
        """
        try:
            total_questions = len(questions)
            correct_answers = 0
            results = []
        
            for question in questions:
                question_id = question.id
                correct_answer = question.correct_answer
                user_answer = user_answers.get(question_id, "")
            
                is_correct = user_answer == correct_answer
                if is_correct:
                    correct_answers += 1
            
            # Create detailed result for each question
                question_result = QuestionResult(
                question_id=question_id,
                question=question.question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                explanation=question.explanation,
                topic=question.topic
            )
            
            results.append(question_result)
        
        # Calculate score
            score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Calculate grade based on score percentage
            if score_percentage >= 90:
                grade = "A"
            elif score_percentage >= 80:
                grade = "B"
            elif score_percentage >= 70:
                grade = "C"
            elif score_percentage >= 60:
                grade = "D"
            else:
                grade = "F"
        
        # Create comprehensive quiz result
            quiz_result = QuizResult(
            total_questions=total_questions,
            correct_answers=correct_answers,
            incorrect_answers=total_questions - correct_answers,
            score_percentage=score_percentage,
            grade=grade,
            results=results
            )
        
            logger.info(f"Quiz evaluated: {correct_answers}/{total_questions} correct ({score_percentage:.1f}%)")
            return quiz_result
        
        except Exception as e:
            logger.error(f"Error evaluating quiz: {str(e)}")
        # Return empty result on error
            return QuizResult(
                 total_questions=0,
                correct_answers=0,
                 incorrect_answers=0,
                 score_percentage=0.0,
                 grade="F",
                results=[]
            )
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get summary of the processed document
        """
        return self.quiz_rag_service.get_document_summary()
    
    def cleanup_resources(self):
        """
        Clean up quiz service resources
        """
        self.quiz_rag_service.cleanup()
        logger.info("Quiz service resources cleaned up")
    
    def generate_quiz_from_uploaded_file(self, file_path: str, num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Generate quiz from uploaded file in the exact format matching paste.txt
        """
        try:
            success = self.process_document_for_quiz(file_path)
            if not success:
                return {"questions": []}
            
            questions = self.generate_quiz_questions(num_questions, difficulty)
            
            # Convert to the exact format shown in paste.txt
            formatted_questions = []
            for i, question in enumerate(questions, 1):
                # Convert options to the required format with label, text, and is_correct
                formatted_options = []
                for opt in question.options:
                    formatted_options.append({
                        "label": opt.label,
                        "text": opt.text,
                        "is_correct": opt.is_correct
                    })
                
                # Get context from RAG service for this question
                context = ""
                try:
                    # Try to get the context that was used to generate this question
                    if hasattr(question, 'context') and question.context:
                        context = question.context
                    else:
                        # Fallback: get relevant context from RAG service
                        context_chunks = self.quiz_rag_service.get_context_for_question_generation()
                        if context_chunks and len(context_chunks) > 0:
                            # Use the first available context chunk or try to find relevant one
                            context = context_chunks[0] if isinstance(context_chunks[0], str) else str(context_chunks[0])
                            # Truncate context if too long
                            if len(context) > 500:
                                context = context[:500] + "..."
                except Exception as ctx_error:
                    logger.warning(f"Could not retrieve context for question {i}: {str(ctx_error)}")
                    context = f"Context from uploaded document for question {i}"
                
                formatted_question = {
                    "id": question.id,
                    "question": question.question,
                    "options": formatted_options,
                    "correct_answer": question.correct_answer,
                    "explanation": question.explanation,
                    "difficulty": difficulty,
                    "topic": getattr(question, 'topic', 'General'),
                    "context": context
                }
                formatted_questions.append(formatted_question)
            
            # Return in the exact format with "questions" wrapper
            result = {
                "questions": formatted_questions
            }
            
            logger.info(f"Generated {len(formatted_questions)} questions in required format")
            return result
            
        except Exception as e:
            logger.error(f"Error generating quiz from uploaded file: {str(e)}")
            return {"questions": []}