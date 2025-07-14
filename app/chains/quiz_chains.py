from langchain.schema import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import logging
import random
import hashlib
from database.models.quiz_models import QuizQuestion, QuizOption, DifficultyLevel

logger = logging.getLogger(__name__)

class QuizQuestionParser(BaseOutputParser):
    """Custom parser for quiz question responses with enhanced JSON handling"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured quiz question format"""
        try:
            # Clean the text first
            text = text.strip()
            
            # DEBUG: Log the raw response
            logger.debug(f"Raw LLM response (first 500 chars): {text[:500]}...")
            logger.debug(f"Raw LLM response length: {len(text)}")
            
            # Try multiple parsing strategies in order of preference
            parsing_strategies = [
                self._parse_direct_json,
                self._parse_extracted_json,
                self._parse_cleaned_json,
                self._parse_with_ast_literal_eval,
                self._manual_parse
            ]
            
            for strategy in parsing_strategies:
                try:
                    result = strategy(text)
                    if result and self._validate_parsed_result(result):
                        logger.debug(f"Successfully parsed with {strategy.__name__}: {result}")
                        return result
                except Exception as e:
                    logger.debug(f"{strategy.__name__} failed: {str(e)}")
                    continue
            
            # If all strategies fail, create fallback
            logger.warning("All parsing strategies failed, creating fallback question")
            return self._create_fallback_question(text)
            
        except Exception as e:
            logger.error(f"Error parsing quiz question: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            return self._create_fallback_question(text)
    
    def _parse_direct_json(self, text: str) -> Dict[str, Any]:
        """Try parsing as direct JSON"""
        if text.startswith('{') and text.endswith('}'):
            return json.loads(text)
        raise ValueError("Not direct JSON format")
    
    def _parse_extracted_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text using regex"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        raise ValueError("No JSON found in text")
    
    def _parse_cleaned_json(self, text: str) -> Dict[str, Any]:
        """Clean and parse JSON with comprehensive quote fixing"""
        cleaned_text = text
        
        # Remove any text before the first { and after the last }
        if '{' in cleaned_text and '}' in cleaned_text:
            start = cleaned_text.index('{')
            end = cleaned_text.rindex('}') + 1
            cleaned_text = cleaned_text[start:end]
        
        # Comprehensive quote fixing
        cleaned_text = self._fix_json_quotes(cleaned_text)
        
        # Remove trailing commas
        cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)
        
        # Fix any remaining issues
        cleaned_text = self._fix_json_syntax(cleaned_text)
        
        logger.debug(f"Cleaned JSON: {cleaned_text}")
        return json.loads(cleaned_text)
    
    def _fix_json_quotes(self, text: str) -> str:
        """Comprehensive quote fixing for JSON"""
        # Replace single quotes around keys with double quotes
        text = re.sub(r"(\s*)'([^']*)'(\s*:)", r'\1"\2"\3', text)
        
        # Replace single quotes around simple string values (not containing escaped quotes)
        text = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r': "\1"', text)
        
        # Handle array elements with single quotes
        text = re.sub(r'\[\s*\'([^\']*)\'\s*\]', r'["\1"]', text)
        
        # Handle single quotes in arrays more comprehensively
        pattern = r'(\[|\,)\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'\s*(?=\,|\])'
        text = re.sub(pattern, r'\1 "\2"', text)
        
        return text
    
    def _fix_json_syntax(self, text: str) -> str:
        """Fix common JSON syntax issues"""
        # Ensure boolean values are lowercase
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        
        # Remove any trailing commas before closing brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix spacing around colons and commas
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        return text
    
    def _parse_with_ast_literal_eval(self, text: str) -> Dict[str, Any]:
        """Try parsing with ast.literal_eval for Python-style syntax"""
        import ast
        
        # Extract dictionary-like structure
        dict_match = re.search(r'\{.*\}', text, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group()
            # Try to evaluate as Python literal
            try:
                result = ast.literal_eval(dict_str)
                if isinstance(result, dict):
                    return result
            except (ValueError, SyntaxError):
                pass
        
        raise ValueError("Could not parse with ast.literal_eval")
    
    def _validate_parsed_result(self, result: Dict[str, Any]) -> bool:
        """Validate that the parsed result has the required structure"""
        required_keys = ['question', 'options', 'correct_answer', 'explanation']
        
        if not isinstance(result, dict):
            return False
        
        for key in required_keys:
            if key not in result:
                logger.debug(f"Missing required key: {key}")
                return False
        
        if not isinstance(result['options'], list) or len(result['options']) < 2:
            logger.debug("Invalid options format")
            return False
        
        # Validate correct_answer format
        correct_answer = result['correct_answer']
        if not isinstance(correct_answer, str):
            logger.debug("correct_answer is not a string")
            return False
        
        # Normalize to uppercase letter
        normalized_answer = correct_answer.strip().upper()
        if len(normalized_answer) > 0 and normalized_answer[0] in 'ABCD':
            result['correct_answer'] = normalized_answer[0]
            return True
        
        logger.debug(f"Invalid correct_answer format: {correct_answer}")
        return False
    
    def _manual_parse(self, text: str) -> Dict[str, Any]:
        """Manual parsing when JSON parsing fails"""
        logger.debug(f"Starting manual parsing on text: {text}")
        
        # Try to find structured content with more flexible patterns
        question_match = re.search(r'["\']?question["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        options_match = re.search(r'["\']?options["\']?\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r'["\']?correct_answer["\']?\s*:\s*["\']?([ABCD])["\']?', text, re.IGNORECASE)
        explanation_match = re.search(r'["\']?explanation["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        
        # Extract question
        question = question_match.group(1) if question_match else "Generated question from content"
        
        # Extract options
        options = []
        if options_match:
            options_str = options_match.group(1)
            # Extract individual options
            option_pattern = r'["\']([^"\']+)["\']'
            option_matches = re.findall(option_pattern, options_str)
            options = option_matches[:4]  # Take first 4 options
        
        # Ensure we have 4 options
        if len(options) < 4:
            default_options = ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
            options = options + default_options[len(options):]
        
        # Extract correct answer
        correct_answer = answer_match.group(1).upper() if answer_match else "A"
        
        # Extract explanation
        explanation = explanation_match.group(1) if explanation_match else "Based on the provided content"
        
        result = {
            "question": question,
            "options": options[:4],
            "correct_answer": correct_answer,
            "explanation": explanation
        }
        
        logger.debug(f"Manual parsing result: {result}")
        return result
    
    def _create_fallback_question(self, text: str) -> Dict[str, Any]:
        """Create a fallback question when parsing completely fails"""
        logger.warning("Creating fallback question due to parsing failure")
        return {
            "question": "What is the main topic discussed in the provided content?",
            "options": ["A) Topic A", "B) Topic B", "C) Topic C", "D) Topic D"],
            "correct_answer": "A",
            "explanation": "This is a fallback question generated when parsing failed."
        }

class QuizChains:
    """Modern LangChain chains for quiz generation using LCEL with diversity mechanisms"""
    
    def __init__(self, llm):
        self.llm = llm
        self.question_parser = QuizQuestionParser()
        self.used_questions = set()  # Track used questions
        self.question_styles = [
            "analytical", "factual", "conceptual", "application", "critical thinking"
        ]
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup all the chains for quiz generation using LCEL"""
        logger.debug("Setting up quiz chains")
        self.multiple_choice_chain = self._create_multiple_choice_chain()
        self.true_false_chain = self._create_true_false_chain()
        self.difficulty_analyzer_chain = self._create_difficulty_analyzer_chain()
        self.topic_extractor_chain = self._create_topic_extractor_chain()
        logger.debug("Quiz chains setup complete")
    
    def _create_multiple_choice_chain(self) -> RunnableSequence:
        """Create chain for multiple choice questions with diversity prompts"""
        template = """
Based on the following content, create a {difficulty} level multiple choice question with a {style} approach.

Content: {content}

Question Style Guidelines:
- Analytical: Focus on analyzing relationships, patterns, or causes
- Factual: Test specific facts, definitions, or details
- Conceptual: Test understanding of main concepts and principles
- Application: Test ability to apply knowledge to new situations
- Critical thinking: Require evaluation, comparison, or judgment

Additional Diversity Instructions:
- Question ID: {question_id} (use this to create unique variations)
- Focus on: {focus_aspect}
- Avoid these patterns: {avoid_patterns}
- Make this question distinctly different from typical questions about this topic

Requirements:
- Question should test understanding of the key concepts using the {style} style
- Provide exactly 4 options (A, B, C, D)
- Only one option should be correct
- Make incorrect options plausible but clearly wrong
- Provide a clear explanation for the correct answer
- Ensure the question is unique and not repetitive

CRITICAL: Your response must be VALID JSON only. No additional text before or after.
Use ONLY double quotes for all strings. Do NOT use single quotes anywhere.
Follow this exact format:

{{
    "question": "Your {style} question here",
    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
    "correct_answer": "A",
    "explanation": "Explanation of why this answer is correct"
}}
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "difficulty", "style", "question_id", "focus_aspect", "avoid_patterns"]
        )
        
        return prompt | self.llm | self.question_parser
    
    def _create_true_false_chain(self) -> RunnableSequence:
        """Create chain for true/false questions with diversity prompts"""
        template = """
Based on the following content, create a {difficulty} level true/false question with a {style} approach.

Content: {content}

Question Style Guidelines:
- Analytical: Focus on analyzing relationships, patterns, or causes
- Factual: Test specific facts, definitions, or details
- Conceptual: Test understanding of main concepts and principles
- Application: Test ability to apply knowledge to new situations
- Critical thinking: Require evaluation, comparison, or judgment

Additional Diversity Instructions:
- Question ID: {question_id} (use this to create unique variations)
- Focus on: {focus_aspect}
- Avoid these patterns: {avoid_patterns}
- Make this question distinctly different from typical questions about this topic

Requirements:
- Question should be clearly true or false based on the content using the {style} style
- Avoid ambiguous statements
- Provide a clear explanation
- Ensure the question is unique and not repetitive

CRITICAL: Your response must be VALID JSON only. No additional text before or after.
Use ONLY double quotes for all strings. Do NOT use single quotes anywhere.
Follow this exact format:

{{
    "question": "Your {style} true/false statement here",
    "options": ["A) True", "B) False"],
    "correct_answer": "A",
    "explanation": "Explanation of why this is true/false"
}}
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "difficulty", "style", "question_id", "focus_aspect", "avoid_patterns"]
        )
        
        return prompt | self.llm | self.question_parser
    
    def _create_difficulty_analyzer_chain(self) -> RunnableSequence:
        """Create chain to analyze content difficulty using LCEL"""
        template = """
Analyze the following content and determine its complexity level.

Content: {content}

Consider:
- Vocabulary complexity
- Concept difficulty
- Technical depth
- Required background knowledge

Respond with one word: "easy", "medium", or "hard"

Difficulty:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content"]
        )
        
        return prompt | self.llm
    
    def _create_topic_extractor_chain(self) -> RunnableSequence:
        """Create chain to extract main topics from content using LCEL"""
        template = """
Extract the main topics/themes from the following content.

Content: {content}

List the top 3 main topics, separated by commas.
Focus on the most important concepts or subjects discussed.

Topics:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content"]
        )
        
        return prompt | self.llm
    
    def _get_question_hash(self, question: str) -> str:
        """Generate a hash for a question to detect duplicates"""
        # Normalize the question for comparison
        normalized = re.sub(r'[^\w\s]', '', question.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_diversity_params(self, question_id: int, content: str) -> Dict[str, str]:
        """Get diversity parameters for question generation"""
        # Cycle through different question styles
        style = self.question_styles[question_id % len(self.question_styles)]
        
        # Generate different focus aspects
        focus_aspects = [
            "specific details and facts",
            "underlying principles and concepts",
            "practical applications",
            "relationships and connections",
            "implications and consequences",
            "processes and procedures",
            "comparisons and contrasts",
            "cause and effect relationships"
        ]
        focus_aspect = focus_aspects[question_id % len(focus_aspects)]
        
        # Generate avoid patterns based on question number
        avoid_patterns = [
            "simple definition questions",
            "yes/no questions without depth",
            "questions starting with 'What is'",
            "questions with obvious answers",
            "questions that test memorization only"
        ]
        avoid_pattern = avoid_patterns[question_id % len(avoid_patterns)]
        
        return {
            "style": style,
            "question_id": f"q_{question_id}",
            "focus_aspect": focus_aspect,
            "avoid_patterns": avoid_pattern
        }
    
    def generate_multiple_choice_question(self, content: str, difficulty: str = "medium", question_id: int = 1) -> Dict[str, Any]:
        """Generate a multiple choice question from content with diversity"""
        try:
            logger.debug(f"Generating multiple choice question {question_id} with difficulty: {difficulty}")
            logger.debug(f"Content length: {len(content)}")
            
            # Get diversity parameters
            diversity_params = self._get_diversity_params(question_id, content)
            
            # Add some randomness to content selection
            if len(content) > 1500:
                max_start = len(content) - 1500
                start_pos = random.randint(0, max_start)
                content_chunk = content[start_pos:start_pos + 1500]
            else:
                content_chunk = content
            
            result = self.multiple_choice_chain.invoke({
                "content": content_chunk,
                "difficulty": difficulty,
                **diversity_params
            })
            
            # Check for duplicates
            question_hash = self._get_question_hash(result.get("question", ""))
            if question_hash in self.used_questions:
                logger.warning(f"Duplicate question detected, regenerating...")
                # Add more randomness and try again
                diversity_params["question_id"] = f"q_{question_id}_retry_{random.randint(1, 100)}"
                diversity_params["focus_aspect"] = random.choice([
                    "unique aspects and special cases",
                    "advanced implications",
                    "contextual applications",
                    "interdisciplinary connections"
                ])
                result = self.multiple_choice_chain.invoke({
                    "content": content_chunk,
                    "difficulty": difficulty,
                    **diversity_params
                })
            
            # Track this question
            self.used_questions.add(question_hash)
            
            logger.debug(f"Generated question result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating multiple choice question: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            return self._create_error_question()
    
    def generate_true_false_question(self, content: str, difficulty: str = "medium", question_id: int = 1) -> Dict[str, Any]:
        """Generate a true/false question from content with diversity"""
        try:
            logger.debug(f"Generating true/false question {question_id} with difficulty: {difficulty}")
            logger.debug(f"Content length: {len(content)}")
            
            # Get diversity parameters
            diversity_params = self._get_diversity_params(question_id, content)
            
            # Add some randomness to content selection
            if len(content) > 1500:
                max_start = len(content) - 1500
                start_pos = random.randint(0, max_start)
                content_chunk = content[start_pos:start_pos + 1500]
            else:
                content_chunk = content
            
            result = self.true_false_chain.invoke({
                "content": content_chunk,
                "difficulty": difficulty,
                **diversity_params
            })
            
            # Check for duplicates
            question_hash = self._get_question_hash(result.get("question", ""))
            if question_hash in self.used_questions:
                logger.warning(f"Duplicate true/false question detected, regenerating...")
                # Add more randomness and try again
                diversity_params["question_id"] = f"q_{question_id}_retry_{random.randint(1, 100)}"
                diversity_params["focus_aspect"] = random.choice([
                    "specific factual accuracy",
                    "conceptual correctness",
                    "logical validity",
                    "contextual truth"
                ])
                result = self.true_false_chain.invoke({
                    "content": content_chunk,
                    "difficulty": difficulty,
                    **diversity_params
                })
            
            # Track this question
            self.used_questions.add(question_hash)
            
            logger.debug(f"Generated true/false result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating true/false question: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            return self._create_error_question(question_type="true_false")
    
    def analyze_content_difficulty(self, content: str) -> str:
        """Analyze the difficulty level of content"""
        try:
            logger.debug(f"Analyzing content difficulty for {len(content)} characters")
            
            result = self.difficulty_analyzer_chain.invoke({"content": content[:1000]})
            
            logger.debug(f"Difficulty analysis raw result: {result}")
            
            # Handle different response types
            if hasattr(result, 'content'):
                difficulty = result.content.strip().lower()
            else:
                difficulty = str(result).strip().lower()
            
            logger.debug(f"Extracted difficulty: {difficulty}")
            
            if difficulty in ["easy", "medium", "hard"]:
                return difficulty
            else:
                logger.warning(f"Invalid difficulty '{difficulty}', defaulting to medium")
                return "medium"  # Default fallback
                
        except Exception as e:
            logger.error(f"Error analyzing content difficulty: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            return "medium"
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content"""
        try:
            logger.debug(f"Extracting topics from {len(content)} characters")
            
            result = self.topic_extractor_chain.invoke({"content": content[:1000]})
            
            logger.debug(f"Topic extraction raw result: {result}")
            
            # Handle different response types
            if hasattr(result, 'content'):
                topics_text = result.content.strip()
            else:
                topics_text = str(result).strip()
            
            logger.debug(f"Topics text: {topics_text}")
            
            topics = [topic.strip() for topic in topics_text.split(',')]
            logger.debug(f"Extracted topics: {topics}")
            
            return topics[:3]  # Return top 3 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            return ["General Topic"]
    
    def clear_used_questions(self):
        """Clear the set of used questions (call this when starting a new quiz)"""
        self.used_questions.clear()
        logger.debug("Cleared used questions set")
    
    def _create_error_question(self, question_type: str = "multiple_choice") -> Dict[str, Any]:
        """Create a fallback question when generation fails"""
        logger.warning(f"Creating error question for type: {question_type}")
        
        if question_type == "true_false":
            return {
                "question": "The provided content contains useful information.",
                "options": ["A) True", "B) False"],
                "correct_answer": "A",
                "explanation": "This is a fallback question due to generation error."
            }
        else:
            return {
                "question": "What type of content was provided?",
                "options": ["A) Text content", "B) No content", "C) Invalid content", "D) Unknown content"],
                "correct_answer": "A",
                "explanation": "This is a fallback question due to generation error."
            }

class QuizGenerationPipeline:
    """Enhanced pipeline for diverse quiz generation"""
    
    def __init__(self, quiz_chains: QuizChains, quiz_rag_service):
        self.chains = quiz_chains
        self.rag_service = quiz_rag_service
        
    def generate_quiz_questions(self, 
                              content_chunks: List[str], 
                              num_questions: int = 8,
                              difficulty: str = "medium",
                              question_type: str = "multiple_choice") -> List[QuizQuestion]:
        """Generate a diverse set of quiz questions"""
        logger.debug(f"Starting quiz generation: {num_questions} questions, difficulty: {difficulty}, type: {question_type}")
        logger.debug(f"Content chunks: {len(content_chunks)}")
        
        # Clear previous questions to start fresh
        self.chains.clear_used_questions()
        
        questions = []
        
        # Shuffle content chunks for more diversity
        shuffled_chunks = content_chunks.copy()
        random.shuffle(shuffled_chunks)
        
        # Create varied content segments
        content_segments = self._create_varied_content_segments(shuffled_chunks, num_questions)
        
        for i in range(num_questions):
            try:
                logger.debug(f"Generating question {i+1}/{num_questions}")
                
                # Use varied content segments
                content = content_segments[i]
                
                logger.debug(f"Using content segment {i}, length: {len(content)}")
                
                # Generate question based on type with question ID for diversity
                if question_type == "true_false":
                    logger.debug("Generating true/false question")
                    question_data = self.chains.generate_true_false_question(content, difficulty, i+1)
                else:
                    logger.debug("Generating multiple choice question")
                    question_data = self.chains.generate_multiple_choice_question(content, difficulty, i+1)
                
                logger.debug(f"Question data received: {question_data}")
                
                # Validate question data
                if not self._validate_question_data(question_data):
                    logger.error(f"Invalid question data for question {i+1}")
                    continue
                
                # Extract topic from content
                topics = self.chains.extract_topics(content)
                topic = topics[0] if topics else "General"
                
                logger.debug(f"Topic extracted: {topic}")
                
                # Create QuizQuestion object with proper option handling
                options = []
                option_labels = []
                for opt_idx, opt_text in enumerate(question_data["options"]):
                    logger.debug(f"Processing option {opt_idx}: {opt_text}")
                    
                    # Handle different option formats
                    label, text = self._process_option(opt_text, opt_idx, option_labels)
                    option_labels.append(label)
                    
                    # Determine if this option is correct
                    is_correct = (label == question_data["correct_answer"])
                    
                    logger.debug(f"Option created: {label}) {text} (correct: {is_correct})")
                    
                    options.append(QuizOption(
                        label=label,
                        text=text,
                        is_correct=is_correct
                    ))
                
                # Create question ID with hash for uniqueness
                question_hash = hashlib.md5(question_data["question"].encode()).hexdigest()[:8]
                question_id = f"q_{i+1}_{question_hash}"
                
                quiz_question = QuizQuestion(
                    id=question_id,
                    question=question_data["question"],
                    options=options,
                    correct_answer=question_data["correct_answer"],
                    explanation=question_data["explanation"],
                    difficulty=DifficultyLevel(difficulty),
                    topic=topic,
                    context=content[:200] + "..." if len(content) > 200 else content
                )
                
                questions.append(quiz_question)
                logger.debug(f"Question {i+1} created successfully")
                
            except Exception as e:
                logger.error(f"Error generating question {i+1}: {str(e)}")
                logger.error(f"Full error details: {repr(e)}")
                
                # Create a fallback question
                fallback_question = self._create_fallback_quiz_question(i+1, difficulty)
                if fallback_question:
                    questions.append(fallback_question)
                    logger.debug(f"Fallback question {i+1} created")
        
        logger.debug(f"Quiz generation complete: {len(questions)} questions created")
        return questions
    
    def _process_option(self, opt_text: Any, opt_idx: int, existing_labels: List[str]) -> Tuple[str, str]:
        """Process and normalize an option text"""
        if isinstance(opt_text, str):
            # Handle formatted options like "A) Option text"
            if len(opt_text) > 2 and opt_text[1] == ')':
                label = opt_text[0].upper()
                text = opt_text[3:]
            # Handle formats like "A. Option text"
            elif len(opt_text) > 2 and opt_text[1] == '.':
                label = opt_text[0].upper()
                text = opt_text[2:].lstrip()
            else:
                # Assign next available label
                label = self._get_next_option_label(opt_idx, existing_labels)
                text = opt_text
        else:
            # Handle non-string options
            label = self._get_next_option_label(opt_idx, existing_labels)
            text = str(opt_text)
        
        return label, text
    
    def _get_next_option_label(self, opt_idx: int, existing_labels: List[str]) -> str:
        """Get the next available option label"""
        # Standard labels for first 4 options
        standard_labels = ['A', 'B', 'C', 'D']
        
        if opt_idx < len(standard_labels):
            label = standard_labels[opt_idx]
            # Ensure label isn't already used
            if label not in existing_labels:
                return label
        
        # If standard labels exhausted, use next letter
        next_char = chr(65 + len(existing_labels))
        return next_char
    
    def _create_varied_content_segments(self, content_chunks: List[str], num_questions: int) -> List[str]:
        """Create varied content segments for diverse question generation"""
        segments = []
        
        for i in range(num_questions):
            chunk_index = i % len(content_chunks)
            content = content_chunks[chunk_index]
            
            # Create different segments from the same content
            if len(content) > 1000:
                # For longer content, create different segments
                segment_size = min(1500, len(content) // 3)
                segment_start = (i * 317) % max(1, len(content) - segment_size)  # Use prime number for better distribution
                segment = content[segment_start:segment_start + segment_size]
            else:
                segment = content
            
            segments.append(segment)
        
        return segments

    def _validate_question_data(self, question_data: dict) -> bool:
        """Validate that question_data has required fields"""
        logger.debug(f"Validating question data: {question_data}")
        
        required_fields = ["question", "options", "correct_answer", "explanation"]
    
        for field in required_fields:
            if field not in question_data:
                logger.error(f"Missing required field: {field}")
                return False
    
        if not isinstance(question_data["options"], list) or len(question_data["options"]) < 2:
            logger.error("Invalid options: must be a list with at least 2 options")
            return False
        
        # Validate correct_answer format
        correct_answer = question_data["correct_answer"]
        if not isinstance(correct_answer, str) or len(correct_answer) != 1:
            logger.error(f"Invalid correct_answer format: {correct_answer}")
            return False
        if correct_answer.upper() not in 'ABCD':
            logger.error(f"correct_answer must be A, B, C, or D: {correct_answer}")
            return False
    
        logger.debug("Question data validation passed")
        return True
    
    def _create_fallback_quiz_question(self, question_id: int, difficulty: str) -> Optional[QuizQuestion]:
        """Create a fallback quiz question when generation fails"""
        try:
            logger.debug(f"Creating fallback question {question_id}")
            
            options = [
                QuizOption(label="A", text="Option A", is_correct=True),
                QuizOption(label="B", text="Option B", is_correct=False),
                QuizOption(label="C", text="Option C", is_correct=False),
                QuizOption(label="D", text="Option D", is_correct=False)
            ]
            
            fallback_question = QuizQuestion(
                id=f"fallback_q_{question_id}",
                question="This is a fallback question due to generation error.",
                options=options,
                correct_answer="A",
                explanation="This question was generated as a fallback due to an error.",
                difficulty=DifficultyLevel(difficulty),
                topic="General",
                context="Fallback context"
            )
            
            logger.debug(f"Fallback question {question_id} created successfully")
            return fallback_question
            
        except Exception as e:
            logger.error(f"Error creating fallback question: {str(e)}")
            return None