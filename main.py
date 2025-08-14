#!/usr/bin/env python3
"""
AI System Main Entry Point
Local AI system with offline prediction capabilities and optional online updates
"""

import os
import sys
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import existing components
from ai_engine import AIModel, ModelManager, OnlineUpdater
from ai_engine.enhanced_ai import EnhancedAI
from ai_engine.advanced_core import AdvancedAICore
from ai_engine.smart_ai import SmartAI

# Import new self-improving modules
from ai_engine.self_evaluation import SelfEvaluationModule
from ai_engine.error_analysis import ErrorAnalysisModule
from ai_engine.iterative_learning import IterativeLearningModule
from ai_engine.knowledge_retrieval import KnowledgeRetrievalModule
from utils import (
    initialize_ai_storage,
    preprocess_input,
    LogManager,
    DatabaseManager,
    JSONManager,
    ConfigManager,
    UtilityFunctions,
    PerformanceMonitor,
    CacheManager,
    handle_errors,
    measure_performance,
)


class AISystemCLI:
    """Command-line interface for the AI System"""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize AI System CLI

        Args:
            config_path: Configuration file path
        """
        # Load configuration
        self.config = ConfigManager.load_config(config_path)
        self.storage_path = self.config["storage_path"]

        # Initialize logging
        self.logger = LogManager.setup_logging(
            level=self.config["log_level"], log_file=self.config.get("log_file")
        )

        # Generate session ID
        self.session_id = UtilityFunctions.generate_session_id()
        self.logger.info(f"Starting new session: {self.session_id}")

        # Initialize components
        self.model = None
        self.model_manager = None
        self.updater = None
        self.enhanced_ai = EnhancedAI()  # Initialize enhanced AI
        self.advanced_ai = AdvancedAICore()  # Initialize advanced AI core
        self.smart_ai = SmartAI()  # Initialize smart AI with auto-correct
        
        # Initialize self-improving modules
        self.self_evaluation = SelfEvaluationModule()
        self.error_analysis = ErrorAnalysisModule()
        self.iterative_learning = IterativeLearningModule()
        self.knowledge_retrieval = KnowledgeRetrievalModule()
        
        self.performance_monitor = PerformanceMonitor(self.storage_path)
        self.cache = CacheManager()

        # Session tracking
        self.interaction_count = 0
        self.session_start_time = time.time()

        # Initialize storage and components
        self._initialize_system()

    @handle_errors(default_return=False, log_errors=True)
    def _initialize_system(self) -> bool:
        """
        Initialize storage system and AI components

        Returns:
            True if initialization successful
        """
        try:
            # Initialize storage system
            self.logger.info("Initializing storage system...")
            initialize_ai_storage(self.storage_path)

            # Initialize model manager
            self.model_manager = ModelManager(storage_path=self.storage_path)

            # Initialize online updater
            updater_config = self.config.get("updater", {})
            self.updater = OnlineUpdater(
                storage_path=self.storage_path, config=updater_config
            )

            # Load or create default model
            self._load_or_create_model()

            # Check for updates if enabled
            if self.config.get("auto_update_check", False):
                self._check_for_updates()

            self.logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            # Ensure we have a basic model even if initialization fails
            try:
                self.logger.info("Creating fallback model...")
                self.model = AIModel(model_type="text_classifier")
                self._train_default_model()
                return True
            except Exception as fallback_error:
                self.logger.error(f"Fallback model creation failed: {fallback_error}")
                return False

    @handle_errors(default_return=False)
    def _load_or_create_model(self) -> bool:
        """
        Load existing model or create new default model

        Returns:
            True if model loaded/created successfully
        """
        try:
            # Try to load existing models
            available_models = self.model_manager.list_models()

            if available_models:
                # Load the most recent model
                model_info = max(available_models, key=lambda x: x.get("created_at", ""))
                model_name = model_info["name"]

                self.logger.info(f"Loading existing model: {model_name}")
                self.model = self.model_manager.load_model(model_name)

                if self.model:
                    self.logger.info(f"Successfully loaded model: {model_name}")
                    return True

            # Create new default model if no existing model found
            self.logger.info("Creating new default model...")
            default_model_type = self.config.get("default_model_type", "text_classifier")

            self.model = AIModel(model_type=default_model_type)

            # Train with some default data if available
            self._train_default_model()

            # Save the new model
            model_name = f"default_{default_model_type}_{int(time.time())}"
            success = self.model_manager.save_model(self.model, model_name)

            if success:
                self.logger.info(f"Created and saved new default model: {model_name}")
                return True
            else:
                self.logger.error("Failed to save default model")
                return False
        except Exception as e:
            self.logger.error(f"Error in _load_or_create_model: {e}")
            return False

    def _train_default_model(self) -> None:
        """Train default model with sample data"""
        try:
            if self.model.model_type == "text_classifier":
                # Sample text data for initial training
                sample_texts = [
                    "This is a positive example",
                    "This is a negative example",
                    "Another positive case",
                    "Another negative case",
                    "Good example text",
                    "Bad example text",
                ]
                sample_labels = [1, 0, 1, 0, 1, 0]  # Binary classification

                self.model.train(sample_texts, sample_labels)
                self.logger.info("Trained default text classifier with sample data")

            elif self.model.model_type == "numerical_classifier":
                # Sample numerical data
                import numpy as np

                sample_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                sample_labels = [0, 1, 0, 1]

                self.model.train(sample_data, sample_labels)
                self.logger.info(
                    "Trained default numerical classifier with sample data"
                )

        except Exception as e:
            self.logger.warning(f"Failed to train default model: {e}")

    @handle_errors(default_return=False)
    def _check_for_updates(self) -> bool:
        """
        Check for and apply online updates

        Returns:
            True if update check completed
        """
        try:
            self.logger.info("Checking for online updates...")

            has_updates = self.updater.check_for_updates()

            if has_updates:
                self.logger.info("Updates available, performing update...")
                success = self.updater.perform_full_update()

                if success:
                    self.logger.info("Updates applied successfully")
                    # Reload model if it was updated
                    self._load_or_create_model()
                else:
                    self.logger.warning("Update failed")
            else:
                self.logger.info("No updates available")

            return True

        except Exception as e:
            self.logger.warning(f"Update check failed: {e}")
            return False

    @measure_performance(log_result=True)
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate AI response with simplified logic

        Args:
            user_input: User's input text

        Returns:
            Dictionary with response data
        """
        start_time = time.time()

        try:
            # Handle special commands
            if user_input.strip().lower().startswith("feedback "):
                feedback_result = self.self_evaluation.process_feedback_command(user_input)
                return {
                    "success": True,
                    "response": feedback_result,
                    "prediction": "feedback_processed",
                    "confidence": 1.0,
                    "data_type": "command",
                    "model_type": "feedback_system",
                    "response_time": time.time() - start_time,
                    "cached": False,
                }

            # Validate input
            if not user_input or not user_input.strip():
                return {
                    "success": False,
                    "error": "Empty input provided",
                    "response": "Please provide some input for me to process.",
                }

            # Check cache for repeated queries
            cache_key = UtilityFunctions.calculate_hash(user_input.strip().lower())
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self.logger.debug("Returning cached result")
                cached_result["cached"] = True
                return cached_result

            # Simplified processing logic - use Enhanced AI as primary
            try:
                enhanced_result = self.enhanced_ai.process_query(user_input)
                
                if enhanced_result['success']:
                    response_text = enhanced_result['response']
                    confidence = enhanced_result.get('confidence', 0.8)
                    data_type = enhanced_result.get('type', 'enhanced')
                    prediction = 'enhanced_response'
                    model_type = "enhanced_ai"
                    
                    # Learn from this successful interaction
                    self.enhanced_ai.learn_from_interaction(user_input, response_text)
                else:
                    # Fallback to basic response
                    response_text = "I understand your query. Let me provide you with relevant information."
                    confidence = 0.5
                    data_type = 'fallback'
                    prediction = 'fallback_response'
                    model_type = "fallback"
                    
            except Exception as e:
                self.logger.error(f"Enhanced AI error: {e}")
                response_text = "I'm here to help! What would you like to know?"
                confidence = 0.5
                data_type = 'error_fallback'
                prediction = 'error_fallback'
                model_type = "error_fallback"

            # Calculate response time
            response_time = time.time() - start_time

            # Create result dictionary
            result = {
                "success": True,
                "response": response_text,
                "prediction": prediction,
                "confidence": confidence,
                "data_type": data_type,
                "model_type": model_type,
                "response_time": response_time,
                "cached": False,
            }

            # Cache the result
            self.cache.set(cache_key, result)

            # Self-Evaluation (simplified)
            try:
                evaluation = self.self_evaluation.evaluate_response(
                    question=user_input,
                    answer=response_text,
                    response_time=response_time,
                    model_type=model_type,
                    context_info={
                        "predicted_model": model_type,
                        "model_confidence": confidence
                    }
                )
                result["evaluation_score"] = evaluation.overall_score
            except Exception as e:
                self.logger.error(f"Self-evaluation error: {e}")

            # Error Analysis (simplified)
            try:
                error_analysis = self.error_analysis.analyze_response(
                    question=user_input,
                    answer=response_text,
                    model_type=model_type
                )
                result["error_analysis"] = error_analysis
            except Exception as e:
                self.logger.error(f"Error analysis error: {e}")

            # Learn from interaction (simplified)
            try:
                self.iterative_learning.learn_from_interaction(
                    question=user_input,
                    answer=response_text,
                    model_type=model_type,
                    score=result.get("evaluation_score", 0.5),
                    success=result.get("evaluation_score", 0.5) >= 0.6
                )
            except Exception as e:
                self.logger.error(f"Learning error: {e}")

            return result

        except Exception as e:
            self.logger.error(f"Error in process_user_input: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your request. Please try again.",
                "response_time": time.time() - start_time,
            }

    def _generate_response(
        self, prediction: Any, confidence: float, data_type: str, user_input: str = ""
    ) -> str:
        """
        Generate human-readable response from prediction

        Args:
            prediction: Model prediction
            confidence: Prediction confidence
            data_type: Type of input data
            user_input: Original user input for context

        Returns:
            Human-readable response text
        """
        confidence_threshold = self.config.get("model_settings", {}).get(
            "confidence_threshold", 0.5
        )

        # Generate conversational responses based on user input
        user_input_lower = user_input.lower().strip()
        
        # Handle common greetings and questions
        if any(word in user_input_lower for word in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            return "Hello! I'm your AI assistant. How can I help you today?"
        
        if any(word in user_input_lower for word in ["who are you", "what are you", "your name"]):
            return "I'm an AI assistant designed to help you with various tasks. I can process text, answer questions, and provide insights. How can I assist you?"
        
        if any(word in user_input_lower for word in ["how are you", "how do you do"]):
            return "I'm functioning well, thank you for asking! I'm ready to help you with any questions or tasks you might have."
        
        if any(word in user_input_lower for word in ["what can you do", "help", "capabilities"]):
            return "I can help you with text analysis, answer questions, provide insights, and assist with various tasks. Just ask me anything!"
        
        if any(word in user_input_lower for word in ["thank you", "thanks", "thx"]):
            return "You're welcome! I'm happy to help. Is there anything else you'd like to know?"
        
        if any(word in user_input_lower for word in ["bye", "goodbye", "see you", "exit", "quit"]):
            return "Goodbye! Feel free to return if you need any more assistance."
        
        # Handle questions
        if "?" in user_input:
            if any(word in user_input_lower for word in ["what", "how", "why", "when", "where", "who"]):
                return f"That's an interesting question about '{user_input}'. Based on my analysis, I can provide insights and help you find answers. What specific aspect would you like to explore?"
        
        # Handle statements and general text
        if data_type == "text":
            if isinstance(prediction, (int, float)):
                if prediction == 1:
                    return f"I understand you're expressing something positive. That's great! I'm here to help and support you with whatever you need."
                elif prediction == 0:
                    return f"I sense this might be challenging or negative. I'm here to help and support you. Would you like to talk more about it or is there something specific I can assist you with?"
                else:
                    return f"I've analyzed your input and processed it successfully. How can I help you further with this topic?"
            else:
                return f"I've processed your input and understand what you're saying. How can I assist you with this?"
        
        elif data_type == "numerical":
            return f"I've analyzed the numerical data you provided. Based on my calculations, here are the insights I can offer. Would you like me to explain the results in more detail?"
        
        else:
            return f"I've processed your input successfully. How can I help you with this information?"

        # Fallback response
        if confidence < confidence_threshold:
            return f"I'm processing your input. While I'm not completely certain about the classification (confidence: {confidence:.2f}), I'm here to help you with whatever you need."
        
        return "I understand your input and I'm here to help. What would you like to know or discuss?"

    def _log_interaction(
        self, user_input: str, ai_response: str, confidence: float, response_time: float
    ) -> None:
        """
        Log user interaction to database

        Args:
            user_input: User's input
            ai_response: AI system response
            confidence: Prediction confidence
            response_time: Response time in seconds
        """
        db_path = Path(self.storage_path) / "database.db"

        success = DatabaseManager.log_interaction(
            db_path=db_path,
            session_id=self.session_id,
            user_input=user_input,
            ai_response=ai_response,
            confidence=confidence,
            model_used=self.model.model_type if self.model else "unknown",
            response_time=response_time,
        )

        if success:
            self.interaction_count += 1
        else:
            self.logger.warning("Failed to log interaction to database")

    def collect_feedback(
        self, rating: int, correction: str = "", notes: str = ""
    ) -> bool:
        """
        Collect user feedback for incremental learning

        Args:
            rating: User rating (1-5)
            correction: Corrected response if applicable
            notes: Additional feedback notes

        Returns:
            True if feedback processed successfully
        """
        try:
            # Log feedback to database
            db_path = Path(self.storage_path) / "database.db"

            feedback_logged = DatabaseManager.log_feedback(
                db_path=db_path,
                session_id=self.session_id,
                interaction_id=self.interaction_count,
                feedback_type="rating",
                feedback_value=rating,
                notes=f"correction: {correction}, notes: {notes}",
            )

            if not feedback_logged:
                self.logger.warning("Failed to log feedback to database")

            # Apply incremental learning if correction provided
            if correction and hasattr(self.model, "incremental_learning"):
                try:
                    # Get last user input from interaction history
                    last_input = self._get_last_user_input()

                    if last_input:
                        # Process correction as new training data
                        processed_correction, data_type = preprocess_input(correction)

                        # Create training label based on rating
                        label = 1 if rating >= 4 else 0

                        # Apply incremental learning
                        self.model.incremental_learning([processed_correction], [label])

                        # Save updated model
                        model_name = (
                            f"updated_{self.model.model_type}_{int(time.time())}"
                        )
                        self.model_manager.save_model(self.model, model_name)

                        self.logger.info(
                            "Applied incremental learning from user feedback"
                        )

                except Exception as e:
                    self.logger.error(f"Incremental learning failed: {e}")

            # Update knowledge base
            knowledge_updates = {
                "user_preferences": {
                    "feedback_style": "corrective" if correction else "rating_only"
                },
                "interaction_history": [
                    {
                        "session_id": self.session_id,
                        "interaction_id": self.interaction_count,
                        "rating": rating,
                        "timestamp": time.time(),
                    }
                ],
            }

            JSONManager.update_knowledge_json(self.storage_path, knowledge_updates)

            return True

        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
            return False

    def _get_last_user_input(self) -> Optional[str]:
        """
        Get the last user input from database

        Returns:
            Last user input or None
        """
        try:
            db_path = Path(self.storage_path) / "database.db"

            query = """
            SELECT user_input FROM user_interactions 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
            """

            results = DatabaseManager.execute_query(
                db_path, query, (self.session_id,), fetch=True
            )

            if results:
                return results[0]["user_input"]

        except Exception as e:
            self.logger.error(f"Failed to get last user input: {e}")

        return None

    def display_help(self) -> None:
        """Display help information"""
        help_text = """
╔════════════════════════════════════════════════════════════════════════╗
║                           AI SYSTEM HELP                               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ COMMANDS:                                                              ║
║   help        - Show this help message                                 ║
║   exit/quit   - Exit the AI system                                     ║
║   status      - Show system status                                     ║
║   models      - List available models                                  ║
║   update      - Check for online updates                               ║
║   feedback    - Provide feedback on last response                      ║
║   clear       - Clear screen                                           ║
║   stats       - Show session statistics                                ║
║   clear_learning - Clear all learning data                            ║
║   retrain       - Force immediate model retraining                     ║
║   clear_cache   - Clear knowledge retrieval cache                      ║
║   learning_stats - Show iterative learning statistics                  ║
║   error_stats   - Show error analysis statistics                       ║
║   performance_stats - Show self-evaluation statistics                  ║
║                                                                        ║
║ SELF-IMPROVING FEATURES:                                               ║
║   • 🤖 Self-Evaluation: Automatic response scoring and feedback        ║
║   • 🔍 Error Analysis: Detects failures and fetches corrections        ║
║   • 🧠 Iterative Learning: Continuous model improvement                ║
║   • 🌐 Knowledge Retrieval: Multi-source information gathering         ║
║   • 📊 Performance Monitoring: Real-time statistics and metrics        ║
║   • 🔄 Auto-Retraining: Periodic model updates every 24 hours          ║
║   • 💬 Manual Feedback: Use "feedback good/bad <reason>" commands      ║
║                                                                        ║
║ USAGE:                                                                 ║
║   • Type any text or question to get AI predictions                    ║
║   • The system automatically detects text vs numerical data            ║
║   • 🧠 Smart AI: Auto-corrects typos and provides context-aware responses ║
║   • 🚀 Advanced AI: Multi-source search and information synthesis      ║
║   • ⚡ Enhanced AI: Continuous learning and evolution                   ║
║   • AI remembers conversations and provides intelligent responses      ║
║   • Continuously learns and updates with latest information            ║
║   • Provide feedback to improve future predictions                     ║
║   • Use 'feedback' command after any interaction                       ║
║                                                                        ║
║ EXAMPLES:                                                              ║
║   > What is the weather like?                                          ║
║   > 1.5 2.3 4.1                                                       ║
║   > feedback                                                           ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
        """
        print(help_text)

    def display_status(self) -> None:
        """Display system status"""
        uptime = time.time() - self.session_start_time

        status_info = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                          SYSTEM STATUS                                 ║
╠════════════════════════════════════════════════════════════════════════╣
║ Session ID: {self.session_id:<54} ║
║ Uptime: {uptime/3600:.1f} hours{'':<52} ║
║ Interactions: {self.interaction_count:<57} ║
║ Current Model: {self.model.model_type if self.model else 'None':<53} ║
║ Storage Path: {self.storage_path:<54} ║
║ Cache Entries: {len(self.cache.cache):<55} ║
╚════════════════════════════════════════════════════════════════════════╝
        """
        print(status_info)

    def display_models(self) -> None:
        """Display available models"""
        try:
            models = self.model_manager.list_models()

            print("\n" + "=" * 70)
            print("AVAILABLE MODELS")
            print("=" * 70)

            if not models:
                print("No models found.")
                return

            for i, model in enumerate(models, 1):
                print(f"{i}. {model['name']}")
                print(f"   Type: {model.get('model_type', 'Unknown')}")
                print(f"   Created: {model.get('created_at', 'Unknown')}")
                print(f"   Size: {model.get('size', 'Unknown')}")
                print()

        except Exception as e:
            print(f"Error retrieving models: {e}")

    def display_stats(self) -> None:
        """Display session statistics"""
        try:
            # Performance metrics
            response_time_stats = self.performance_monitor.get_metric_summary(
                "response_time"
            )
            confidence_stats = self.performance_monitor.get_metric_summary("confidence")

            print("\n" + "=" * 70)
            print("SESSION STATISTICS")
            print("=" * 70)
            print(f"Total Interactions: {self.interaction_count}")
            print(
                f"Session Duration: {(time.time() - self.session_start_time)/60:.1f} minutes"
            )

            if response_time_stats:
                print(f"Average Response Time: {response_time_stats['mean']:.3f}s")
                print(f"Fastest Response: {response_time_stats['min']:.3f}s")
                print(f"Slowest Response: {response_time_stats['max']:.3f}s")

            if confidence_stats:
                print(f"Average Confidence: {confidence_stats['mean']:.3f}")
                print(f"Highest Confidence: {confidence_stats['max']:.3f}")
                print(f"Lowest Confidence: {confidence_stats['min']:.3f}")

            print(f"Cache Hit Ratio: {len(self.cache.cache)} cached items")
            
            # AI Components Statistics
            print("\n" + "=" * 70)
            print("🤖 AI COMPONENTS STATISTICS")
            print("=" * 70)
            
            # Smart AI Stats
            try:
                smart_stats = self.smart_ai.get_smart_stats()
                if smart_stats:
                    print("🧠 SMART AI (Auto-correct & Context):")
                    print(f"  • Conversations: {smart_stats.get('total_conversations', 0)}")
                    print(f"  • Knowledge Entries: {smart_stats.get('knowledge_entries', 0)}")
                    print(f"  • Typo Corrections: {smart_stats.get('typo_corrections', 0)}")
                    print(f"  • Current Topic: {smart_stats.get('current_context', {}).get('topic', 'None')}")
            except Exception as e:
                print(f"  • Smart AI stats error: {e}")
            
            # Advanced AI Stats
            try:
                advanced_stats = self.advanced_ai.get_advanced_stats()
                if advanced_stats:
                    print("🚀 ADVANCED AI (Multi-source & Synthesis):")
                    print(f"  • Interactions: {advanced_stats.get('total_interactions', 0)}")
                    print(f"  • Knowledge Base: {advanced_stats.get('knowledge_entries', 0)}")
                    print(f"  • Code Snippets: {advanced_stats.get('code_snippets', 0)}")
                    print(f"  • Avg Processing Time: {advanced_stats.get('avg_processing_time', 0):.3f}s")
                    print(f"  • Cache Size: {advanced_stats.get('cache_size', 0)}")
            except Exception as e:
                print(f"  • Advanced AI stats error: {e}")
            
            # Enhanced AI Stats
            try:
                evolution_stats = self.enhanced_ai.get_evolution_stats()
                if evolution_stats:
                    print("⚡ ENHANCED AI (Learning & Evolution):")
                    print(f"  • Evolution Level: {evolution_stats.get('evolution_level', 'Unknown')}")
                    print(f"  • Total Interactions: {evolution_stats.get('total_interactions', 0)}")
                    print(f"  • Knowledge Entries: {evolution_stats.get('knowledge_entries', 0)}")
                    print(f"  • User Preferences: {evolution_stats.get('user_preferences', 0)}")
                    print(f"  • Learning Patterns: {evolution_stats.get('learning_patterns', 0)}")
                    print(f"  • Average Rating: {evolution_stats.get('average_rating', 0):.2f}")
            except Exception as e:
                print(f"  • Enhanced AI stats error: {e}")
            
            print("=" * 70)

        except Exception as e:
            print(f"Error retrieving statistics: {e}")

    def _display_learning_statistics(self) -> None:
        """Display iterative learning statistics"""
        try:
            print("\n" + "=" * 70)
            print("🧠 ITERATIVE LEARNING STATISTICS")
            print("=" * 70)
            
            stats = self.iterative_learning.get_learning_statistics()
            
            if stats:
                model_info = stats.get("model_info", {})
                print(f"Model Path: {model_info.get('model_path', 'N/A')}")
                print(f"Has Trained Model: {model_info.get('has_trained_model', False)}")
                print(f"Number of Classes: {model_info.get('num_classes', 0)}")
                
                embeddings_info = stats.get("embeddings_info", {})
                print(f"Number of Embeddings: {embeddings_info.get('num_embeddings', 0)}")
                print(f"Similarity Threshold: {embeddings_info.get('similarity_threshold', 0.0)}")
                print(f"Max Results: {embeddings_info.get('max_results', 0)}")
                
                training_info = stats.get("training_info", {})
                print(f"Minimum Training Samples: {training_info.get('min_samples', 0)}")
                print(f"Retraining Interval: {training_info.get('retraining_interval', 'N/A')}")
                print(f"Last Retraining: {training_info.get('last_retraining', 'Never')}")
            else:
                print("No learning statistics available")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"Error retrieving learning statistics: {e}")

    def _display_error_statistics(self) -> None:
        """Display error analysis statistics"""
        try:
            print("\n" + "=" * 70)
            print("🔍 ERROR ANALYSIS STATISTICS")
            print("=" * 70)
            
            stats = self.error_analysis.get_error_statistics()
            
            if stats:
                overall = stats.get("overall", {})
                print(f"Total Errors: {overall.get('total_errors', 0)}")
                print(f"Corrected Errors: {overall.get('corrected_errors', 0)}")
                print(f"Correction Rate: {overall.get('correction_rate', 0.0):.1f}%")
                print(f"Average Confidence: {overall.get('average_confidence', 0.0):.3f}")
                
                error_types = stats.get("error_types", [])
                if error_types:
                    print("\nError Types Distribution:")
                    for error_type in error_types:
                        print(f"  • {error_type['type']}: {error_type['count']}")
                
                correction_sources = stats.get("correction_sources", [])
                if correction_sources:
                    print("\nCorrection Sources Effectiveness:")
                    for source in correction_sources:
                        print(f"  • {source['source']}: {source['count']} corrections (avg confidence: {source['avg_confidence']:.3f})")
            else:
                print("No error statistics available")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"Error retrieving error statistics: {e}")

    def _display_performance_statistics(self) -> None:
        """Display self-evaluation performance statistics"""
        try:
            print("\n" + "=" * 70)
            print("📊 SELF-EVALUATION PERFORMANCE STATISTICS")
            print("=" * 70)
            
            stats = self.self_evaluation.get_performance_stats()
            
            if stats:
                overall = stats.get("overall", {})
                print(f"Total Responses: {overall.get('total_responses', 0)}")
                print(f"Average Score: {overall.get('average_score', 0.0):.3f}")
                print(f"Average Response Time: {overall.get('average_response_time', 0.0):.3f}s")
                print(f"Success Rate: {overall.get('success_rate', 0.0):.1f}%")
                
                models = stats.get("models", [])
                if models:
                    print("\nModel Performance:")
                    for model in models:
                        print(f"  • {model['model_type']}: {model['average_score']:.3f} score, {model['successful_responses']}/{model['total_responses']} successful")
                
                recent_evaluations = stats.get("recent_evaluations", [])
                if recent_evaluations:
                    print("\nRecent Evaluations:")
                    for i, eval_item in enumerate(recent_evaluations[:5], 1):
                        print(f"  {i}. {eval_item['question']} → Score: {eval_item['score']:.3f} ({eval_item['model_type']})")
            else:
                print("No performance statistics available")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"Error retrieving performance statistics: {e}")

    def handle_feedback_command(self) -> None:
        """Handle interactive feedback collection"""
        try:
            print("\n" + "=" * 50)
            print("FEEDBACK COLLECTION")
            print("=" * 50)

            # Get rating
            while True:
                try:
                    rating = int(input("Rate the last response (1-5): "))
                    if 1 <= rating <= 5:
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")

            # Get optional correction
            correction = input("Provide correction (optional): ").strip()

            # Get optional notes
            notes = input("Additional notes (optional): ").strip()

            # Process feedback
            success = self.collect_feedback(rating, correction, notes)

            if success:
                print(
                    "✓ Thank you for your feedback! It will help improve future responses."
                )
            else:
                print("✗ There was an error processing your feedback.")

        except KeyboardInterrupt:
            print("\nFeedback cancelled.")
        except Exception as e:
            print(f"Error collecting feedback: {e}")
    
    def _clear_learning_data(self) -> None:
        """Clear all learning data to start fresh"""
        try:
            print("\n" + "=" * 50)
            print("CLEARING LEARNING DATA")
            print("=" * 50)
            
            # Clear enhanced AI data
            self.enhanced_ai.conversation_history = []
            self.enhanced_ai.knowledge_base = {}
            self.enhanced_ai.user_preferences = {}
            self.enhanced_ai.current_conversation_context = {
                'topic': None,
                'last_questions': [],
                'related_info': {},
                'conversation_chain': []
            }
            self.enhanced_ai.typo_corrections = {}
            self.enhanced_ai.data_feeds = {
                'last_news_update': None,
                'last_weather_update': None,
                'last_knowledge_update': None
            }
            
            # Save empty data
            self.enhanced_ai._save_persistent_data()
            
            print("✅ All learning data cleared successfully!")
            print("The AI will now start learning fresh from new interactions.")
            print("Enhanced features: Memory, Context Awareness, Auto-correct, Continuous Learning")
            
        except Exception as e:
            print(f"❌ Error clearing learning data: {e}")

    def run(self) -> None:
        """
        Main command-line interface loop
        """
        print("\n" + "=" * 70)
        print("🤖 AI SYSTEM - Local Prediction with Online Updates")
        print("=" * 70)
        print(f"Session: {self.session_id}")
        print("Type 'help' for commands or 'exit' to quit")
        print("=" * 70 + "\n")

        try:
            while True:
                try:
                    # Get user input
                    user_input = input("🔤 You: ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.lower() in ["exit", "quit"]:
                        self._cleanup_and_exit()
                        break

                    elif user_input.lower() == "help":
                        self.display_help()
                        continue

                    elif user_input.lower() == "status":
                        self.display_status()
                        continue

                    elif user_input.lower() == "models":
                        self.display_models()
                        continue

                    elif user_input.lower() == "update":
                        print("Checking for updates...")
                        self._check_for_updates()
                        continue

                    elif user_input.lower() == "feedback":
                        self.handle_feedback_command()
                        continue

                    elif user_input.lower() == "clear":
                        os.system("cls" if os.name == "nt" else "clear")
                        continue

                    elif user_input.lower() == "stats":
                        self.display_stats()
                        continue
                    
                    elif user_input.lower() == "clear_learning":
                        self._clear_learning_data()
                        continue

                    elif user_input.lower() == "stats":
                        self.display_stats()
                        continue

                    elif user_input.lower() == "retrain":
                        print("🔄 Forcing immediate model retraining...")
                        success = self.iterative_learning.force_retraining()
                        if success:
                            print("✅ Model retraining completed successfully!")
                        else:
                            print("❌ Model retraining failed. Check logs for details.")
                        continue

                    elif user_input.lower() == "clear_cache":
                        print("🗑️ Clearing knowledge cache...")
                        self.knowledge_retrieval.clear_cache()
                        print("✅ Knowledge cache cleared!")
                        continue

                    elif user_input.lower() == "learning_stats":
                        self._display_learning_statistics()
                        continue

                    elif user_input.lower() == "error_stats":
                        self._display_error_statistics()
                        continue

                    elif user_input.lower() == "performance_stats":
                        self._display_performance_statistics()
                        continue

                    # Process regular input
                    print("🤖 AI: Processing...", end="", flush=True)

                    result = self.process_user_input(user_input)

                    # Clear processing message
                    print("\r" + " " * 20 + "\r", end="")

                    if result["success"]:
                        # Display response with formatting
                        print(f"🤖 AI: {result['response']}")

                        # Show additional info if requested
                        if result.get("cached"):
                            print("   📋 (from cache)")
                        else:
                            print(
                                f"   📊 Confidence: {result['confidence']:.3f} | "
                                f"Time: {result['response_time']:.3f}s | "
                                f"Type: {result['data_type']}"
                            )

                        print("   💭 Type 'feedback' to rate this response\n")
                    else:
                        print(f"❌ Error: {result['response']}\n")

                except KeyboardInterrupt:
                    print("\n\n🔄 Use 'exit' or 'quit' to properly close the system.")
                    continue

                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break

        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
            print(f"\n❌ Fatal error: {e}")

        finally:
            self._cleanup_and_exit()

    def _cleanup_and_exit(self) -> None:
        """Clean up resources and exit gracefully"""
        try:
            print("\n🔄 Cleaning up...")

            # Save performance metrics
            self.performance_monitor.save_metrics()

            # Clean expired cache entries
            expired_count = self.cache.cleanup_expired()
            if expired_count > 0:
                self.logger.info(f"Cleaned {expired_count} expired cache entries")

            # Save all AI components data
            try:
                self.smart_ai.cleanup()
                print("✅ Smart AI data saved")
            except Exception as e:
                self.logger.error(f"Error saving smart AI data: {e}")
            
            try:
                self.advanced_ai.cleanup()
                print("✅ Advanced AI data saved")
            except Exception as e:
                self.logger.error(f"Error saving advanced AI data: {e}")
            
            try:
                self.enhanced_ai.cleanup()
                print("✅ Enhanced AI data saved")
            except Exception as e:
                self.logger.error(f"Error saving enhanced AI data: {e}")

            # Save self-improving modules data
            try:
                self.self_evaluation.cleanup()
                print("✅ Self-Evaluation data saved")
            except Exception as e:
                self.logger.error(f"Error saving self-evaluation data: {e}")
            
            try:
                self.error_analysis.cleanup()
                print("✅ Error Analysis data saved")
            except Exception as e:
                self.logger.error(f"Error saving error analysis data: {e}")
            
            try:
                self.iterative_learning.cleanup()
                print("✅ Iterative Learning data saved")
            except Exception as e:
                self.logger.error(f"Error saving iterative learning data: {e}")
            
            try:
                self.knowledge_retrieval.cleanup()
                print("✅ Knowledge Retrieval data saved")
            except Exception as e:
                self.logger.error(f"Error saving knowledge retrieval data: {e}")

            # Update session summary in knowledge base
            session_summary = {
                "user_preferences": {
                    "session_duration": time.time() - self.session_start_time,
                    "interaction_count": self.interaction_count,
                    "preferred_model": self.model.model_type if self.model else None,
                }
            }

            JSONManager.update_knowledge_json(self.storage_path, session_summary)

            # Final log
            session_duration = time.time() - self.session_start_time
            self.logger.info(
                f"Session ended. Duration: {session_duration/60:.1f} minutes, "
                f"Interactions: {self.interaction_count}"
            )

            print("✅ Cleanup completed. Goodbye!")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            print(f"⚠️  Cleanup error: {e}")


def main():
    """Main entry point"""
    try:
        # Initialize AI System
        ai_system = AISystemCLI()

        # Run main loop
        ai_system.run()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Failed to start AI System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
