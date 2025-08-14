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
            return False

    @handle_errors(default_return=False)
    def _load_or_create_model(self) -> bool:
        """
        Load existing model or create new default model

        Returns:
            True if model loaded/created successfully
        """
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
        Process user input and generate AI response

        Args:
            user_input: User's input text

        Returns:
            Dictionary with response data
        """
        start_time = time.time()

        try:
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

            # Preprocess input
            self.logger.debug(f"Processing input: {user_input[:100]}...")
            processed_data, data_type = preprocess_input(user_input)

            # Check if model type matches data type
            expected_model_type = f"{data_type}_classifier"
            if self.model.model_type != expected_model_type:
                self.logger.info(
                    f"Switching model type from {self.model.model_type} to {expected_model_type}"
                )

                # Try to load appropriate model
                available_models = self.model_manager.list_models()
                suitable_model = next(
                    (
                        m
                        for m in available_models
                        if m.get("model_type") == expected_model_type
                    ),
                    None,
                )

                if suitable_model:
                    self.model = self.model_manager.load_model(suitable_model["name"])
                else:
                    # Create new model of appropriate type
                    self.model = AIModel(model_type=expected_model_type)
                    self._train_default_model()

            # Make prediction
            predictions = self.model.predict(processed_data)

            if not predictions:
                return {
                    "success": False,
                    "error": "No predictions generated",
                    "response": "I couldn't process your input. Please try rephrasing.",
                }

            # Extract prediction and confidence
            prediction = (
                predictions[0]
                if isinstance(predictions[0], (int, float, str))
                else predictions[0][0]
            )
            confidence = predictions[1] if len(predictions) > 1 else 0.5

            # Generate human-readable response
            response_text = self._generate_response(prediction, confidence, data_type)

            # Calculate response time
            response_time = time.time() - start_time

            # Prepare result
            result = {
                "success": True,
                "response": response_text,
                "prediction": prediction,
                "confidence": confidence,
                "data_type": data_type,
                "model_type": self.model.model_type,
                "response_time": response_time,
                "cached": False,
            }

            # Cache result
            self.cache.set(cache_key, result, ttl=300)  # 5 minute cache

            # Log interaction
            self._log_interaction(user_input, response_text, confidence, response_time)

            # Performance monitoring
            self.performance_monitor.log_metric("response_time", response_time)
            self.performance_monitor.log_metric("confidence", confidence)

            return result

        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again.",
            }

    def _generate_response(
        self, prediction: Any, confidence: float, data_type: str
    ) -> str:
        """
        Generate human-readable response from prediction

        Args:
            prediction: Model prediction
            confidence: Prediction confidence
            data_type: Type of input data

        Returns:
            Human-readable response text
        """
        confidence_threshold = self.config.get("model_settings", {}).get(
            "confidence_threshold", 0.5
        )

        if confidence < confidence_threshold:
            return f"I'm not very confident about this (confidence: {confidence:.2f}), but my prediction is: {prediction}"

        if data_type == "text":
            if isinstance(prediction, (int, float)):
                if prediction == 1:
                    return f"This appears to be positive/affirmative (confidence: {confidence:.2f})"
                elif prediction == 0:
                    return f"This appears to be negative/negative (confidence: {confidence:.2f})"
                else:
                    return (
                        f"My prediction is: {prediction} (confidence: {confidence:.2f})"
                    )
            else:
                return f"Classification result: {prediction} (confidence: {confidence:.2f})"

        elif data_type == "numerical":
            return f"Based on the numerical data, my prediction is: {prediction} (confidence: {confidence:.2f})"

        else:
            return f"Prediction: {prediction} (confidence: {confidence:.2f})"

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
║                                                                        ║
║ USAGE:                                                                 ║
║   • Type any text or question to get AI predictions                    ║
║   • The system automatically detects text vs numerical data            ║
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
            print("=" * 70)

        except Exception as e:
            print(f"Error retrieving statistics: {e}")

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
