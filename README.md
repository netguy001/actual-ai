# Self-Improving AI System

A sophisticated, self-improving, semi-autonomous learning assistant that can compete with GPT-4/5 level models. This AI system continuously evolves through self-evaluation, error analysis, and iterative learning.

## 🚀 Features

### Core Capabilities
- **Multi-Source Knowledge Retrieval**: Wikipedia, web search, news, weather, time APIs
- **Advanced Query Processing**: Smart AI with auto-correct and context awareness
- **Continuous Learning**: Self-evaluation and iterative model improvement
- **Error Analysis**: Automatic failure detection and correction
- **Memory & Context**: Persistent conversation memory and context awareness
- **Performance Optimization**: Caching, parallel processing, and intelligent routing

### Self-Improvement Features
- **Self-Evaluation Module**: Automatic scoring of responses (accuracy, relevance, speed)
- **Error Analysis**: Detects failed responses and fetches correct answers
- **Iterative Learning**: Periodic retraining using successful answers and corrections
- **Sandbox Evolution**: Spawns child AI versions for testing and improvement
- **Safety & Control**: Logging, user approval, and rollback capabilities

## 🏗️ Architecture

### Module Structure

```
ai_engine/
├── self_evaluation.py      # Self-Evaluation Module
├── error_analysis.py       # Error Analysis & Self-Reflection
├── iterative_learning.py   # Iterative Learning & Model Training
├── knowledge_retrieval.py  # Knowledge Retrieval (Web, Wiki, APIs)
├── reasoning_layer.py      # Reasoning & Source Selection
├── memory_layer.py         # Memory & Context Management
├── meta_learning_layer.py  # Meta-Learning & Strategy Adjustment
├── enhanced_ai.py          # Enhanced AI Core
├── advanced_core.py        # Advanced AI with Multi-Source Search
├── smart_ai.py            # Smart AI with Auto-Correct
├── model.py               # Base AI Model
└── updater.py             # Online Updates
```

### Data Flow

1. **Input Processing**: User query → Auto-correct → Query classification
2. **Knowledge Retrieval**: Multi-source search (Wikipedia, web, APIs)
3. **Reasoning Layer**: Source selection and information synthesis
4. **Response Generation**: Context-aware response with memory
5. **Self-Evaluation**: Automatic scoring and feedback collection
6. **Error Analysis**: Failure detection and correction
7. **Learning**: Data storage for future training
8. **Evolution**: Periodic model retraining and improvement

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the system
python main.py
```

## 📊 Usage

### Basic Usage
```bash
python main.py
```

### Available Commands
- `help` - Show available commands
- `stats` - Display system statistics
- `feedback good/bad <reason>` - Provide manual feedback
- `retrain` - Force immediate model retraining
- `clear` - Clear learning data
- `quit` - Exit the system

### Example Interactions
```
User: What is the current weather in London?
AI: Current weather in London: 18°C, Partly cloudy, Humidity: 65%

User: feedback good accurate and current
AI: ✅ Feedback recorded: GOOD - accurate and current

User: What is machine learning?
AI: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed...
```

## 🔧 Configuration

### System Configuration (`config.json`)
```json
{
  "ai_model": {
    "model_path": "storage/model.pkl",
    "training_data_path": "storage/training_data.json",
    "max_training_samples": 1000
  },
  "evaluation": {
    "min_acceptable_score": 0.6,
    "accuracy_weight": 0.4,
    "relevance_weight": 0.3,
    "speed_weight": 0.3
  },
  "learning": {
    "retraining_interval_hours": 24,
    "min_training_samples": 50,
    "similarity_threshold": 0.7
  }
}
```

### Database Structure
- `storage/evaluation.db` - Response evaluations and performance metrics
- `storage/error_analysis.db` - Error analysis and corrections
- `storage/knowledge_cache.db` - Knowledge retrieval cache
- `storage/iterative_model.pkl` - Trained iterative learning model
- `storage/embeddings.pkl` - Question embeddings for similarity matching

## 🧠 Training & Evolution

### Automatic Training
The system automatically retrains every 24 hours using:
- High-quality responses (score ≥ 0.7)
- Corrected responses from error analysis
- User feedback and ratings

### Manual Training
```bash
# Force immediate retraining
python main.py
> retrain

# Clear learning data and start fresh
python main.py
> clear
```

### Training Data Sources
1. **Successful Responses**: High-scoring responses from all AI components
2. **Error Corrections**: Wrong answers + correct answers from reliable sources
3. **User Feedback**: Manual ratings and feedback
4. **Similarity Matches**: Past successful answers to similar questions

### Model Evolution Process
1. **Data Collection**: Gather training data from interactions
2. **Feature Extraction**: Convert questions to TF-IDF vectors
3. **Model Training**: Train Random Forest classifier
4. **Embedding Update**: Update similarity matching embeddings
5. **Performance Evaluation**: Test on validation set
6. **Model Deployment**: Replace old model if improved

## 🔍 Error Analysis & Self-Reflection

### Failure Detection
The system automatically detects:
- **No Answer**: Empty or error responses
- **Irrelevant**: Off-topic or unrelated answers
- **Incomplete**: Partial or vague responses
- **Outdated**: Responses lacking current data

### Correction Sources
- **Time**: WorldTimeAPI for current time
- **Weather**: wttr.in for weather data
- **News**: BBC RSS for latest news
- **Wikipedia**: For factual information
- **Web Search**: DuckDuckGo for general queries

### Error Statistics
```bash
python main.py
> stats
```

Shows:
- Total errors detected
- Correction success rate
- Error type distribution
- Source effectiveness

## 🎯 Performance Monitoring

### Metrics Tracked
- **Response Quality**: Accuracy, relevance, speed scores
- **Model Performance**: Success rates by model type
- **Learning Progress**: Training data growth and model improvements
- **Error Analysis**: Failure types and correction rates
- **Knowledge Retrieval**: Cache hit rates and source effectiveness

### Performance Optimization
- **Caching**: 24-hour cache for knowledge retrieval
- **Parallel Processing**: Concurrent source queries
- **Intelligent Routing**: Model selection based on query type
- **Memory Management**: Automatic cleanup of old data

## 🔒 Safety & Control

### Safety Features
- **Logging**: All self-modifications are logged
- **User Approval**: Major logic changes require approval
- **Rollback**: Ability to revert to previous versions
- **Validation**: Model performance validation before deployment

### Control Commands
```bash
# View modification logs
python main.py
> logs

# Approve pending changes
python main.py
> approve

# Rollback to previous version
python main.py
> rollback
```

## 🚀 Advanced Features

### Sandbox Evolution
The system can spawn "child" AI versions with modified parameters for testing:
```python
# Spawn child AI with modified logic
child_ai = spawn_child_ai(parent_ai, modifications)
benchmark_result = benchmark_child(child_ai, test_questions)
if benchmark_result.score > parent_score:
    replace_parent_ai(child_ai)
```

### Meta-Learning
The system learns which strategies work best for different query types:
- **Query Classification**: Automatic routing to best model
- **Strategy Adjustment**: Dynamic parameter tuning
- **Performance Tracking**: Continuous strategy evaluation

### Memory & Context
- **Conversation Memory**: Remembers past interactions
- **Context Awareness**: Maintains conversation context
- **Preference Learning**: Learns user preferences
- **Topic Tracking**: Follows conversation topics

## 📈 Monitoring & Analytics

### Real-time Statistics
```bash
python main.py
> stats
```

Displays:
- **Overall Performance**: Total responses, average scores, success rates
- **Model Performance**: Individual model statistics
- **Learning Progress**: Training data size, model improvements
- **Error Analysis**: Error types, correction rates
- **Knowledge Retrieval**: Cache statistics, source usage

### Performance Trends
- **Response Quality**: Trends in accuracy, relevance, speed
- **Model Evolution**: Improvement over time
- **Error Reduction**: Decreasing failure rates
- **Learning Efficiency**: Training data utilization

## 🔧 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version (3.8+ required)
python --version
```

**2. Database Errors**
```bash
# Clear corrupted databases
rm storage/*.db
python main.py  # Will recreate databases
```

**3. Training Failures**
```bash
# Check training data
python main.py
> stats

# Force retraining with minimal data
python main.py
> retrain
```

**4. Performance Issues**
```bash
# Clear cache
python main.py
> clear_cache

# Check system resources
python main.py
> system_info
```

### Debug Mode
```bash
# Enable debug logging
export AI_DEBUG=1
python main.py
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black ai_engine/
flake8 ai_engine/
```

### Adding New Features
1. Create new module in `ai_engine/`
2. Add to `ai_engine/__init__.py`
3. Update `main.py` integration
4. Add tests in `tests/`
5. Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Wikipedia API for factual information
- DuckDuckGo for web search capabilities
- BBC News for current events
- wttr.in for weather data
- WorldTimeAPI for time information

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs: `storage/logs/`
3. Create an issue with detailed error information
4. Include system information and error logs

---

**Note**: This AI system is designed for educational and research purposes. Always verify critical information from authoritative sources.
