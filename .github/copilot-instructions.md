# AI Agent Instructions for Tweet Sentiment Classifier

## Project Overview
This is a Tweet sentiment classification application that uses a fine-tuned RoBERTa model to classify tweets into Positive, Neutral, or Negative sentiment categories. The project demonstrates efficient integration of Hugging Face Transformers with a Gradio web interface.

## Key Components

### Model Architecture
- Location: `sentiment_model/` contains the fine-tuned RoBERTa model files
- Model type: RoBERTa-based sequence classification (3 classes)
- Performance: Achieves 0.82 Macro F1 Score
- Key files:
  - `model.safetensors`: Contains model weights
  - `config.json`: Model configuration
  - `tokenizer.json` + related files: RoBERTa tokenizer configuration

### Application Structure
- `app.py`: Main entry point containing both model inference and UI code
- Pattern: Single-file architecture for simplicity
- Key dependencies in `requirements.txt`:
  - transformers: Model loading and inference
  - gradio: Web UI framework
  - torch: Deep learning backend
  - huggingface-hub: Model loading utilities

## Development Workflows

### Environment Setup
```bash
python -m venv venv  # Create virtual environment
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py  # Launches Gradio interface on localhost
```

### Model Inference Pattern
```python
# Example from app.py
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
```

## Project-Specific Conventions

### Label Mapping
- Labels are mapped as: {0: "Negative", 1: "Neutral", 2: "Positive"}
- Access via `id2label` dictionary in `app.py`

### Model Loading
- Always load from local `sentiment_model/` directory
- Use `AutoTokenizer` and `AutoModelForSequenceClassification` for compatibility

### Input Processing
- Tweets are automatically truncated and padded
- No preprocessing required - model handles raw text input

## Integration Points
- Model ↔ UI: Through the `classify_tweet()` function in `app.py`
- External: None - fully self-contained application