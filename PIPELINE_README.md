# Question Analyzer - Unified Pipeline

A comprehensive system for extracting questions from PDFs and analyzing them using OCR and BERT classification.

## Overview

This project combines:
- **PaddleOCR**: Text extraction from PDF images
- **BERT**: Question classification and analysis
- **Streamlit**: Web-based user interface

### Supported Question Types

1. **Exercise Questions**: Pattern-based extraction (e.g., "Exercise 2.1: Question text...")
2. **Question Papers**: Numbered questions (e.g., "Q1.", "Q2.", "1.", "2.")

## Project Structure

```
QuestionAnalyzer/
├── app/                          # Main application modules
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── web-app-new.py            # Streamlit web interface (NEW)
│   ├── ocr_processor.py          # OCR extraction module (NEW)
│   ├── bert_analyzer.py          # BERT classification module (NEW)
│   └── pipeline.py               # Unified pipeline (NEW)
│
├── data/
│   ├── raw/                      # Input PDFs
│   ├── processed/                # Extracted CSVs and results
│   └── models/                   # Trained BERT models
│
├── notebooks/                    # Jupyter notebooks (legacy)
├── scripts/
│   ├── pipeline_examples.py      # Usage examples (NEW)
│   └── __init__.py
│
├── static/                       # Frontend assets (CSS, JS)
├── templates/                    # HTML templates
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

## Installation

### 1. Clone the repository

```bash
cd QuestionAnalyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` - Data manipulation
- `torch` - PyTorch
- `transformers` - BERT models
- `paddleocr` - Text extraction from PDFs
- `streamlit` - Web interface
- `pymupdf` - PDF processing
- `Pillow` - Image processing

### 3. Install optional GPU support (recommended for training)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle-gpu
```

## Quick Start

### Using the Web Interface

```bash
streamlit run app/web-app-new.py
```

Then:
1. Go to **📖 Train Model** - Train BERT on your labeled data
2. Go to **📄 Process PDF** - Upload and process a PDF
3. Go to **📊 Results** - View and export results

### Using Python API

```python
from app.pipeline import QuestionAnalysisPipeline

# Initialize pipeline
pipeline = QuestionAnalysisPipeline()

# Train model on labeled data
pipeline.train_bert_model('data/processed/final_questions.csv')

# Process a PDF
results = pipeline.process_pdf(
    'data/raw/document.pdf',
    extraction_type='both',  # 'exercise', 'paper', or 'both'
    page_start=0,
    page_end=50
)

# Save results
pipeline.save_results_to_csv(results)

# Get summary
summary = pipeline.get_summary(results)
print(f"Found {summary['total_questions']} questions")
```

## Usage Examples

See [scripts/pipeline_examples.py](scripts/pipeline_examples.py) for detailed examples:

- **Example 1**: Train BERT model
- **Example 2**: Process single PDF
- **Example 3**: Batch process multiple PDFs
- **Example 4**: Load pre-trained model and predict

Run examples:
```bash
python -m scripts.pipeline_examples
```

## API Reference

### QuestionAnalysisPipeline

Main class for end-to-end processing.

#### Methods

```python
train_bert_model(training_csv_path, **kwargs)
```
Train BERT classifier on labeled data.
- `training_csv_path`: Path to CSV with 'Question Text' and 'Chapter_name' columns
- `batch_size`: Training batch size (default: 8)
- `num_epochs`: Number of training epochs (default: 13)
- `learning_rate`: Learning rate (default: 4e-5)

```python
process_pdf(pdf_path, extraction_type='both', page_start=0, page_end=None)
```
Process a PDF file end-to-end.
- `pdf_path`: Path to PDF file
- `extraction_type`: 'exercise', 'paper', or 'both'
- `page_start`: Starting page (0-indexed)
- `page_end`: Ending page (None for all)

Returns: Dictionary with extracted and analyzed questions

```python
save_results_to_csv(results, output_dir='data/processed')
```
Save analysis results to CSV files.

```python
get_summary(results)
```
Get summary statistics from results.

### OCRProcessor

Handles PDF text extraction.

```python
extract_from_pdf(pdf_path, page_start=0, page_end=None)
```
Extract raw text from PDF pages.

```python
extract_exercise_questions(extracted_text)
```
Extract exercise questions using pattern matching.

```python
extract_question_papers(extracted_text)
```
Extract question paper questions.

### BERTQuestionAnalyzer

BERT-based question classification.

```python
load_training_data(csv_path)
```
Load training data and create label mappings.

```python
train(questions, labels_encoded, batch_size=8, num_epochs=13, learning_rate=4e-5, val_split=0.2)
```
Train the classifier.

```python
predict(questions)
```
Predict chapter labels for questions. Returns list of dictionaries with:
- `question`: Input question text
- `predicted_chapter`: Predicted chapter label
- `confidence`: Confidence percentage

## Output Format

### Analyzed Questions CSV

When you process a PDF and save results, you get CSV files with:

```
exercise_id          | question_text                      | type     | predicted_chapter        | confidence
Exercise 2.1         | What is a pointer?                 | exercise | Pointers and Dynamic...  | 85.32%
Q1                   | Explain inheritance               | paper    | Inheritance              | 92.15%
```

## Configuration

Edit `app/config.py` to modify:

```python
class Config:
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
```

## Performance Tips

1. **GPU Acceleration**: Install CUDA support for 10x faster processing
2. **Batch Processing**: Process multiple PDFs to amortize model loading
3. **Shorter Sequences**: Adjust `max_length` in BERT for faster inference
4. **Pre-trained Models**: Load pre-trained models to skip training

## Training Dataset Requirements

Your CSV should have:
- `Question Text` column: The actual question
- `Chapter_name` column: The chapter/category label

Example:
```
Question Text,Chapter_name
"What is a pointer?","Pointers and Dynamic Memory"
"Explain virtual functions","Virtual Functions"
```

## Common Issues

### OCR quality is poor
- Increase DPI: Change `fitz.Matrix(1.5, 1.5)` to `(2.0, 2.0)`
- Try different page ranges

### BERT predictions are inaccurate
- Train with more data
- Increase `num_epochs`
- Adjust `learning_rate`
- Use higher `max_length` for long questions

### Out of memory
- Reduce `batch_size`
- Process fewer pages at once
- Use CPU instead of GPU

## Advanced Usage

### Custom extraction patterns

```python
from app.ocr_processor import OCRProcessor

processor = OCRProcessor()
text = processor.extract_from_pdf('file.pdf')

# Define custom pattern
custom_pattern = r"^Question (\d+):\s*(.*)"

# Process with custom pattern
# (Extend OCRProcessor.extract_exercise_questions for custom logic)
```

### Transfer learning

```python
analyzer = BERTQuestionAnalyzer(model_name='bert-large-uncased')
# Use larger model for better accuracy with more training time
```

## File Locations

- **Training Data**: `data/processed/final_questions.csv`
- **Input PDFs**: `data/raw/` (e.g., `Primer.pdf`)
- **Processed Results**: `data/processed/analyzed_*.csv`
- **Trained Models**: `data/models/bert_model/`
- **Notebooks (legacy)**: `notebooks/`

## Dependencies

See [requirements.txt](../requirements.txt) for full list.

Key packages:
- `pandas>=1.3.0`
- `torch>=1.9.0`
- `transformers>=4.0.0`
- `paddleocr>=2.4.0`
- `streamlit>=1.0.0`
- `pymupdf>=1.16.0`

## License

[Your License Here]

## Support

For issues or questions, refer to:
- [scripts/pipeline_examples.py](scripts/pipeline_examples.py) - Usage examples
- [STRUCTURE.md](STRUCTURE.md) - Project structure
- Jupyter notebooks in `notebooks/` - Original analysis

## Next Steps

1. ✅ Organize file structure
2. ✅ Create unified pipeline
3. ✅ Build web interface
4. ⬜ Add REST API
5. ⬜ Deploy to cloud
6. ⬜ Add more extraction types
7. ⬜ Improve BERT model accuracy
