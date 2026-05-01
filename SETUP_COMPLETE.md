# Question Analyzer - Complete Unified Pipeline

## What Was Created

You now have a **complete, production-ready pipeline** that combines OCR and BERT into one seamless workflow.

### New Files Created

#### Core Modules (app/)
1. **`app/ocr_processor.py`** - OCR text extraction
   - Extracts text from PDFs using PaddleOCR
   - Handles 2 extraction types: Exercise questions & Question papers
   - Clean, reusable class-based API

2. **`app/bert_analyzer.py`** - BERT classification
   - Fine-tunes BERT on your labeled questions
   - Predicts chapter/category for new questions
   - Includes model saving/loading

3. **`app/pipeline.py`** - Unified pipeline
   - Orchestrates OCR → BERT workflow
   - Single function handles everything
   - Auto-exports results to CSV
   - Provides summary statistics

4. **`app/web-app-new.py`** - Web interface
   - Full Streamlit application
   - 4-page interface: Home, Train, Process, Results
   - File upload, progress tracking, CSV export

#### Documentation
1. **`STRUCTURE.md`** - Project structure guide
2. **`PIPELINE_README.md`** - Complete documentation
3. **`MIGRATION_GUIDE.md`** - How to migrate from notebooks

#### Examples & Setup
1. **`scripts/pipeline_examples.py`** - 4 usage examples
2. **`scripts/setup.py`** - Initialization script

---

## The Flow (Simplified)

### Old Way (What you had)
```
extractQuestions.ipynb  →  [manual] →  extracted_exercises-4.csv
                                          ↓
                                    [manual loading]
                                          ↓
                                    Bert_mode.ipynb
                                          ↓
                                    [manual prediction]
                                          ↓
                                    scattered results
```

### New Way (What you have now)
```
ANY PDF FILE
    ↓
pipeline.process_pdf()
    ├─ OCRProcessor.extract_all()
    │  ├─ Exercise Questions (regex pattern matching)
    │  └─ Question Papers (numbered pattern matching)
    │
    └─ BERTQuestionAnalyzer.predict()
       └─ Classify each question into chapters
           ↓
        AUTOMATIC CSV EXPORT
```

---

## Core Concepts

### 1. **OCRProcessor** - Text Extraction
```python
processor = OCRProcessor()
results = processor.extract_all('file.pdf')
# Returns: exercises[], papers[], raw_text
```

**Extracts two types:**
- **Exercise questions**: Pattern "Exercise 2.1: question text..."
- **Question papers**: Pattern "Q1. question..." or "1. question..."

### 2. **BERTQuestionAnalyzer** - Classification
```python
analyzer = BERTQuestionAnalyzer()
analyzer.train(questions, labels)  # Train on labeled data
predictions = analyzer.predict(['New question'])
# Returns: predicted_chapter, confidence
```

**Key features:**
- Train once on your labeled data
- Predict on any new questions
- Save/load trained models

### 3. **QuestionAnalysisPipeline** - Orchestration
```python
pipeline = QuestionAnalysisPipeline()
pipeline.train_bert_model('training.csv')
results = pipeline.process_pdf('file.pdf')
pipeline.save_results_to_csv(results)
summary = pipeline.get_summary(results)
```

**Brings it all together:**
- One function for entire workflow
- Automatic error handling
- Progress tracking
- Result export

---

## Quick Start (3 Steps)

### Step 1: Train Model (5 min)
```python
from app.pipeline import QuestionAnalysisPipeline

pipeline = QuestionAnalysisPipeline()
pipeline.train_bert_model('data/processed/final_questions.csv')
```

**What happens:**
- Loads your labeled questions
- Creates chapter mappings
- Trains BERT classifier (13 epochs)
- Saves model to `data/models/bert_model/`

### Step 2: Process PDF (2 min)
```python
results = pipeline.process_pdf(
    'data/raw/Primer.pdf',
    extraction_type='both',  # or 'exercise' or 'paper'
    page_start=0,
    page_end=50
)
```

**What happens:**
- Extracts all text from PDF pages
- Identifies exercise questions & question papers
- Classifies each question with BERT
- Returns results with predictions

### Step 3: Save Results (auto)
```python
pipeline.save_results_to_csv(results)
```

**Output:**
- `data/processed/analyzed_exercises_YYYYMMDD_HHMMSS.csv`
- `data/processed/analyzed_papers_YYYYMMDD_HHMMSS.csv`

---

## Result Format

Each output CSV contains:

```
For Exercises:
exercise_id | question_text | type | predicted_chapter | confidence
Exercise 2.1 | What is... | exercise | Pointers and... | 85.32%

For Papers:
question_id | question_text | type | predicted_chapter | confidence  
Q1 | Explain... | paper | Inheritance | 92.15%
```

---

## Usage Examples

### Example 1: Web Interface (Easiest)
```bash
streamlit run app/web-app-new.py
# Then:
# 1. Go to "Train Model" page
# 2. Upload training CSV and click "Start Training"
# 3. Go to "Process PDF" page
# 4. Upload a PDF and click "Process PDF"
# 5. Go to "Results" to see and download results
```

### Example 2: Python Script (Most Control)
```python
from app.pipeline import QuestionAnalysisPipeline

pipeline = QuestionAnalysisPipeline()

# Train
pipeline.train_bert_model('data/processed/final_questions.csv')

# Process multiple PDFs
for pdf_file in ['file1.pdf', 'file2.pdf']:
    results = pipeline.process_pdf(f'data/raw/{pdf_file}')
    pipeline.save_results_to_csv(results)
    
    summary = pipeline.get_summary(results)
    print(f"{pdf_file}: {summary['total_questions']} questions found")
```

### Example 3: Just Extract (No BERT)
```python
from app.ocr_processor import OCRProcessor

processor = OCRProcessor()
results = processor.extract_all('file.pdf')

exercises = results['exercises']    # Just extract exercises
papers = results['papers']          # Just extract papers
raw = results['raw_text']           # Raw OCR output
```

### Example 4: Just Classify (No OCR)
```python
from app.bert_analyzer import BERTQuestionAnalyzer

analyzer = BERTQuestionAnalyzer()
analyzer.load_training_data('training.csv')
analyzer.train(questions, labels)

# Classify some questions
results = analyzer.predict([
    "What is inheritance?",
    "Explain templates",
    "How do pointers work?"
])

for r in results:
    print(f"Q: {r['question']}")
    print(f"→ {r['predicted_chapter']} ({r['confidence']})")
```

---

## Directory Structure After Setup

```
QuestionAnalyzer/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ocr_processor.py         ← NEW: OCR extraction
│   ├── bert_analyzer.py         ← NEW: BERT classification
│   ├── pipeline.py              ← NEW: Unified pipeline
│   └── web-app-new.py           ← NEW: Updated web app
│
├── data/
│   ├── raw/                     ← Input PDFs go here
│   │   └── Primer.pdf
│   ├── processed/               ← Output CSVs saved here
│   │   ├── final_questions.csv  (training data)
│   │   └── analyzed_*.csv       (auto-generated results)
│   └── models/
│       └── bert_model/          ← Trained BERT model
│           ├── config.json
│           ├── pytorch_model.bin
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           └── label_map.json
│
├── notebooks/                   ← Old notebooks (keep for reference)
├── scripts/
│   ├── pipeline_examples.py     ← NEW: Code examples
│   ├── setup.py                 ← NEW: Setup script
│   └── __init__.py
│
├── STRUCTURE.md                 ← Project structure
├── PIPELINE_README.md           ← NEW: Full documentation
├── MIGRATION_GUIDE.md           ← NEW: Migration guide
└── requirements.txt
```

---

## Key Improvements Over Old System

| Feature | Old | New |
|---------|-----|-----|
| **Notebook Notebooks Needed** | 2 separate | 1 unified interface |
| **Manual Steps** | Many | None (fully automatic) |
| **Exercise Questions** | ✅ | ✅ |
| **Question Papers** | ❌ | ✅ |
| **Error Handling** | ❌ | ✅ |
| **Progress Tracking** | ❌ | ✅ |
| **Web Interface** | ⚠️ Basic | ✅ Full featured |
| **Batch Processing** | Manual | ✅ Easy |
| **Auto CSV Export** | Manual | ✅ Automatic |
| **Statistics/Summary** | Manual | ✅ Auto-generated |

---

## Common Tasks

### Train model with custom parameters
```python
pipeline.train_bert_model(
    'data/processed/final_questions.csv',
    batch_size=16,          # Default: 8
    num_epochs=20,          # Default: 13
    learning_rate=3e-5      # Default: 4e-5
)
```

### Process only exercise questions
```python
results = pipeline.process_pdf('file.pdf', extraction_type='exercise')
```

### Process specific page range
```python
results = pipeline.process_pdf('file.pdf', page_start=10, page_end=100)
```

### Get summary statistics
```python
summary = pipeline.get_summary(results)
print(f"Total: {summary['total_questions']}")
print(f"Distribution: {summary['chapters_in_exercises']}")
```

### Save trained model for later
```python
# After training:
pipeline.bert_analyzer.save_model('my_model_path/')

# Later, load and use:
pipeline.bert_analyzer.load_model('my_model_path/')
```

---

## Troubleshooting

**Q: How do I know if it worked?**
- Check for CSV files in `data/processed/analyzed_*.csv`
- Each CSV should have your extracted questions with chapter predictions

**Q: OCR is extracting gibberish**
- Increase DPI in `ocr_processor.py` line with `fitz.Matrix(1.5, 1.5)`
- Try `(2.0, 2.0)` for higher quality

**Q: BERT predictions are bad**
- Train with more labeled data
- Increase num_epochs to 20-30
- Use a larger model: `BertModel(model_name='bert-large-uncased')`

**Q: Out of memory**
- Reduce batch_size to 4 or 2
- Process fewer pages at once

**Q: Missing paddleocr error**
- Run: `pip install paddleocr`
- For GPU: `pip install paddlepaddle-gpu`

---

## Next Steps

1. ✅ **Organize files** - Done! New structure is clean
2. ✅ **Combine OCR & BERT** - Done! Unified pipeline created
3. ✅ **Simplify flow** - Done! One function does everything
4. ✅ **Support both question types** - Done! Handles exercises & papers
5. ⬜ **Deploy web app** - Optional: `streamlit run app/web-app-new.py`
6. ⬜ **Build REST API** - Optional: Add Flask endpoints
7. ⬜ **Improve accuracy** - Optional: Train with more data

---

## Files to Read

1. **PIPELINE_README.md** - Full API documentation
2. **MIGRATION_GUIDE.md** - Detailed migration from notebooks
3. **scripts/pipeline_examples.py** - 4 runnable examples
4. **app/ocr_processor.py** - OCR implementation
5. **app/bert_analyzer.py** - BERT implementation
6. **app/pipeline.py** - Pipeline orchestration

---

## Support

- **Examples**: See `scripts/pipeline_examples.py`
- **Documentation**: See `PIPELINE_README.md`
- **Web Interface**: Run `streamlit run app/web-app-new.py`
- **Old Notebooks**: Still available in `notebooks/` for reference

---

## Summary

You now have:
- ✅ **Organized file structure** - Everything in its place
- ✅ **Unified pipeline** - One function for OCR → BERT → Export
- ✅ **Two extraction types** - Exercises AND question papers
- ✅ **Web interface** - No coding needed for end users
- ✅ **Clean API** - Easy to extend and customize
- ✅ **Full documentation** - Examples and guides included

**Ready to use!** 🚀

Start with: `streamlit run app/web-app-new.py`
