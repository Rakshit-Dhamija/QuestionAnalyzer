# 🎯 Quick Reference Guide

## What You Have Now

```
📚 UNIFIED QUESTION ANALYZER PIPELINE
├─ 🔍 OCR Extraction (PaddleOCR)
│  ├─ Exercise Questions (Exercise X.Y pattern)
│  └─ Question Papers (Q1, Q2 pattern)
├─ 🧠 BERT Classification
│  ├─ Train on labeled data
│  └─ Predict chapter for new questions
└─ 🌐 Web Interface (Streamlit)
   ├─ Train page
   ├─ Process page
   └─ Results page
```

## Two Ways to Use It

### Option 1: Web Interface (Easiest)
```bash
streamlit run app/web-app-new.py
```
Then click through the pages in your browser.

### Option 2: Python Script (Most Control)
```python
from app.pipeline import QuestionAnalysisPipeline

pipeline = QuestionAnalysisPipeline()
pipeline.train_bert_model('data/processed/final_questions.csv')
results = pipeline.process_pdf('data/raw/file.pdf')
pipeline.save_results_to_csv(results)
```

## The 3-Minute Workflow

1. **TRAIN** (1 min)
   ```python
   pipeline.train_bert_model('training_data.csv')
   ```

2. **PROCESS** (1 min)
   ```python
   results = pipeline.process_pdf('your_file.pdf')
   ```

3. **EXPORT** (instant)
   ```python
   pipeline.save_results_to_csv(results)
   ```

Done! Results in `data/processed/analyzed_*.csv`

## Core Methods

```
pipeline.train_bert_model()      Train BERT on labeled questions
pipeline.process_pdf()            Extract and classify questions
pipeline.save_results_to_csv()   Export results to CSV
pipeline.get_summary()            Get statistics
```

## Input Files Needed

1. **Training CSV**: `data/processed/final_questions.csv`
   ```
   Question Text,Chapter_name
   "What is a pointer?","Pointers and Dynamic Memory"
   "Explain templates","Templates"
   ```

2. **PDF to Process**: `data/raw/your_file.pdf`

## Output Files Generated

```
data/processed/
├─ analyzed_exercises_20260501_120000.csv
└─ analyzed_papers_20260501_120000.csv
```

Each CSV contains:
- `question_id` or `exercise_id`
- `question_text`
- `type` (exercise or paper)
- `predicted_chapter`
- `confidence`

## New Features vs Old

| Feature | Before | Now |
|---------|--------|-----|
| Steps required | 2 notebooks + manual | 1 unified pipeline |
| Extract exercises | ✅ | ✅ |
| Extract papers | ❌ | ✅ NEW |
| Web interface | ⚠️ | ✅ Full |
| Error handling | ❌ | ✅ |
| Auto-export | ❌ | ✅ |

## File Locations

```
IMPORTANT FILES:
app/pipeline.py              ← Main interface (start here!)
app/ocr_processor.py         ← OCR logic
app/bert_analyzer.py         ← BERT logic
app/web-app-new.py           ← Web interface

DOCUMENTATION:
SETUP_COMPLETE.md            ← Read this first!
PIPELINE_README.md           ← Full documentation
MIGRATION_GUIDE.md           ← Migrating from notebooks
scripts/pipeline_examples.py  ← Code examples

DATA FOLDERS:
data/raw/                    ← Put PDFs here
data/processed/              ← Results saved here
data/models/bert_model/      ← Trained model saved here
```

## Common Tasks

### Train on custom data
```python
pipeline.train_bert_model('my_training.csv')
```

### Extract only exercise questions
```python
results = pipeline.process_pdf('file.pdf', extraction_type='exercise')
```

### Extract only question papers
```python
results = pipeline.process_pdf('file.pdf', extraction_type='paper')
```

### Process specific pages
```python
results = pipeline.process_pdf('file.pdf', page_start=10, page_end=100)
```

### Get summary stats
```python
summary = pipeline.get_summary(results)
print(summary['chapters_in_exercises'])
```

### Process multiple PDFs
```python
for pdf_file in ['file1.pdf', 'file2.pdf']:
    r = pipeline.process_pdf(f'data/raw/{pdf_file}')
    pipeline.save_results_to_csv(r)
```

## Architecture

```
PDF INPUT
   ↓
OCRProcessor
   ├─ Extract exercise questions (regex pattern)
   ├─ Extract question papers (regex pattern)
   └─ Get raw text
   ↓
BERTQuestionAnalyzer
   ├─ Tokenize questions
   ├─ Pass through BERT model
   ├─ Get chapter predictions
   └─ Return with confidence
   ↓
pipeline.py
   ├─ Combine OCR + BERT results
   ├─ Auto-format to CSV
   └─ Save with timestamp
   ↓
CSV OUTPUT
```

## Performance

- **Training**: 2-10 minutes (depends on GPU/CPU)
- **OCR extraction**: 1 page per 5-10 seconds
- **BERT prediction**: 5-10 questions per second (GPU)
- **Total for 50 pages**: ~5-10 minutes

## Requirements

```
pandas              Data manipulation
torch               PyTorch framework
transformers        BERT models
paddleocr           PDF text extraction
streamlit           Web interface
pymupdf             PDF processing
Pillow              Image processing
```

Install all: `pip install -r requirements.txt`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bad OCR quality | Increase DPI: change `(1.5, 1.5)` to `(2.0, 2.0)` |
| Wrong predictions | Train with more data, increase epochs |
| Out of memory | Reduce batch_size, process fewer pages |
| Module not found | Run `pip install -r requirements.txt` |
| Model not loading | Retrain: `pipeline.train_bert_model(...)` |

## Getting Help

1. Read: `SETUP_COMPLETE.md`
2. Examples: `scripts/pipeline_examples.py`
3. Docs: `PIPELINE_README.md`
4. Migration: `MIGRATION_GUIDE.md`

## Key Takeaways

✨ **Everything automated** - Just call one function
✨ **Two extraction types** - Exercises AND question papers
✨ **Production ready** - Error handling, progress tracking
✨ **Web interface** - No coding needed
✨ **Fully documented** - Guides and examples included

## Start Here

```python
from app.pipeline import QuestionAnalysisPipeline

# Initialize
pipeline = QuestionAnalysisPipeline()

# Train (do this once)
pipeline.train_bert_model('data/processed/final_questions.csv')

# Process any PDF
results = pipeline.process_pdf('data/raw/Primer.pdf')

# Save results
pipeline.save_results_to_csv(results)

# That's it! ✅
```

Or for web interface:
```bash
streamlit run app/web-app-new.py
```

---

**You're ready to go!** 🚀

All questions extraction → analysis → export is now **one simple workflow**.
