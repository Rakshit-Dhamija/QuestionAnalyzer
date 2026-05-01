# 📖 Documentation Index

## Start Here

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** ⭐ START HERE
   - Visual overview
   - Common tasks
   - 3-minute workflow
   - Troubleshooting

2. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)**
   - What was created
   - Complete flow explanation
   - Examples for all use cases

## Detailed Documentation

3. **[PIPELINE_README.md](PIPELINE_README.md)**
   - Full API reference
   - Installation instructions
   - Module documentation
   - Performance tips

4. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**
   - How old system worked
   - How new system works
   - File-by-file mapping
   - Migration steps

5. **[STRUCTURE.md](STRUCTURE.md)**
   - Project folder organization
   - What each folder contains
   - Next steps for development

## Code Examples

6. **[scripts/pipeline_examples.py](scripts/pipeline_examples.py)**
   - Example 1: Train model
   - Example 2: Process PDF
   - Example 3: Batch processing
   - Example 4: Load and predict

7. **[scripts/setup.py](scripts/setup.py)**
   - Initialization script
   - Dependency checking
   - Directory creation

## Quick Navigation

### I want to...

**Get started immediately**
→ Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
→ Run: `streamlit run app/web-app-new.py`

**Understand what was created**
→ Read: [SETUP_COMPLETE.md](SETUP_COMPLETE.md)

**See code examples**
→ Read: [scripts/pipeline_examples.py](scripts/pipeline_examples.py)

**Understand the new architecture**
→ Read: [PIPELINE_README.md](PIPELINE_README.md)

**Learn about migration from notebooks**
→ Read: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

**Understand the project structure**
→ Read: [STRUCTURE.md](STRUCTURE.md)

## File Organization

### Core Application
```
app/
├── ocr_processor.py         OCR text extraction
├── bert_analyzer.py         BERT classification  
├── pipeline.py              Unified interface
├── web-app-new.py          Web UI
└── config.py               Configuration
```

### Data
```
data/
├── raw/                    Input PDFs
├── processed/              Output CSVs
└── models/                 Trained models
```

### Scripts & Examples
```
scripts/
├── pipeline_examples.py    Code examples (4 examples)
├── setup.py               Initialization
└── __init__.py
```

### Notebooks (Legacy)
```
notebooks/
├── extractQuestions.ipynb  (now: ocr_processor.py)
├── Bert_mode.ipynb        (now: bert_analyzer.py)
└── ... others
```

## Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| QUICK_REFERENCE.md | Quick overview | 5 min |
| SETUP_COMPLETE.md | What was built | 10 min |
| PIPELINE_README.md | Full API docs | 15 min |
| MIGRATION_GUIDE.md | How to migrate | 10 min |
| STRUCTURE.md | Project structure | 5 min |

## The Three Core Modules

### 1. OCRProcessor (app/ocr_processor.py)
**Purpose**: Extract text from PDFs

**Key Methods**:
- `extract_from_pdf()` - Get raw text
- `extract_exercise_questions()` - Parse exercises
- `extract_question_papers()` - Parse question papers
- `extract_all()` - Do both

**Use when**: You need to extract text from a PDF

### 2. BERTQuestionAnalyzer (app/bert_analyzer.py)
**Purpose**: Classify questions by chapter

**Key Methods**:
- `load_training_data()` - Load labeled questions
- `train()` - Train the classifier
- `predict()` - Classify new questions
- `save_model()` / `load_model()` - Save/load trained model

**Use when**: You need to classify questions into categories

### 3. QuestionAnalysisPipeline (app/pipeline.py)
**Purpose**: Unified end-to-end workflow

**Key Methods**:
- `train_bert_model()` - Train classifier
- `process_pdf()` - Extract AND classify
- `save_results_to_csv()` - Export results
- `get_summary()` - Get statistics

**Use when**: You want everything in one place

## Common Workflows

### Workflow 1: Web Interface (Easiest)
```bash
streamlit run app/web-app-new.py
```
→ Use browser interface
→ No coding needed

### Workflow 2: Python Script (Recommended)
```python
from app.pipeline import QuestionAnalysisPipeline
p = QuestionAnalysisPipeline()
p.train_bert_model('data/processed/final_questions.csv')
results = p.process_pdf('data/raw/file.pdf')
p.save_results_to_csv(results)
```

### Workflow 3: Low-Level API (Advanced)
```python
from app.ocr_processor import OCRProcessor
from app.bert_analyzer import BERTQuestionAnalyzer

# Extract manually
processor = OCRProcessor()
text = processor.extract_from_pdf('file.pdf')
exercises = processor.extract_exercise_questions(text)

# Classify manually
analyzer = BERTQuestionAnalyzer()
analyzer.train(questions, labels)
predictions = analyzer.predict([q['question_text'] for q in exercises])
```

## What Each Module Replaces

| Old | New |
|-----|-----|
| extractQuestions.ipynb | ocr_processor.py |
| Bert_mode.ipynb | bert_analyzer.py |
| (nothing) | pipeline.py ← NEW unified interface |
| web-app.py | web-app-new.py ← Updated |

## Key Improvements

✨ **Before**: 2 separate notebooks + manual steps
✨ **After**: 1 unified pipeline + automatic

✨ **Before**: Only exercise questions
✨ **After**: Exercise questions AND question papers

✨ **Before**: Manual error handling
✨ **After**: Built-in error handling + progress tracking

## Support Resources

1. **Quick answers**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **How to use**: See [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
3. **Code examples**: See [scripts/pipeline_examples.py](scripts/pipeline_examples.py)
4. **Full documentation**: See [PIPELINE_README.md](PIPELINE_README.md)
5. **Migrating from notebooks**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## Next Steps

1. ✅ Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. ✅ Run `streamlit run app/web-app-new.py`
3. ✅ Train the model (via web interface)
4. ✅ Process your first PDF
5. ✅ Check results in `data/processed/`

## Questions?

- **How do I use it?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **What was created?** → [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
- **API documentation?** → [PIPELINE_README.md](PIPELINE_README.md)
- **Code examples?** → [scripts/pipeline_examples.py](scripts/pipeline_examples.py)

---

**You're all set!** 🚀

Everything is organized, documented, and ready to use.

Start with: `streamlit run app/web-app-new.py`
