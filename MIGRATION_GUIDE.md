"""
MIGRATION GUIDE
From separate notebooks to unified pipeline
"""

# ============================================================
# OLD STRUCTURE → NEW STRUCTURE
# ============================================================

Old Flow:
```
extractQuestions.ipynb (OCR)
        ↓ (manual output)
extracted_exercises-4.csv
        ↓ (manual loading)
Bert_mode.ipynb (BERT training)
        ↓ (manual prediction)
results (scattered files)
```

New Flow:
```
PDFs (input)
  ↓
Pipeline.process_pdf()
  ├─ OCRProcessor.extract_all()
  │  ├─ extract_exercise_questions()
  │  └─ extract_question_papers()
  └─ BERTQuestionAnalyzer.predict()
    └─ CSV exports (automatic)
```

# ============================================================
# FILE MAPPING
# ============================================================

extractQuestions.ipynb  →  app/ocr_processor.py (OCRProcessor class)
Bert_mode.ipynb         →  app/bert_analyzer.py (BERTQuestionAnalyzer class)
web-app.py              →  app/web-app-new.py (Updated Streamlit app)

# ============================================================
# USAGE COMPARISON
# ============================================================

## OLD WAY (Notebooks)

1. Open extractQuestions.ipynb
   - Set pdf_path = "Primer.pdf"
   - Set page range
   - Run all cells
   - Manually save extracted_exercises-4.csv

2. Open Bert_mode.ipynb
   - Load final_questions.csv
   - Train model (13 epochs)
   - Manually predict on new questions

3. Process results manually (copy-paste, Excel, etc.)

## NEW WAY (Pipeline)

```python
from app.pipeline import QuestionAnalysisPipeline

# ONE line of setup
pipeline = QuestionAnalysisPipeline()

# Train once
pipeline.train_bert_model('data/processed/final_questions.csv')

# Process any PDF
results = pipeline.process_pdf('any_file.pdf')

# Save automatically
pipeline.save_results_to_csv(results)
```

# ============================================================
# FEATURE COMPARISON
# ============================================================

Feature                 | Old (Notebooks) | New (Pipeline)
---                    | ---             | ---
Extract Exercise Qs    | ✅              | ✅
Extract Paper Qs       | ❌              | ✅
Train BERT             | ✅              | ✅
Batch predictions      | ❌ (manual)     | ✅ (automatic)
Combine results        | ❌ (manual)     | ✅ (automatic)
Web interface          | ⚠️ (basic)      | ✅ (full featured)
Save to CSV            | ✅ (manual)     | ✅ (automatic)
Error handling         | ❌              | ✅
Progress tracking      | ❌              | ✅
Summary stats          | ❌              | ✅

# ============================================================
# STEP-BY-STEP MIGRATION
# ============================================================

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place your training data
```
data/processed/final_questions.csv  (required for training)
data/raw/Primer.pdf                 (or any PDF)
```

### Step 3: Train the model (one-time)
```python
from app.pipeline import QuestionAnalysisPipeline

pipeline = QuestionAnalysisPipeline()
pipeline.train_bert_model('data/processed/final_questions.csv')

# Model saved to data/models/bert_model/
```

### Step 4: Process PDFs
```python
# Now you can process any PDF
results = pipeline.process_pdf('data/raw/any_file.pdf')
pipeline.save_results_to_csv(results)
```

### Step 5: Or use the web interface
```bash
streamlit run app/web-app-new.py
```

# ============================================================
# WHAT EACH MODULE DOES
# ============================================================

## app/ocr_processor.py
Replaces: extractQuestions.ipynb

Classes:
- OCRProcessor: Main OCR handler

Methods:
- extract_from_pdf(): Get raw text from PDF
- extract_exercise_questions(): Parse exercise pattern
- extract_question_papers(): Parse question paper pattern
- extract_all(): Do both extractions at once

Usage:
```python
processor = OCRProcessor()
results = processor.extract_all('file.pdf', page_start=0, page_end=50)
exercises = results['exercises']  # Exercise questions
papers = results['papers']        # Paper questions
```

## app/bert_analyzer.py
Replaces: Bert_mode.ipynb

Classes:
- QuestionDataset: PyTorch dataset
- BERTClassifier: Neural network model
- BERTQuestionAnalyzer: High-level interface

Methods:
- load_training_data(): Load CSV
- train(): Train model
- predict(): Classify questions
- save_model(): Save trained model
- load_model(): Load trained model

Usage:
```python
analyzer = BERTQuestionAnalyzer()
questions, labels = analyzer.load_training_data('train.csv')
analyzer.train(questions, labels)
predictions = analyzer.predict(['New question'])
```

## app/pipeline.py
NEW unified interface

Classes:
- QuestionAnalysisPipeline: Main orchestrator

Methods:
- train_bert_model(): Train OCR + BERT pipeline
- process_pdf(): End-to-end processing
- save_results_to_csv(): Export results
- get_summary(): Statistics

Usage:
```python
pipeline = QuestionAnalysisPipeline()
pipeline.train_bert_model('train.csv')
results = pipeline.process_pdf('file.pdf')
summary = pipeline.get_summary(results)
```

# ============================================================
# COMMON QUESTIONS
# ============================================================

Q: Do I need to keep the notebooks?
A: No, they're for reference. The pipeline replaces them.

Q: Can I use old trained models?
A: You need to retrain because the new pipeline structure is different.
   Takes ~2 minutes on GPU or ~10 minutes on CPU.

Q: How do I extract just exercise questions?
A: pipeline.process_pdf('file.pdf', extraction_type='exercise')

Q: How do I process multiple PDFs?
A: Loop and call process_pdf() for each, or see
   scripts/pipeline_examples.py Example 3

Q: Where do results get saved?
A: data/processed/analyzed_*.csv (auto-timestamped)

Q: Can I use a different BERT model?
A: Yes: QuestionAnalysisPipeline(bert_model_name='bert-large-uncased')

Q: How accurate is the BERT classification?
A: Depends on training data. Current: ~65-70% accuracy on test set.
   Improve by: using more data, more epochs, or larger BERT model

# ============================================================
# TROUBLESHOOTING
# ============================================================

Issue: OCR quality is poor
Solution:
- Increase DPI in ocr_processor.py
- Adjust page ranges
- Verify PDF quality

Issue: BERT predictions are wrong
Solution:
- Train with more data
- Increase num_epochs
- Check that labels in CSV are correct

Issue: Memory error during OCR
Solution:
- Reduce page range (page_start, page_end)
- Process one PDF at a time
- Use CPU for OCR: processor = OCRProcessor()

Issue: Model not loading
Solution:
- Retrain model: pipeline.train_bert_model('train.csv')
- Check model path exists: data/models/bert_model/

# ============================================================
# NEW FEATURES (NOT IN OLD NOTEBOOKS)
# ============================================================

1. ✨ Automatic question paper extraction
   - Previously: Only exercise questions
   - Now: Can extract both types or choose one

2. ✨ Unified pipeline
   - Previously: Manual steps between notebooks
   - Now: One function call handles everything

3. ✨ Batch processing
   - Previously: One PDF at a time, manual
   - Now: Easy loop for multiple PDFs

4. ✨ Progress tracking
   - Previously: Silent execution
   - Now: Real-time status messages

5. ✨ Error handling
   - Previously: Crashes with cryptic errors
   - Now: Helpful error messages

6. ✨ Web interface
   - Previously: CLI notebooks
   - Now: Full-featured Streamlit app

7. ✨ Auto-export
   - Previously: Manual CSV saving
   - Now: Automatic timestamped exports

8. ✨ Summary statistics
   - Previously: Manual analysis
   - Now: Automatic chapter distribution, counts, etc.

# ============================================================
# FILE ORGANIZATION AFTER MIGRATION
# ============================================================

After setup, your structure will look like:

```
QuestionAnalyzer/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ocr_processor.py      ← Replace extractQuestions
│   ├── bert_analyzer.py      ← Replace Bert_mode
│   ├── pipeline.py           ← New unified interface
│   └── web-app-new.py        ← Updated web app
│
├── data/
│   ├── raw/                  ← Input PDFs
│   │   └── Primer.pdf
│   ├── processed/            ← Output CSVs
│   │   ├── final_questions.csv
│   │   └── analyzed_*.csv    ← Auto-generated
│   └── models/
│       └── bert_model/       ← Trained model
│
├── notebooks/                ← Keep for reference (optional)
│   ├── extractQuestions.ipynb
│   ├── Bert_mode.ipynb
│   └── ... (others)
│
├── scripts/
│   ├── pipeline_examples.py  ← Usage examples
│   └── __init__.py
│
├── STRUCTURE.md
├── PIPELINE_README.md        ← New documentation
├── MIGRATION_GUIDE.md        ← This file
└── requirements.txt
```

# ============================================================
# GETTING STARTED (3 STEPS)
# ============================================================

1. TRAIN THE MODEL (5 minutes)
   ```python
   from app.pipeline import QuestionAnalysisPipeline
   pipeline = QuestionAnalysisPipeline()
   pipeline.train_bert_model('data/processed/final_questions.csv')
   ```

2. PROCESS A PDF (2 minutes)
   ```python
   results = pipeline.process_pdf('data/raw/Primer.pdf')
   pipeline.save_results_to_csv(results)
   ```

3. CHECK RESULTS
   ```bash
   ls data/processed/analyzed_*.csv
   cat data/processed/analyzed_*.csv | head
   ```

Done! Your PDF is processed and results are saved.

# ============================================================
# NEXT STEPS
# ============================================================

□ Read PIPELINE_README.md for full documentation
□ Run scripts/pipeline_examples.py to see examples
□ Try the web interface: streamlit run app/web-app-new.py
□ Process your own PDFs
□ Improve model accuracy with more training data
□ Consider deploying the Streamlit app

Good luck! 🚀
