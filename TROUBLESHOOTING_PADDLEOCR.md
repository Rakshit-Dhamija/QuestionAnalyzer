"""
TROUBLESHOOTING: PaddleOCR Installation Issues

If you're seeing errors related to PaddlePaddle or PaddleOCR, 
this guide will help you resolve them.
"""

# ============================================================
# ERROR: paddlex/utils/device.py - get_default_device() 
# or paddle/base/core.py ImportError
# ============================================================

Problem:
--------
You see an error like:
  File ".../paddle/base/core.py", line 380, in <module>
    raise e
  ImportError: ... (related to paddle core module)

Cause:
------
PaddlePaddle has compatibility issues or is not properly installed.

Solution (Choose One):
---------------------

OPTION 1: Quick Fix (Recommended for Windows)
=============================================
Run these commands:

1. Uninstall paddle:
   pip uninstall paddle paddlepaddle -y

2. Install CPU version:
   pip install paddlepaddle==2.5.0 -i https://pypi.org/simple

3. Install paddleocr:
   pip install paddleocr

4. Test:
   python -c "from paddleocr import PaddleOCR; print('OK')"


OPTION 2: Use Conda (More Reliable)
===================================
1. Install conda (if not already installed)
   https://www.anaconda.com/download

2. Create new environment:
   conda create -n paddle_env python=3.10

3. Activate:
   conda activate paddle_env

4. Install paddle and paddleocr:
   conda install -c conda-forge paddlepaddle paddleocr

5. Install other dependencies:
   pip install -r requirements.txt


OPTION 3: Use without OCR
=========================
If you still have issues, you can use just the BERT classification:

from app.bert_analyzer import BERTQuestionAnalyzer

analyzer = BERTQuestionAnalyzer()
analyzer.train(questions, labels)
predictions = analyzer.predict(new_questions)

This works without OCR. You just can't extract from PDFs.


OPTION 4: Docker (Most Reliable)
================================
If you have Docker, run:

docker pull continuumio/anaconda3
docker run -it --name paddle_env continuumio/anaconda3 /bin/bash

Then inside the container:
pip install paddlepaddle paddleocr torch transformers streamlit pandas


OPTION 5: Use Different OCR (Tesseract)
=======================================
Replace PaddleOCR with Tesseract:

1. Install Tesseract:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: brew install tesseract
   - Linux: apt-get install tesseract-ocr

2. Create new ocr_processor.py with Tesseract

See: TESSERACT_OCR_ALTERNATIVE.md (coming soon)


WORKAROUND: Process PDFs Manually
=================================
If OCR won't work, you can:

1. Use a free online PDF to text converter:
   - https://pdftotext.com/
   - https://smallpdf.com/pdf-to-text

2. Or use a desktop PDF reader to copy text manually

3. Save extracted text to a CSV:
   exercise_id,question_text
   Exercise 2.1,Question text here...
   Exercise 2.2,Another question...

4. Then use BERT to classify:
   pipeline.bert_analyzer.train(questions, labels)
   results = pipeline.bert_analyzer.predict(new_questions)


WORKAROUND: Check Python Version
=================================
PaddlePaddle may require specific Python versions.

Check your version:
  python --version

Recommended: Python 3.9 or 3.10
May have issues: Python 3.11+ or 3.7-

To use Python 3.10:
  conda create -n py310 python=3.10
  conda activate py310
  pip install -r requirements.txt


Check if Installation Worked
============================

Test 1 - Import paddle:
  python -c "import paddle; print(f'Paddle: {paddle.__version__}')"

Test 2 - Import paddleocr:
  python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

Test 3 - Initialize OCR:
  python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(); print('OCR initialized!')"

If Test 3 fails, try OPTION 1 or 2 above.


Current Workaround: Use Without OCR
====================================

The web app is now designed to work WITHOUT OCR if needed:

1. Train the BERT model:
   streamlit run app/web-app-new.py
   → Go to "Train Model" page
   → Upload training CSV
   → Train successfully without OCR!

2. Classify manually:
   You can use BERT to classify questions without PDF extraction.
   Just provide the questions as a CSV or text.

3. Process PDFs with alternative tools:
   Extract text from PDF → Save to CSV → Use BERT

See: app/web-app-new.py for full implementation


Detailed Error Messages
======================

ERROR 1: ImportError: cannot import name 'core'
SOLUTION: Option 1 or 2 above

ERROR 2: RuntimeError: Unsupported device type
SOLUTION: Reinstall with correct architecture (CPU vs GPU)

ERROR 3: ModuleNotFoundError: No module named 'paddle'
SOLUTION: pip install paddlepaddle

ERROR 4: CUDA version mismatch
SOLUTION: Use CPU version: pip install paddlepaddle==2.5.0


Getting Help
============

1. Check if this file has your error → Fix as described

2. Search GitHub issues:
   https://github.com/PaddlePaddle/PaddleOCR/issues

3. Check System:
   - Windows: Use Option 1
   - Mac: Use Option 2 (Conda)
   - Linux: Use Option 2 (Conda) or Option 4 (Docker)

4. As Last Resort: Use OPTION 3 (No OCR) or OPTION 5 (Tesseract)


Test the Application Works
==========================

Even if OCR fails, test other features:

python -c "
from app.bert_analyzer import BERTQuestionAnalyzer

analyzer = BERTQuestionAnalyzer()
print('✅ BERT Analyzer works!')
"

python -c "
import streamlit as st
print('✅ Streamlit works!')
"

python -c "
import pandas as pd
import torch
print('✅ Core dependencies work!')
"


If BERT works, you're good to go - you can use the app without OCR!


================================================================
Still stuck? Try the latest workaround:
Use Python 3.9 + Option 1 (Quick Fix)
================================================================
