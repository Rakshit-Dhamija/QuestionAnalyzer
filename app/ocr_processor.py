"""
OCR Processing Module
Handles PDF extraction for both exercise questions and question papers
"""

import csv
import re
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict


class OCRProcessor:
    """OCR processor for extracting text from PDFs"""
    
    def __init__(self, lang="en"):
        """Initialize PaddleOCR (lazy - will be initialized on first use)"""
        self.ocr = None
        self.lang = lang
        self.ocr_initialized = False
        self.ocr_error = None
    
    def _initialize_ocr(self):
        """Lazy initialize PaddleOCR on first use"""
        if self.ocr_initialized:
            return
        
        if self.ocr_error:
            raise RuntimeError(f"OCR initialization failed: {self.ocr_error}")
        
        try:
            print("[OCR] Initializing PaddleOCR... (first use, may take a moment)")
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
            self.ocr_initialized = True
            print("[OCR] PaddleOCR initialized successfully!")
        except Exception as e:
            self.ocr_error = str(e)
            print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_from_pdf(self, pdf_path: str, page_start: int = 0, page_end: int = None) -> str:
        """
        Extract raw text from PDF pages
        
        Args:
            pdf_path: Path to PDF file
            page_start: Starting page number (0-indexed)
            page_end: Ending page number (None = all pages)
            
        Returns:
            Extracted text content
        """
        # Initialize OCR on first use
        self._initialize_ocr()
        
        extracted_text = ""
        
        with fitz.open(pdf_path) as pdf:
            total_pages = len(pdf)
            end_page = page_end if page_end else total_pages
            
            for pg_num in range(page_start, min(end_page, total_pages)):
                page = pdf[pg_num]
                
                # Render page at higher DPI for better OCR
                mat = fitz.Matrix(1.5, 1.5)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to numpy array
                img = np.array(Image.frombytes("RGB", [pm.width, pm.height], pm.samples))
                
                # Perform OCR
                result = self.ocr.ocr(img, cls=True)
                
                extracted_text += f"\n--- Page {pg_num + 1} ---\n"
                
                if result:
                    for line in result:
                        for entry in line:
                            text = entry[1][0].strip()
                            extracted_text += text + "\n"
        
        return extracted_text
    
    def extract_exercise_questions(self, extracted_text: str) -> List[Dict[str, str]]:
        """
        Extract exercise questions from OCR text
        Pattern: "Exercise X.Y: Question text"
        
        Args:
            extracted_text: Raw OCR extracted text
            
        Returns:
            List of dicts with 'exercise_id' and 'question_text'
        """
        questions = []
        exercise_pattern = re.compile(r"^(Exercise\s+\d+\.\d+):?\s*(.*)")
        
        lines = extracted_text.split('\n')
        current_exercise = None
        current_question = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            
            match = exercise_pattern.match(line)
            if match:
                # Save previous question if exists
                if current_exercise and current_question:
                    questions.append({
                        'exercise_id': current_exercise,
                        'question_text': ' '.join(current_question),
                        'type': 'exercise'
                    })
                
                current_exercise = match.group(1)
                current_question = [match.group(2)]
            elif current_exercise:
                # Continue accumulating question text
                current_question.append(line)
        
        # Save last question
        if current_exercise and current_question:
            questions.append({
                'exercise_id': current_exercise,
                'question_text': ' '.join(current_question),
                'type': 'exercise'
            })
        
        return questions
    
    def extract_question_papers(self, extracted_text: str) -> List[Dict[str, str]]:
        """
        Extract questions from question papers
        Pattern: "Q1.", "Q2.", etc. or "1.", "2.", etc.
        
        Args:
            extracted_text: Raw OCR extracted text
            
        Returns:
            List of dicts with 'question_id' and 'question_text'
        """
        questions = []
        # Match "Q1.", "Q2." or "1.", "2." patterns
        q_pattern = re.compile(r"^(?:Q\s*)?(\d+)[.)\s]+(.+)")
        
        lines = extracted_text.split('\n')
        current_q_num = None
        current_question = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            
            match = q_pattern.match(line)
            if match:
                # Save previous question if exists
                if current_q_num and current_question:
                    questions.append({
                        'question_id': f"Q{current_q_num}",
                        'question_text': ' '.join(current_question),
                        'type': 'paper'
                    })
                
                current_q_num = match.group(1)
                current_question = [match.group(2)]
            elif current_q_num:
                current_question.append(line)
        
        # Save last question
        if current_q_num and current_question:
            questions.append({
                'question_id': f"Q{current_q_num}",
                'question_text': ' '.join(current_question),
                'type': 'paper'
            })
        
        return questions
    
    def extract_all(self, pdf_path: str, page_start: int = 0, 
                   page_end: int = None) -> Dict[str, List[Dict]]:
        """
        Extract both exercise questions and question paper questions
        
        Args:
            pdf_path: Path to PDF file
            page_start: Starting page number
            page_end: Ending page number
            
        Returns:
            Dictionary with 'exercises' and 'papers' keys
        """
        # Extract raw text from PDF
        raw_text = self.extract_from_pdf(pdf_path, page_start, page_end)
        
        # Extract both types
        exercises = self.extract_exercise_questions(raw_text)
        papers = self.extract_question_papers(raw_text)
        
        return {
            'exercises': exercises,
            'papers': papers,
            'raw_text': raw_text
        }
