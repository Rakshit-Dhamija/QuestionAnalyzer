"""
OCR Processing Module
Handles PDF extraction for both exercise questions and question papers
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Tuple, Dict


def _disable_paddle_gpu():
    """Keep PaddleOCR on CPU to avoid Paddle/PyTorch CUDA pybind conflicts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLAGS_selected_gpus"] = ""
    os.environ["PADDLE_PDX_DEVICE"] = "cpu"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


class OCRProcessor:
    """OCR processor for extracting text from PDFs"""
    
    def __init__(self, lang="en"):
        """Initialize PaddleOCR (lazy - will be initialized on first use)"""
        self.ocr = None
        self.lang = lang
        self.ocr_initialized = False
        self.ocr_error = None
    
    def _initialize_ocr(self):
        """Validate that OCR can be run in the isolated worker process."""
        if self.ocr_error:
            raise RuntimeError(f"OCR initialization failed: {self.ocr_error}")

    def _create_ocr_instance(self):
        """
        PaddleOCR must not be initialized in the Streamlit/PyTorch process.
        OCR is executed by app.ocr_worker in a separate Python process.
        """
        raise RuntimeError("PaddleOCR is initialized in the isolated OCR worker process.")

    def _extract_from_pdf_in_worker(
        self, pdf_path: str, page_start: int = 0, page_end: int = None
    ) -> str:
        """Run PaddleOCR in a separate process to avoid Paddle/PyTorch pybind conflicts."""
        _disable_paddle_gpu()

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        output_path = output_file.name
        output_file.close()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["FLAGS_selected_gpus"] = ""
        env["PADDLE_PDX_DEVICE"] = "cpu"
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        env["FLAGS_use_mkldnn"] = "0"
        env["FLAGS_enable_mkldnn"] = "0"
        env["ONEDNN_VERBOSE"] = "0"

        command = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "ocr_worker.py"),
            "--pdf",
            os.path.abspath(pdf_path),
            "--start",
            str(page_start),
            "--lang",
            self.lang,
            "--out",
            output_path,
        ]
        if page_end is not None:
            command.extend(["--end", str(page_end)])

        try:
            print("[OCR] Running PaddleOCR in isolated CPU worker...")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            )
            payload = {}
            if os.path.getsize(output_path) > 0:
                with open(output_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

            if not payload.get("ok"):
                detail = (
                    payload.get("traceback")
                    or payload.get("error")
                    or (completed.stderr or completed.stdout or "").strip()
                )
                raise RuntimeError(detail or f"OCR worker exited with code {completed.returncode}")

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(detail or f"OCR worker exited with code {completed.returncode}")

            return payload.get("text", "")
        except Exception:
            raise
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

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
        self._initialize_ocr()
        return self._extract_from_pdf_in_worker(pdf_path, page_start, page_end)
    
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
