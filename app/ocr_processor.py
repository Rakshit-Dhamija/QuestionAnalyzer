"""
OCR Processing Module
Handles PDF extraction for both exercise questions and question papers
"""

import csv
import hashlib
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
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_mkldnn"] = "0"
    os.environ["ONEDNN_VERBOSE"] = "0"


def _short_error(message: str, max_lines: int = 12) -> str:
    """Keep app-facing OCR errors readable."""
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    useful = [
        line for line in lines
        if not line.startswith("File ")
        and not line.startswith("Traceback")
        and not line.startswith("^")
    ]
    return "\n".join(useful[:max_lines]) or "Unknown OCR error."


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

    def _cache_key(self, pdf_path: str, page_start: int = 0, page_end: int = None) -> str:
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)

        payload = {
            "file_hash": hasher.hexdigest(),
            "lang": self.lang,
            "page_start": page_start,
            "page_end": page_end,
            "version": 2,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _get_next_scan_folder(self, base_dir: str = "ocr_scans") -> str:
        """Get the next numbered scan folder (scan_001, scan_002, etc.)"""
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith("scan_") and d[5:].isdigit()]
        next_num = len(existing) + 1
        return os.path.join(base_dir, f"scan_{next_num:03d}")

    def _cache_path(self, cache_key: str) -> str:
        cache_dir = os.path.join("data", "cache", "ocr")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{cache_key}.json")

    def _extract_from_pdf_in_worker(
        self, pdf_path: str, page_start: int = 0, page_end: int = None,
        timeout_seconds: int = 900,
    ) -> str:
        """Run PaddleOCR in a separate process to avoid Paddle/PyTorch pybind conflicts."""
        _disable_paddle_gpu()

        cache_key = self._cache_key(pdf_path, page_start, page_end)
        cache_path = self._cache_path(cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("ok"):
                print(f"[OCR] Using cached OCR result from {cache_path}")
                return cached.get("text", "")

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        output_path = output_file.name
        output_file.close()

        # Create numbered scan folder
        scan_folder = self._get_next_scan_folder()

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
            "--scan-folder",
            os.path.abspath(scan_folder),
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
                timeout=timeout_seconds,
            )
            payload = {}
            if os.path.getsize(output_path) > 0:
                with open(output_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

            if not payload.get("ok"):
                detail = (
                    payload.get("error")
                    or payload.get("traceback")
                    or (completed.stderr or completed.stdout or "").strip()
                )
                raise RuntimeError(_short_error(detail or f"OCR worker exited with code {completed.returncode}"))

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(detail or f"OCR worker exited with code {completed.returncode}")

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            print(
                f"[OCR] Extracted text with {payload.get('backend', 'unknown')} "
                f"from {payload.get('pages_processed', 0)} page(s). "
                f"Scans saved to {scan_folder}"
            )
            return payload.get("text", "")
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"OCR timed out after {timeout_seconds} seconds. Try a smaller page range.") from e
        except Exception as e:
            raise RuntimeError(_short_error(str(e))) from e
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
        Extract exercise questions from OCR text.
        Supports many formats like:
          - Exercise 2.1: text
          - EXERCISE 2.1
          - Ex 2.1.1
          - Ex. 2-1
          - Exercise 5
        
        Args:
            extracted_text: Raw OCR extracted text
            
        Returns:
            List of dicts with 'exercise_id' and 'question_text'
        """
        questions = []
        exercise_pattern = re.compile(
            r"^(?:Exercise|Exercises|Ex|Ex\.|Question|Questions)\s*[:.]?\s*(?P<number>\d+(?:[.-]\d+)*)(?:\s*[:.)-]+\s*(?P<text>.*)|\s+(?P<text2>.*))$",
            re.IGNORECASE,
        )
        optional_bracket_pattern = re.compile(r"^\(?\s*(?P<number>\d+(?:[.-]\d+)*)\)?\s*(?:[:.)-]+\s*(?P<text>.*))$")
        question_heading_pattern = re.compile(r'^(?:Exercise|Exercises|Ex|Ex\.|Question|Questions)\s*$', re.IGNORECASE)

        lines = extracted_text.split('\n')
        current_exercise = None
        current_question = []
        current_heading = None

        def save_current():
            if current_exercise and current_question:
                text = ' '.join(current_question).strip()
                if text:
                    questions.append({
                        'exercise_id': current_exercise,
                        'question_text': text,
                        'type': 'exercise'
                    })

        for line in lines:
            raw_line = line.strip()
            if not raw_line or raw_line.startswith("---"):
                continue

            match = exercise_pattern.match(raw_line)
            if match:
                save_current()

                number = match.group('number')
                text = match.group('text') or match.group('text2') or ''
                keyword = raw_line.split()[0].capitalize()
                current_exercise = f"{keyword} {number}"
                current_question = [text] if text else []
                current_heading = None
                continue

            # Sometimes headings are broken across lines, like "Exercise" then "2.1" or "Question" then "1"
            heading_match = question_heading_pattern.match(raw_line)
            if heading_match:
                save_current()
                current_heading = heading_match.group(0).capitalize()
                current_exercise = None
                current_question = []
                continue

            number_only = optional_bracket_pattern.match(raw_line)
            if number_only and current_heading is not None:
                save_current()
                current_exercise = f"{current_heading} {number_only.group('number')}"
                text = number_only.group('text') or ''
                current_question = [text] if text else []
                continue

            if current_exercise:
                current_question.append(raw_line)

        save_current()

        if not questions:
            questions = self.extract_numbered_study_questions(extracted_text)

        return questions

    def extract_numbered_study_questions(self, extracted_text: str) -> List[Dict[str, str]]:
        """
        Extract textbook questions from sections such as Review Questions,
        Problems, Exercises, or Self-Test where items are simply numbered.
        This helps books like the 8051 PDF that do not use "Exercise 2.1".
        """
        questions = []
        section_pattern = re.compile(
            r"^(?:review\s+questions|problems|exercises?|self[-\s]?test|questions)$",
            re.IGNORECASE,
        )
        stop_pattern = re.compile(
            r"^(?:answers?\s+to|summary|recommended|references|bibliography|objectives|"
            r"chapter\s+\d+|\d+\s*:\s+[A-Z]|[A-Z][A-Z\s]{8,})",
            re.IGNORECASE,
        )
        item_pattern = re.compile(r"^(\d{1,3})\s*[.)]\s+(.+)$")
        marks_pattern = re.compile(r"^\(?\s*\d+\s*(?:[+*]\s*\d+\s*)*(?:=\s*\d+)?\s*\)?$")

        in_section = False
        current_id = None
        current_question = []
        section_name = "Question"

        def save_current():
            if current_id and current_question:
                text = " ".join(current_question).strip()
                if text:
                    questions.append({
                        "exercise_id": current_id,
                        "question_text": text,
                        "type": "exercise",
                    })

        for raw_line in extracted_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            if not line or line.startswith("---"):
                continue

            if section_pattern.match(line):
                save_current()
                in_section = True
                current_id = None
                current_question = []
                section_name = line.title()
                continue

            if in_section and stop_pattern.match(line):
                save_current()
                in_section = False
                current_id = None
                current_question = []
                continue

            if not in_section or marks_pattern.match(line):
                continue

            item_match = item_pattern.match(line)
            if item_match:
                save_current()
                current_id = f"{section_name} {item_match.group(1)}"
                current_question = [item_match.group(2).strip()]
            elif current_id:
                current_question.append(line)

        save_current()
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
        # Handles "Q.1(a)", "Q1(b)", "Q.3", "Question 1", "1.", "2)" and similar OCR variants.
        q_prefixed_pattern = re.compile(
            r"^(?:Q(?:uestion)?)(?:\s*\.|\s*)\s*(\d{1,3})\s*"
            r"(?:[.)-]?\s*(?:\(([a-zA-Z])\)|([a-zA-Z])\)))?"
            r"[\s.)-]*(.*)$",
            re.IGNORECASE,
        )
        question_word_pattern = re.compile(
            r"^Question\s*(\d{1,3})(?:\s*[:.,)-]+\s*(.*))?$",
            re.IGNORECASE,
        )
        numbered_pattern = re.compile(
            r"^(\d{1,3})\s*[.)-]\s*"
            r"(?:\(([a-zA-Z])\)|([a-zA-Z])\))?\s*(.*)$",
            re.IGNORECASE,
        )
        marks_pattern = re.compile(r"^(?:\(?\s*)?(?:\d+\s*)?(?:[+*]\s*\d+\s*)*(?:=\s*\d+)?\)?$")
        header_pattern = re.compile(
            r"^(?:roll\b|thapar\b|department\b|exam(?:ination)?\b|course\b|time\b|"
            r"name of faculty\b|note:|b\.?\s*e\.?)",
            re.IGNORECASE,
        )
        
        lines = extracted_text.split('\n')
        current_q_num = None
        current_question = []

        def save_current_question():
            if current_q_num and current_question:
                questions.append({
                    'question_id': current_q_num,
                    'question_text': ' '.join(current_question).strip(),
                    'type': 'paper'
                })
        
        for line in lines:
            line = re.sub(r"\s+", " ", line.strip())
            if not line or line.startswith("---"):
                continue
            if header_pattern.match(line):
                save_current_question()
                current_q_num = None
                current_question = []
                continue
            if marks_pattern.match(line):
                continue
            
            match = (
                q_prefixed_pattern.match(line)
                or question_word_pattern.match(line)
                or numbered_pattern.match(line)
            )
            if match:
                # Save previous question if exists
                save_current_question()
                
                groups = match.groups()
                part = (groups[1] or groups[2] or "").lower() if len(groups) >= 3 else ""
                q_num = match.group(1)
                current_q_num = f"Q{q_num}{f'({part})' if part else ''}"
                first_line = groups[3].strip() if len(groups) >= 4 and groups[3] is not None else ''
                current_question = [first_line] if first_line else []
            elif current_q_num:
                current_question.append(line)
        
        # Save last question
        save_current_question()
        
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
