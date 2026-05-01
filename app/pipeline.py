"""
Unified Question Analysis Pipeline
Combines OCR extraction and BERT classification in a single workflow
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os

from .ocr_processor import OCRProcessor


class QuestionAnalysisPipeline:
    """Main pipeline for end-to-end question extraction and analysis"""
    
    def __init__(self, bert_model_name: str = 'bert-base-uncased'):
        """
        Initialize the analysis pipeline
        
Args:
            bert_model_name: Pre-trained BERT model to use
        """
        # Initialize OCR processor (lazy - will initialize on first use)
        try:
            self.ocr_processor = OCRProcessor()
        except Exception as e:
            print(f"[WARNING] OCR processor initialization deferred: {e}")
            self.ocr_processor = None
        
        self.bert_model_name = bert_model_name
        self.bert_analyzer = None
        self.extraction_type = None  # 'exercise' or 'paper'

    def _get_bert_analyzer(self):
        """Load PyTorch/BERT only when training or prediction is actually needed."""
        if self.bert_analyzer is None:
            from .bert_analyzer import BERTQuestionAnalyzer

            self.bert_analyzer = BERTQuestionAnalyzer(self.bert_model_name)
        return self.bert_analyzer
    
    def train_bert_model(self, training_csv_path: str, **kwargs):
        """
        Train BERT model on existing labeled data
        
        Args:
            training_csv_path: Path to CSV with labeled questions
            **kwargs: Additional arguments for training (batch_size, num_epochs, learning_rate)
        """
        print("[BERT] Loading training data...")
        bert_analyzer = self._get_bert_analyzer()
        questions, labels = bert_analyzer.load_training_data(training_csv_path)
        
        print("[BERT] Starting model training...")
        bert_analyzer.train(questions, labels, **kwargs)
        print("[BERT] Training completed!")
    
    def process_pdf(self, pdf_path: str, extraction_type: str = 'both',
                   page_start: int = 0, page_end: Optional[int] = None) -> Dict:
        """
        Process a PDF file end-to-end
        
        Args:
            pdf_path: Path to PDF file
            extraction_type: 'exercise', 'paper', or 'both'
            page_start: Starting page (0-indexed)
            page_end: Ending page (None for all)
            
        Returns:
            Dictionary with extracted and analyzed questions
        """
        if self.ocr_processor is None:
            raise RuntimeError(
                "OCR processor not available. "
                "Please ensure PaddlePaddle and PaddleOCR are properly installed. "
                "Run: pip install paddleocr paddlepaddle"
            )
        
        print(f"\n[PIPELINE] Starting PDF processing: {pdf_path}")
        
        # Step 1: Extract text and questions
        print(f"[OCR] Extracting text from PDF...")
        extracted_data = self.ocr_processor.extract_all(pdf_path, page_start, page_end)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'pdf_path': pdf_path,
            'extraction_type': extraction_type,
            'exercises': [],
            'papers': [],
            'analyzed_exercises': [],
            'analyzed_papers': []
        }
        
        # Step 2: Classify questions with BERT
        if extraction_type in ['exercise', 'both'] and extracted_data['exercises']:
            print(f"[BERT] Analyzing {len(extracted_data['exercises'])} exercise questions...")
            exercise_texts = [q['question_text'] for q in extracted_data['exercises']]
            analyzed = self._get_bert_analyzer().predict(exercise_texts)
            
            # Merge with original extraction
            for original, analyzed_result in zip(extracted_data['exercises'], analyzed):
                merged = {
                    **original,
                    'predicted_chapter': analyzed_result['predicted_chapter'],
                    'confidence': analyzed_result['confidence']
                }
                results['exercises'].append(original)
                results['analyzed_exercises'].append(merged)
        
        if extraction_type in ['paper', 'both'] and extracted_data['papers']:
            print(f"[BERT] Analyzing {len(extracted_data['papers'])} question paper questions...")
            paper_texts = [q['question_text'] for q in extracted_data['papers']]
            analyzed = self._get_bert_analyzer().predict(paper_texts)
            
            # Merge with original extraction
            for original, analyzed_result in zip(extracted_data['papers'], analyzed):
                merged = {
                    **original,
                    'predicted_chapter': analyzed_result['predicted_chapter'],
                    'confidence': analyzed_result['confidence']
                }
                results['papers'].append(original)
                results['analyzed_papers'].append(merged)
        
        print("[PIPELINE] Processing completed!")
        
        return results
    
    def save_results_to_csv(self, results: Dict, output_dir: str = 'data/processed'):
        """
        Save analysis results to CSV files
        
        Args:
            results: Results dictionary from process_pdf()
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save exercises
        if results['analyzed_exercises']:
            df = pd.DataFrame(results['analyzed_exercises'])
            filename = os.path.join(output_dir, f"analyzed_exercises_{timestamp}.csv")
            df.to_csv(filename, index=False)
            print(f"[SAVE] Exercises saved to: {filename}")
        
        # Save question papers
        if results['analyzed_papers']:
            df = pd.DataFrame(results['analyzed_papers'])
            filename = os.path.join(output_dir, f"analyzed_papers_{timestamp}.csv")
            df.to_csv(filename, index=False)
            print(f"[SAVE] Question papers saved to: {filename}")
    
    def get_summary(self, results: Dict) -> Dict:
        """
        Get summary statistics from analysis results
        
        Args:
            results: Results dictionary from process_pdf()
            
        Returns:
            Summary dictionary with counts and statistics
        """
        summary = {
            'total_exercises': len(results['analyzed_exercises']),
            'total_papers': len(results['analyzed_papers']),
            'total_questions': len(results['analyzed_exercises']) + len(results['analyzed_papers'])
        }
        
        # Chapter distribution for exercises
        if results['analyzed_exercises']:
            exercises_df = pd.DataFrame(results['analyzed_exercises'])
            summary['chapters_in_exercises'] = exercises_df['predicted_chapter'].value_counts().to_dict()
        
        # Chapter distribution for papers
        if results['analyzed_papers']:
            papers_df = pd.DataFrame(results['analyzed_papers'])
            summary['chapters_in_papers'] = papers_df['predicted_chapter'].value_counts().to_dict()
        
        return summary
