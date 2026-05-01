"""
Example: Using the Question Analysis Pipeline
Demonstrates how to use the unified pipeline for end-to-end question analysis
"""

from app.pipeline import QuestionAnalysisPipeline
import pandas as pd


def example_1_train_model():
    """Example 1: Train BERT model on labeled questions"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training BERT Model")
    print("="*60)
    
    # Initialize pipeline
    pipeline = QuestionAnalysisPipeline()
    
    # Train on existing labeled data
    training_csv = "data/processed/final_questions.csv"
    
    print(f"\nTraining on: {training_csv}")
    pipeline.train_bert_model(
        training_csv,
        batch_size=8,
        num_epochs=13,
        learning_rate=4e-5
    )
    
    print("✅ Model training complete!")


def example_2_process_pdf():
    """Example 2: Process a new PDF file"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Process PDF File")
    print("="*60)
    
    # Initialize pipeline
    pipeline = QuestionAnalysisPipeline()
    
    # First, train the model (in real use, you'd load a pre-trained model)
    print("\nTraining model first...")
    pipeline.train_bert_model("data/processed/final_questions.csv")
    
    # Process a PDF file
    pdf_path = "data/raw/Primer.pdf"  # Your PDF file
    
    print(f"\nProcessing PDF: {pdf_path}")
    results = pipeline.process_pdf(
        pdf_path,
        extraction_type='both',  # Extract both exercises and papers
        page_start=0,
        page_end=50  # First 50 pages
    )
    
    # Display summary
    summary = pipeline.get_summary(results)
    print(f"\nResults Summary:")
    print(f"  Total Questions: {summary['total_questions']}")
    print(f"  Exercise Questions: {summary['total_exercises']}")
    print(f"  Question Papers: {summary['total_papers']}")
    
    # Show some results
    if results['analyzed_exercises']:
        print("\nSample Exercise Questions:")
        for q in results['analyzed_exercises'][:3]:
            print(f"  - {q['exercise_id']}: {q['question_text'][:50]}...")
            print(f"    Predicted Chapter: {q['predicted_chapter']} ({q['confidence']})")
    
    # Save results
    pipeline.save_results_to_csv(results)
    
    print("\n✅ Processing complete!")


def example_3_batch_process():
    """Example 3: Process multiple PDF files"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Process Multiple PDFs")
    print("="*60)
    
    import os
    from pathlib import Path
    
    pipeline = QuestionAnalysisPipeline()
    
    # Train model once
    print("Training model...")
    pipeline.train_bert_model("data/processed/final_questions.csv")
    
    # Process all PDFs in a directory
    pdf_dir = "data/raw/"
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    all_results = []
    
    for pdf_file in pdf_files[:3]:  # Process first 3 files
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            results = pipeline.process_pdf(
                str(pdf_file),
                extraction_type='exercise'  # Only exercise questions
            )
            all_results.append(results)
            
            summary = pipeline.get_summary(results)
            print(f"  ✅ Found {summary['total_questions']} questions")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Combine all results
    all_exercises = []
    for res in all_results:
        all_exercises.extend(res['analyzed_exercises'])
    
    # Create combined dataframe
    df = pd.DataFrame(all_exercises)
    print(f"\nCombined Results: {len(df)} total questions")
    print(f"Chapter Distribution:\n{df['predicted_chapter'].value_counts()}")


def example_4_load_and_predict():
    """Example 4: Load pre-trained model and predict on new data"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Load Trained Model and Predict")
    print("="*60)
    
    pipeline = QuestionAnalysisPipeline()
    
    # Load pre-trained model
    model_path = "data/models/bert_model"
    print(f"Loading model from: {model_path}")
    
    try:
        pipeline.bert_analyzer.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Predict on new questions
        test_questions = [
            "What is a pointer in C++?",
            "Explain the concept of inheritance",
            "How do you use templates in C++?",
            "What are virtual functions?",
            "Describe the STL containers"
        ]
        
        print("\nPredicting chapters for test questions:")
        predictions = pipeline.bert_analyzer.predict(test_questions)
        
        for pred in predictions:
            print(f"\n  Q: {pred['question']}")
            print(f"  → Chapter: {pred['predicted_chapter']} ({pred['confidence']})")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================
# RUN EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║        Question Analysis Pipeline - Examples           ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Uncomment the example you want to run:
    
    # Example 1: Train model
    # example_1_train_model()
    
    # Example 2: Process a PDF
    # example_2_process_pdf()
    
    # Example 3: Batch process multiple PDFs
    # example_3_batch_process()
    
    # Example 4: Load model and predict
    # example_4_load_and_predict()
    
    print("""
    
    ╔════════════════════════════════════════════════════════╗
    ║                    Quick Start Guide                   ║
    ╠════════════════════════════════════════════════════════╣
    ║                                                        ║
    ║  1. To train a model:                                  ║
    ║     pipeline.train_bert_model('path/to/training.csv') ║
    ║                                                        ║
    ║  2. To process a PDF:                                  ║
    ║     results = pipeline.process_pdf('file.pdf')        ║
    ║                                                        ║
    ║  3. To save results:                                   ║
    ║     pipeline.save_results_to_csv(results)             ║
    ║                                                        ║
    ║  4. To get summary:                                    ║
    ║     summary = pipeline.get_summary(results)           ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    """)
