"""
Setup and initialization script for Question Analyzer
Run this to set up the project for first use
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    dirs = [
        'data/raw',
        'data/processed',
        'data/models',
        'notebooks',
        'static/css',
        'static/js',
        'static/images',
        'templates',
        'scripts',
        'tests',
        'docs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")


def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required = [
        'pandas',
        'torch',
        'transformers',
        'streamlit',
        'fitz',  # pymupdf
        'PIL',   # Pillow
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_data_files():
    """Check if training data exists"""
    print("\n📊 Checking data files...")
    
    training_file = 'data/processed/final_questions.csv'
    
    if os.path.exists(training_file):
        print(f"  ✅ Training data found: {training_file}")
        return True
    else:
        print(f"  ⚠️  Training data not found: {training_file}")
        print("     You'll need to provide this CSV for training")
        return False


def show_quick_start():
    """Show quick start guide"""
    print("\n" + "="*60)
    print("🚀 QUICK START GUIDE")
    print("="*60)
    
    print("""
1. PLACE YOUR DATA
   Put your training CSV in: data/processed/final_questions.csv
   (Must have 'Question Text' and 'Chapter_name' columns)

2. TRAIN THE MODEL (one-time, ~2-10 minutes)
   python -c "
from app.pipeline import QuestionAnalysisPipeline
p = QuestionAnalysisPipeline()
p.train_bert_model('data/processed/final_questions.csv')
   "

3. PROCESS A PDF
   Put your PDF in: data/raw/
   
   Then use the web interface:
   streamlit run app/web-app-new.py
   
   Or programmatically:
   python -c "
from app.pipeline import QuestionAnalysisPipeline
p = QuestionAnalysisPipeline()
r = p.process_pdf('data/raw/yourfile.pdf')
p.save_results_to_csv(r)
   "

4. VIEW RESULTS
   Check: data/processed/analyzed_*.csv

For more details, see:
  • PIPELINE_README.md - Full documentation
  • MIGRATION_GUIDE.md - Migration from old notebooks
  • scripts/pipeline_examples.py - Code examples
    """)


def main():
    """Run setup"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║            Question Analyzer Setup                     ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check data files
    data_ok = check_data_files()
    
    # Show quick start
    show_quick_start()
    
    # Summary
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE")
    print("="*60)
    
    if not deps_ok:
        print("\n⚠️  Install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    if not data_ok:
        print("\n⚠️  Add training data:")
        print("   - Place CSV in: data/processed/final_questions.csv")
        print("   - Must have columns: 'Question Text', 'Chapter_name'")
    
    print("""
Next steps:
1. Install dependencies (if needed)
2. Add training data (if needed)
3. Train model: python scripts/setup.py --train
4. Process PDFs: streamlit run app/web-app-new.py

Good luck! 🚀
    """)


if __name__ == "__main__":
    main()
