"""
Question Analyzer Web Application
Simplified interface for PDF processing and question analysis
"""

import streamlit as st
import os
from pathlib import Path
import pandas as pd

from app.pipeline import QuestionAnalysisPipeline


# Page configuration
st.set_page_config(
    page_title="Question Analyzer",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = QuestionAnalysisPipeline()
        st.session_state.pipeline_error = None
    except Exception as e:
        st.session_state.pipeline = None
        st.session_state.pipeline_error = str(e)

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar
st.sidebar.title("📚 Question Analyzer")
st.sidebar.write("Extract and analyze questions from PDFs using OCR and BERT")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📖 Train Model", "📄 Process PDF", "📊 Results"]
)

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.title("📚 Question Analyzer")
    st.write("## Welcome!")
    
    # Show warning if OCR not available
    if st.session_state.pipeline_error:
        st.warning(f"""
⚠️ **OCR Not Available**

{st.session_state.pipeline_error}

You can still use BERT for question classification. To enable PDF extraction, install:
```bash
pip install paddleocr paddlepaddle
```
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### Features:
        - 🔍 **OCR Extraction**: Extract text from PDFs
        - 📝 **Dual Extraction**: Exercise questions & Question papers
        - 🧠 **BERT Analysis**: Classify questions by chapter
        - 📊 **Detailed Reports**: Save and analyze results
        """)
    
    with col2:
        st.success("""
        ### Workflow:
        1. **Train** BERT model on labeled data
        2. **Upload** a new PDF file
        3. **Extract** questions using OCR
        4. **Analyze** with BERT classification
        5. **Export** results to CSV
        """)
    
    st.markdown("---")
    st.write("""
    ### How it works:
    
    This pipeline combines two powerful technologies:
    - **PaddleOCR**: Extracts text from PDF images
    - **BERT**: Classifies questions into chapters/categories
    
    You can extract two types of questions:
    - **Exercise Questions**: Pattern-based (e.g., "Exercise 2.1")
    - **Question Papers**: Numbered questions (e.g., "Q1.", "Q2.")
    """)

# ============================================================
# TRAIN MODEL PAGE
# ============================================================
elif page == "📖 Train Model":
    st.title("🧠 Train BERT Model")
    
    # Check for pipeline errors
    if st.session_state.pipeline_error:
        st.info("""
ℹ️ **Note:** OCR may not be available, but you can still train and use the BERT model.

The OCR feature requires additional dependencies:
```bash
pip install paddleocr paddlepaddle
```

For now, you can train the model and use just the BERT classification features.
        """)
    
    st.write("Train the BERT model on labeled question data.")
    
    uploaded_csv = st.file_uploader(
        "Upload training CSV (with 'Question Text' and 'Chapter_name' columns)",
        type=['csv'],
        key='train_csv'
    )
    
    if uploaded_csv is not None:
        # Show preview
        df = pd.read_csv(uploaded_csv)
        st.write("### Preview:")
        st.dataframe(df.head())
        
        # Training parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.slider("Batch Size", 4, 32, 8)
        with col2:
            num_epochs = st.slider("Epochs", 5, 20, 13)
        with col3:
            learning_rate = st.selectbox(
                "Learning Rate",
                [1e-5, 3e-5, 4e-5, 8e-5],
                format_func=lambda x: f"{x:.0e}"
            )
        
        if st.button("🚀 Start Training", key='train_btn'):
            # Save uploaded file temporarily
            temp_path = "temp_training_data.csv"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_csv.getvalue())
            
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.write("⏳ Training in progress...")
                
                st.session_state.pipeline.train_bert_model(
                    temp_path,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate
                )
                
                st.session_state.model_trained = True
                progress_bar.progress(100)
                st.success("✅ Model trained successfully!")
                
                # Save model
                model_dir = "data/models/bert_model"
                st.session_state.pipeline.bert_analyzer.save_model(model_dir)
                st.info(f"Model saved to: {model_dir}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# ============================================================
# PROCESS PDF PAGE
# ============================================================
elif page == "📄 Process PDF":
    st.title("📄 Process PDF File")
    
    # Check for pipeline errors
    if st.session_state.pipeline_error:
        st.error(f"""
❌ **Pipeline Initialization Error**

{st.session_state.pipeline_error}

**Solution:** Please install missing dependencies:
```bash
pip install paddleocr paddlepaddle
```

Then refresh this page.
        """)
        # return
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train a model first (📖 Train Model page)")
    else:
        st.success("✅ Model is trained and ready!")
    
    # File upload
    uploaded_pdf = st.file_uploader(
        "Upload a PDF file",
        type=['pdf'],
        key='process_pdf'
    )
    
    if uploaded_pdf is not None:
        # Save temporarily
        temp_pdf_path = "temp_file.pdf"
        with open(temp_pdf_path, 'wb') as f:
            f.write(uploaded_pdf.getvalue())
        
        st.write(f"📄 File: {uploaded_pdf.name}")
        
        # Processing options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extraction_type = st.selectbox(
                "Extraction Type",
                ["both", "exercise", "paper"],
                format_func=lambda x: {
                    "both": "📝 Both Types",
                    "exercise": "📖 Exercise Questions",
                    "paper": "📋 Question Papers"
                }.get(x)
            )
        
        with col2:
            page_start = st.number_input("Start Page", min_value=0, value=0)
        
        with col3:
            page_end = st.number_input("End Page (0 = all)", min_value=0, value=0)
            page_end = None if page_end == 0 else page_end
        
        # Process button
        if st.button("🔄 Process PDF", key='process_btn'):
            try:
                progress_bar = st.progress(0)
                
                progress_bar.progress(33)
                st.write("🔍 Extracting text with OCR...")
                
                progress_bar.progress(66)
                st.write("🧠 Analyzing with BERT...")
                
                results = st.session_state.pipeline.process_pdf(
                    temp_pdf_path,
                    extraction_type=extraction_type,
                    page_start=page_start,
                    page_end=page_end
                )
                
                st.session_state.results = results
                progress_bar.progress(100)
                
                # Show results
                st.success("✅ Processing completed!")
                
                summary = st.session_state.pipeline.get_summary(results)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Questions", summary['total_questions'])
                with col2:
                    st.metric("Exercise Questions", summary['total_exercises'])
                with col3:
                    st.metric("Question Papers", summary['total_papers'])
                
                # Save button
                if st.button("💾 Save Results to CSV", key='save_btn'):
                    st.session_state.pipeline.save_results_to_csv(results)
                    st.info("✅ Results saved to 'data/processed' folder!")
                
            except RuntimeError as e:
                st.error(f"❌ OCR Error: {str(e)}\n\nPlease install: `pip install paddleocr paddlepaddle`")
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

# ============================================================
# RESULTS PAGE
# ============================================================
elif page == "📊 Results":
    st.title("📊 Analysis Results")
    
    if st.session_state.results is None:
        st.info("Process a PDF first to see results")
    else:
        results = st.session_state.results
        summary = st.session_state.pipeline.get_summary(results)
        
        # Summary
        st.write("### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", summary['total_questions'])
        with col2:
            st.metric("Exercise Questions", summary['total_exercises'])
        with col3:
            st.metric("Question Papers", summary['total_papers'])
        
        # Exercise questions
        if results['analyzed_exercises']:
            st.write("### Exercise Questions")
            df_exercises = pd.DataFrame(results['analyzed_exercises'])
            st.dataframe(df_exercises, use_container_width=True)
            
            if 'chapters_in_exercises' in summary:
                st.bar_chart(pd.Series(summary['chapters_in_exercises']))
        
        # Question papers
        if results['analyzed_papers']:
            st.write("### Question Paper Questions")
            df_papers = pd.DataFrame(results['analyzed_papers'])
            st.dataframe(df_papers, use_container_width=True)
            
            if 'chapters_in_papers' in summary:
                st.bar_chart(pd.Series(summary['chapters_in_papers']))


# Footer
st.markdown("---")
st.write("💡 Tip: Use the sidebar to navigate between pages")
