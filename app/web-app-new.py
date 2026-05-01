"""
Question Analyzer Web Application
Flow: Step 1 (OCR Extract) → Step 2 (Train Model) → Step 3 (Analyze & Results)
"""

import os

# Must be set BEFORE importing transformers to prevent it from loading TensorFlow.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import io
import streamlit as st
import pandas as pd
from pathlib import Path

from app.pipeline import QuestionAnalysisPipeline

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Question Analyzer",
    page_icon="📚",
    layout="wide"
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
defaults = {
    "pipeline": None,
    "pipeline_error": None,
    "model_trained": False,
    # Step 1 outputs
    "exercise_df": None,   # DataFrame: exercise_id | question_text | chapter_name
    "paper_df": None,      # DataFrame: question_id | question_text
    "chapter_map": {},     # {1: "Chapter Name", 2: "Chapter Name", ...}
    # Step 3 output
    "analysis_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Initialize pipeline once
if st.session_state.pipeline is None and st.session_state.pipeline_error is None:
    try:
        st.session_state.pipeline = QuestionAnalysisPipeline()
    except Exception as e:
        st.session_state.pipeline_error = str(e)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("📚 Question Analyzer")
st.sidebar.caption("OCR + BERT Chapter Classification")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📄 Step 1: OCR Extraction",
        "🧠 Step 2: Train Model",
        "📊 Step 3: Analyze & Results",
    ],
)

# Pipeline status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Pipeline Status")
s1 = "✅" if st.session_state.exercise_df is not None or st.session_state.paper_df is not None else "⏳"
s2 = "✅" if st.session_state.model_trained else "⏳"
s3 = "✅" if st.session_state.analysis_results is not None else "⏳"
st.sidebar.markdown(f"{s1} Step 1: OCR Extraction")
st.sidebar.markdown(f"{s2} Step 2: Model Trained")
st.sidebar.markdown(f"{s3} Step 3: Results Ready")


# ─────────────────────────────────────────────
# Helper: parse chapter index from raw OCR text
# ─────────────────────────────────────────────
def parse_chapter_index(index_text: str) -> dict:
    """
    Extract {chapter_number: chapter_name} from OCR'd index/ToC text.
    Handles patterns like:
      - "Chapter 1: Getting Started"
      - "1. Getting Started"
      - "1  Getting Started"
    """
    chapter_map = {}
    patterns = [
        re.compile(r'chapter\s+(\d+)[:\s.\-–]+(.+)', re.IGNORECASE),
        re.compile(r'^(\d+)[.\s)\-]+([A-Z][^\d]{3,})', re.MULTILINE),
    ]
    for line in index_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('---'):
            continue
        for pat in patterns:
            m = pat.match(line)
            if m:
                try:
                    num = int(m.group(1))
                    name = m.group(2).strip().rstrip('.')
                    if name:
                        chapter_map[num] = name
                except ValueError:
                    pass
                break
    return chapter_map


def chapter_num_from_exercise_id(exercise_id: str) -> int | None:
    """Extract chapter number from 'Exercise 2.3' → 2"""
    m = re.search(r'(\d+)', exercise_id)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📚 Question Analyzer")
    st.markdown("### Analyze any textbook or question paper — chapter by chapter.")

    if st.session_state.pipeline_error:
        st.warning(
            f"⚠️ **OCR Not Fully Available:** {st.session_state.pipeline_error}\n\n"
            "Install with: `pip install paddleocr paddlepaddle`"
        )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            ### 📄 Step 1 — OCR Extraction
            Upload your **textbook PDF** (exercises) and **question paper PDF**.
            Optionally provide the book's **index/ToC** to map chapter numbers to names.

            👉 Outputs two CSVs ready for analysis.
            """
        )
    with col2:
        st.info(
            """
            ### 🧠 Step 2 — Train Model
            Use the **exercise questions CSV** (with chapter labels extracted from the book)
            to train a **BERT classifier**.

            👉 The model learns which questions belong to which chapter.
            """
        )
    with col3:
        st.info(
            """
            ### 📊 Step 3 — Analyze & Results
            Run the trained model on the **question paper CSV**.
            See **chapter-wise weightage**: how many questions from each chapter appear.

            👉 Export results to CSV.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        **How the pieces fit together:**

        ```
        Textbook PDF ──► OCR ──► Exercise Questions + Chapter Names
                                         │
        Question Paper PDF ──► OCR ──► Paper Questions
                                         │
                                         ▼
                              Train BERT on Exercise CSV
                                         │
                                         ▼
                             Classify Question Paper Questions
                                         │
                                         ▼
                              Chapter-wise Weightage Report
        ```
        """
    )


# ─────────────────────────────────────────────
# STEP 1 — OCR EXTRACTION
# ─────────────────────────────────────────────
elif page == "📄 Step 1: OCR Extraction":
    st.title("📄 Step 1: OCR Extraction")
    st.markdown(
        "Extract questions from your **textbook** and **question paper** using OCR. "
        "Optionally, upload the book's **index / table of contents** to get chapter names."
    )

    if st.session_state.pipeline_error:
        st.error(
            f"❌ Pipeline error: {st.session_state.pipeline_error}\n\n"
            "Install with: `pip install paddleocr paddlepaddle`"
        )

    st.markdown("---")

    # ── Section A: Textbook ──────────────────
    st.subheader("📖 A. Textbook PDF (Exercise Questions)")
    col1, col2 = st.columns([2, 1])

    with col1:
        book_pdf = st.file_uploader(
            "Upload textbook PDF",
            type=["pdf"],
            key="book_pdf_upload",
            help="The model will extract exercise questions (e.g., 'Exercise 2.1')",
        )
    with col2:
        st.markdown("**Page range for exercises:**")
        book_start = st.number_input("Start page", min_value=0, value=0, key="book_start")
        book_end_raw = st.number_input("End page (0 = all)", min_value=0, value=0, key="book_end")
        book_end = None if book_end_raw == 0 else int(book_end_raw)

    # ── Section B: Book Index / ToC ──────────
    st.subheader("🗂️ B. Book Index / Table of Contents (for Chapter Names)")
    st.caption(
        "Upload the index/ToC pages so we can map chapter numbers to chapter names. "
        "You can upload the same PDF with a different page range, or a separate file."
    )
    col3, col4 = st.columns([2, 1])
    with col3:
        index_pdf = st.file_uploader(
            "Upload index PDF (or same book)",
            type=["pdf"],
            key="index_pdf_upload",
            help="Pages containing the table of contents or chapter index",
        )
    with col4:
        st.markdown("**Page range for index:**")
        idx_start = st.number_input("Start page", min_value=0, value=0, key="idx_start")
        idx_end_raw = st.number_input("End page (0 = all)", min_value=0, value=0, key="idx_end")
        idx_end = None if idx_end_raw == 0 else int(idx_end_raw)

    # ── Section C: Question Paper ────────────
    st.subheader("📋 C. Question Paper PDF")
    col5, col6 = st.columns([2, 1])
    with col5:
        paper_pdf = st.file_uploader(
            "Upload question paper PDF",
            type=["pdf"],
            key="paper_pdf_upload",
            help="The paper from which we extract numbered questions (Q1, Q2, ...)",
        )
    with col6:
        st.markdown("**Page range:**")
        paper_start = st.number_input("Start page", min_value=0, value=0, key="paper_start")
        paper_end_raw = st.number_input("End page (0 = all)", min_value=0, value=0, key="paper_end")
        paper_end = None if paper_end_raw == 0 else int(paper_end_raw)

    st.markdown("---")

    if st.button("🔍 Run OCR & Extract", key="run_ocr_btn", type="primary"):
        pipeline = st.session_state.pipeline

        if pipeline is None:
            st.error("Pipeline not initialised. Check OCR dependencies.")
        elif book_pdf is None and paper_pdf is None:
            st.warning("Please upload at least one PDF (textbook or question paper).")
        else:
            progress = st.progress(0)
            status = st.empty()

            # ── Extract index (chapter names) ──
            chapter_map = {}
            if index_pdf is not None:
                status.info("🗂️ Extracting chapter names from index…")
                tmp_idx = "temp_index.pdf"
                with open(tmp_idx, "wb") as f:
                    f.write(index_pdf.getvalue())
                try:
                    raw_index = pipeline.ocr_processor.extract_from_pdf(tmp_idx, idx_start, idx_end)
                    chapter_map = parse_chapter_index(raw_index)
                    if chapter_map:
                        st.success(f"✅ Found {len(chapter_map)} chapter names from index.")
                    else:
                        st.warning(
                            "⚠️ Could not auto-detect chapter names from index. "
                            "Exercise questions will use the exercise ID as chapter label."
                        )
                except Exception as e:
                    st.error(f"Index extraction failed: {e}")
                finally:
                    if os.path.exists(tmp_idx):
                        os.remove(tmp_idx)
            st.session_state.chapter_map = chapter_map
            progress.progress(20)

            # ── Extract exercises from textbook ──
            exercise_rows = []
            if book_pdf is not None:
                status.info("📖 Extracting exercise questions from textbook…")
                tmp_book = "temp_book.pdf"
                with open(tmp_book, "wb") as f:
                    f.write(book_pdf.getvalue())
                try:
                    raw_text = pipeline.ocr_processor.extract_from_pdf(tmp_book, book_start, book_end)
                    exercises = pipeline.ocr_processor.extract_exercise_questions(raw_text)

                    for q in exercises:
                        chapter_num = chapter_num_from_exercise_id(q.get("exercise_id", ""))
                        chapter_name = (
                            chapter_map.get(chapter_num, f"Chapter {chapter_num}")
                            if chapter_num
                            else q.get("exercise_id", "Unknown")
                        )
                        exercise_rows.append({
                            "exercise_id": q.get("exercise_id", ""),
                            "Question Text": q.get("question_text", ""),
                            "Chapter_name": chapter_name,
                        })
                    st.success(f"✅ Extracted {len(exercise_rows)} exercise questions.")
                except Exception as e:
                    st.error(f"Textbook extraction failed: {e}")
                finally:
                    if os.path.exists(tmp_book):
                        os.remove(tmp_book)

            progress.progress(60)
            if exercise_rows:
                st.session_state.exercise_df = pd.DataFrame(exercise_rows)

            # ── Extract question paper ──
            paper_rows = []
            if paper_pdf is not None:
                status.info("📋 Extracting questions from question paper…")
                tmp_paper = "temp_paper.pdf"
                with open(tmp_paper, "wb") as f:
                    f.write(paper_pdf.getvalue())
                try:
                    raw_text = pipeline.ocr_processor.extract_from_pdf(tmp_paper, paper_start, paper_end)
                    questions = pipeline.ocr_processor.extract_question_papers(raw_text)

                    for q in questions:
                        paper_rows.append({
                            "question_id": q.get("question_id", ""),
                            "Question Text": q.get("question_text", ""),
                        })
                    st.success(f"✅ Extracted {len(paper_rows)} question paper questions.")
                except Exception as e:
                    st.error(f"Question paper extraction failed: {e}")
                finally:
                    if os.path.exists(tmp_paper):
                        os.remove(tmp_paper)

            progress.progress(90)
            if paper_rows:
                st.session_state.paper_df = pd.DataFrame(paper_rows)

            progress.progress(100)
            status.success("✅ OCR extraction complete! Scroll down to review & download CSVs.")

    # ── Show results & download ──────────────
    if st.session_state.exercise_df is not None:
        st.markdown("---")
        st.subheader("📖 Extracted Exercise Questions")
        st.dataframe(st.session_state.exercise_df, use_container_width=True)

        csv_ex = st.session_state.exercise_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Exercise Questions CSV",
            data=csv_ex,
            file_name="exercise_questions.csv",
            mime="text/csv",
            key="dl_exercise",
        )
        st.caption("👆 Use this CSV in **Step 2** to train the BERT model.")

    if st.session_state.paper_df is not None:
        st.markdown("---")
        st.subheader("📋 Extracted Question Paper Questions")
        st.dataframe(st.session_state.paper_df, use_container_width=True)

        csv_paper = st.session_state.paper_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Question Paper CSV",
            data=csv_paper,
            file_name="question_paper.csv",
            mime="text/csv",
            key="dl_paper",
        )
        st.caption("👆 Use this CSV in **Step 3** to analyze chapter-wise weightage.")

    if st.session_state.chapter_map:
        st.markdown("---")
        st.subheader("🗂️ Detected Chapter Names")
        chap_df = pd.DataFrame(
            [{"Chapter #": k, "Chapter Name": v} for k, v in sorted(st.session_state.chapter_map.items())]
        )
        st.dataframe(chap_df, use_container_width=True)


# ─────────────────────────────────────────────
# STEP 2 — TRAIN MODEL
# ─────────────────────────────────────────────
elif page == "🧠 Step 2: Train Model":
    st.title("🧠 Step 2: Train the BERT Model")
    st.markdown(
        "Train the BERT classifier on the **exercise questions CSV** (which has `Question Text` "
        "and `Chapter_name` columns). The model learns to predict which chapter a question belongs to."
    )

    if not (st.session_state.exercise_df is not None or True):
        st.info("ℹ️ Complete Step 1 first to auto-populate the training data.")

    st.markdown("---")

    # Source selection
    src = st.radio(
        "Training data source",
        ["Use data from Step 1 (auto)", "Upload a CSV manually"],
        key="train_src",
    )

    train_df = None

    if src == "Use data from Step 1 (auto)":
        if st.session_state.exercise_df is not None:
            train_df = st.session_state.exercise_df
            st.success(f"✅ Using {len(train_df)} exercise questions from Step 1.")
            st.dataframe(train_df.head(), use_container_width=True)
        else:
            st.warning("⚠️ No exercise data found from Step 1. Please complete Step 1 first, or upload a CSV manually.")
    else:
        uploaded_csv = st.file_uploader(
            "Upload training CSV (must have 'Question Text' and 'Chapter_name' columns)",
            type=["csv"],
            key="manual_train_csv",
        )
        if uploaded_csv:
            train_df = pd.read_csv(uploaded_csv)
            st.write("### Preview:")
            st.dataframe(train_df.head(), use_container_width=True)

    if train_df is not None:
        # Validate columns
        required_cols = {"Question Text", "Chapter_name"}
        if not required_cols.issubset(set(train_df.columns)):
            st.error(f"❌ CSV must contain columns: {required_cols}. Found: {set(train_df.columns)}")
        else:
            # Chapter distribution
            st.markdown("**Chapter distribution in training data:**")
            chap_counts = train_df["Chapter_name"].value_counts()
            st.bar_chart(chap_counts)

            st.markdown("---")
            st.subheader("⚙️ Training Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                batch_size = st.slider("Batch Size", 4, 32, 8)
            with col2:
                num_epochs = st.slider("Epochs", 5, 20, 13)
            with col3:
                learning_rate = st.selectbox(
                    "Learning Rate",
                    [1e-5, 3e-5, 4e-5, 8e-5],
                    index=2,
                    format_func=lambda x: f"{x:.0e}",
                )

            if st.button("🚀 Start Training", key="train_btn", type="primary"):
                tmp_csv = "temp_training_data.csv"
                train_df.to_csv(tmp_csv, index=False)
                try:
                    progress = st.progress(0)
                    status = st.empty()
                    status.info("⏳ Training BERT model… this may take several minutes.")
                    progress.progress(10)

                    st.session_state.pipeline.train_bert_model(
                        tmp_csv,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate,
                    )

                    st.session_state.model_trained = True
                    progress.progress(90)

                    # Save model
                    model_dir = "data/models/bert_model"
                    st.session_state.pipeline._get_bert_analyzer().save_model(model_dir)
                    progress.progress(100)

                    status.success(f"✅ Model trained and saved to `{model_dir}`!")
                    st.info("👉 Now go to **Step 3** to analyze the question paper.")

                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                finally:
                    if os.path.exists(tmp_csv):
                        os.remove(tmp_csv)


# ─────────────────────────────────────────────
# STEP 3 — ANALYZE & RESULTS
# ─────────────────────────────────────────────
elif page == "📊 Step 3: Analyze & Results":
    st.title("📊 Step 3: Analyze Question Paper & View Results")
    st.markdown(
        "Use the trained BERT model to classify every question in the **question paper** "
        "by chapter, then see the **chapter-wise weightage**."
    )

    if not st.session_state.model_trained:
        st.warning("⚠️ The BERT model has not been trained yet. Please complete **Step 2** first.")

    st.markdown("---")

    # Source selection
    src = st.radio(
        "Question paper source",
        ["Use data from Step 1 (auto)", "Upload a CSV manually"],
        key="paper_src",
    )

    paper_df = None

    if src == "Use data from Step 1 (auto)":
        if st.session_state.paper_df is not None:
            paper_df = st.session_state.paper_df
            st.success(f"✅ Using {len(paper_df)} questions from Step 1.")
            st.dataframe(paper_df.head(), use_container_width=True)
        else:
            st.warning("⚠️ No question paper data from Step 1. Please complete Step 1 first, or upload a CSV manually.")
    else:
        uploaded_paper = st.file_uploader(
            "Upload question paper CSV (must have 'Question Text' column)",
            type=["csv"],
            key="manual_paper_csv",
        )
        if uploaded_paper:
            paper_df = pd.read_csv(uploaded_paper)
            st.dataframe(paper_df.head(), use_container_width=True)

    if paper_df is not None and "Question Text" not in paper_df.columns:
        st.error("❌ CSV must contain a 'Question Text' column.")
        paper_df = None

    if paper_df is not None:
        if st.button("🔬 Classify Questions", key="classify_btn", type="primary"):
            if not st.session_state.model_trained:
                st.error("❌ Train the model in Step 2 before classifying.")
            else:
                try:
                    status = st.empty()
                    status.info("🧠 Classifying questions with BERT…")

                    questions = paper_df["Question Text"].tolist()
                    predictions = st.session_state.pipeline._get_bert_analyzer().predict(questions)

                    result_df = paper_df.copy()
                    result_df["Predicted Chapter"] = [p["predicted_chapter"] for p in predictions]
                    result_df["Confidence"] = [p["confidence"] for p in predictions]

                    st.session_state.analysis_results = result_df
                    status.success("✅ Classification complete!")

                except Exception as e:
                    st.error(f"❌ Classification failed: {e}")

    # ── Results display ──
    if st.session_state.analysis_results is not None:
        results_df = st.session_state.analysis_results
        st.markdown("---")
        st.subheader("📋 Classified Questions")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Chapter-wise Weightage")

        weightage = results_df["Predicted Chapter"].value_counts()
        total = len(results_df)

        # Percentage table
        weightage_df = pd.DataFrame({
            "Chapter": weightage.index,
            "Questions": weightage.values,
            "Weightage (%)": (weightage.values / total * 100).round(1),
        }).reset_index(drop=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(weightage_df, use_container_width=True)
        with col2:
            st.bar_chart(weightage)

        # Summary metrics
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Questions", total)
        m2.metric("Chapters Covered", len(weightage))
        top_chapter = weightage.idxmax()
        m3.metric("Most Tested Chapter", top_chapter)

        # Download
        st.markdown("---")
        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Results CSV",
            data=csv_out,
            file_name="chapter_analysis_results.csv",
            mime="text/csv",
            key="dl_results",
        )
        csv_wt = weightage_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Chapter Weightage CSV",
            data=csv_wt,
            file_name="chapter_weightage.csv",
            mime="text/csv",
            key="dl_weightage",
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("📚 Question Analyzer · OCR → Train → Analyze")
