"""
Isolated PaddleOCR worker.

PaddleOCR and PyTorch both register native CUDA-related pybind types. Keeping
PaddleOCR in this separate process prevents those registrations from colliding
inside the Streamlit process that imports PyTorch.
"""

import argparse
import json
import os
import shutil
import traceback


PADDLE_INSTALL_FIX = """
PaddleOCR could not load Paddle's native library. This usually means the GPU
build of PaddlePaddle is installed, or both CPU and GPU Paddle packages are
present in the same Python environment.

Fix it in the same terminal/environment used to run Streamlit:

  python -m pip uninstall -y paddlepaddle-gpu paddlepaddle paddleocr paddlex
  python -m pip install paddlepaddle paddleocr

Then fully stop Streamlit and start it again.
""".strip()

PADDLEOCR_RUNTIME_FIX = """
PaddleOCR is installed, but Paddle failed during CPU inference.

Detected environment:
  paddleocr 3.5.0 / paddlex 3.5.1 / paddlepaddle 3.3.0

The failure is inside Paddle's oneDNN CPU executor:
  ConvertPirAttribute2RuntimeAttribute not support ... onednn_instruction.cc

This is a package/runtime compatibility problem, not a PDF parsing problem.
Use a clean CPU OCR stack in the same Python environment that runs Streamlit:

  python -m pip uninstall -y paddleocr paddlex paddlepaddle paddlepaddle-gpu modelscope
  python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  python -m pip install paddleocr==3.3.2

Then fully stop Streamlit and start it again.
""".strip()

TORCH_IMPORT_FIX = """
PaddleOCR/PaddleX tried to import ModelScope, which imported Torch, but your
Torch installation failed to load torch/lib/shm.dll.

Current Torch appears to be a CUDA build. Either repair Torch or remove
ModelScope from this OCR environment:

  python -m pip uninstall -y modelscope

If you need Torch for BERT later, reinstall a CPU Torch build separately.
""".strip()


def _force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLAGS_selected_gpus"] = ""
    os.environ["PADDLE_PDX_DEVICE"] = "cpu"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_mkldnn"] = "0"
    os.environ["ONEDNN_VERBOSE"] = "0"


def _create_ocr(lang):
    _force_cpu()
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        if "_gpuDeviceProperties" in str(e) or "libpaddle" in str(e):
            raise RuntimeError(PADDLE_INSTALL_FIX) from e
        if "torch\\lib\\shm.dll" in str(e) or "torch/lib/shm.dll" in str(e):
            raise RuntimeError(TORCH_IMPORT_FIX) from e
        raise

    constructors = [
        {"use_angle_cls": True, "lang": lang, "device": "cpu"},
        {"use_angle_cls": True, "lang": lang, "use_gpu": False},
        {"use_angle_cls": True, "lang": lang},
    ]

    for kwargs in constructors:
        try:
            _force_cpu()
            return PaddleOCR(**kwargs)
        except TypeError:
            continue
        except Exception as e:
            if "_gpuDeviceProperties" in str(e) or "libpaddle" in str(e):
                raise RuntimeError(PADDLE_INSTALL_FIX) from e
            if "ConvertPirAttribute2RuntimeAttribute" in str(e) or "onednn_instruction" in str(e):
                raise RuntimeError(PADDLEOCR_RUNTIME_FIX) from e
            raise

    try:
        _force_cpu()
        return PaddleOCR(use_angle_cls=True, lang=lang)
    except Exception as e:
        if "_gpuDeviceProperties" in str(e) or "libpaddle" in str(e):
            raise RuntimeError(PADDLE_INSTALL_FIX) from e
        if "ConvertPirAttribute2RuntimeAttribute" in str(e) or "onednn_instruction" in str(e):
            raise RuntimeError(PADDLEOCR_RUNTIME_FIX) from e
        raise


def _run_ocr(ocr, img):
    """Call PaddleOCR across 2.x and 3.x API variants."""
    try:
        return ocr.ocr(img)
    except TypeError:
        if hasattr(ocr, "predict"):
            return ocr.predict(img)
        raise
    except Exception as e:
        if "ConvertPirAttribute2RuntimeAttribute" in str(e) or "onednn_instruction" in str(e):
            raise RuntimeError(PADDLEOCR_RUNTIME_FIX) from e
        raise


def _run_tesseract(pil_img):
    """Optional fallback when Tesseract and pytesseract are installed."""
    if shutil.which("tesseract") is None:
        return None

    try:
        import pytesseract
    except ImportError:
        return None

    text = pytesseract.image_to_string(pil_img)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _texts_from_ocr_result(result):
    """Extract recognized text from PaddleOCR 2.x tuple output or 3.x dict output."""
    texts = []

    def visit(node):
        if node is None:
            return

        if isinstance(node, dict):
            for key in ("rec_texts", "texts"):
                value = node.get(key)
                if isinstance(value, list):
                    texts.extend(str(item).strip() for item in value if str(item).strip())
                    return

            value = node.get("text")
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
                return

            for value in node.values():
                visit(value)
            return

        if hasattr(node, "json") and callable(node.json):
            try:
                visit(node.json)
                return
            except Exception:
                pass

        if isinstance(node, tuple) and node:
            if isinstance(node[0], str) and node[0].strip():
                texts.append(node[0].strip())
                return

        if isinstance(node, list):
            if (
                len(node) >= 2
                and isinstance(node[1], tuple)
                and node[1]
                and isinstance(node[1][0], str)
            ):
                text = node[1][0].strip()
                if text:
                    texts.append(text)
                return

            for item in node:
                visit(item)

    visit(result)
    return texts


def extract_from_pdf(pdf_path, page_start=0, page_end=None, lang="en", scan_folder=None):
    # Plain-text input: read directly, no PDF tooling needed.
    if pdf_path.lower().endswith((".txt", ".md")):
        with open(pdf_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(0, page_start or 0)
        end = len(lines) if page_end is None else min(page_end, len(lines))
        return {
            "text": "".join(lines[start:end]),
            "backend": "text",
            "pages_processed": end - start,
            "total_pages": len(lines),
            "saved_images": [],
        }

    import fitz
    import numpy as np
    from PIL import Image

    extracted_text = ""
    backend = "embedded"
    pages_processed = 0
    saved_images = []

    # Setup scan folder if provided
    if scan_folder:
        os.makedirs(scan_folder, exist_ok=True)
        images_dir = os.path.join(scan_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

    try:
        pdf = fitz.open(pdf_path)
    except Exception:
        import traceback
        return {
            "text": "",
            "backend": "none",
            "pages_processed": 0,
            "total_pages": 0,
            "saved_images": saved_images,
            "error": "Could not open file as PDF: " + traceback.format_exc(limit=1),
        }

    try:
        total_pages = len(pdf)
        end_page = page_end if page_end is not None else total_pages
        start_page = max(0, page_start)
        end_page = min(end_page, total_pages)
    finally:
        pass

    if start_page >= end_page:
        pdf.close()
        return {
            "text": "",
            "backend": "none",
            "pages_processed": 0,
            "total_pages": total_pages,
            "saved_images": saved_images,
        }

    try:
        for pg_num in range(start_page, end_page):
            page_text = pdf[pg_num].get_text("text").strip()
            if page_text:
                extracted_text += f"\n--- Page {pg_num + 1} ---\n{page_text}\n"
                pages_processed += 1

                if scan_folder:
                    mat = fitz.Matrix(1.5, 1.5)
                    pm = pdf[pg_num].get_pixmap(matrix=mat, alpha=False)
                    img_path = os.path.join(images_dir, f"page_{pg_num + 1:04d}.png")
                    pm.save(img_path)
                    saved_images.append(img_path)

        if extracted_text.strip():
            print("[OCR] Used embedded PDF text; PaddleOCR was not needed.")
            if scan_folder:
                text_file = os.path.join(scan_folder, "raw_text.txt")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
            return {
                "text": extracted_text,
                "backend": backend,
                "pages_processed": pages_processed,
                "total_pages": total_pages,
                "saved_images": saved_images,
            }

        # No embedded text: try OCR engines, but fail clearly if none installed.
        print("[OCR] No embedded text; trying OCR engines...")
        ocr = None
        try:
            ocr = _create_ocr(lang)
            backend = "paddleocr"
            print("[OCR] PaddleOCR initialized successfully in worker.")
        except Exception as e:
            if shutil.which("tesseract") is not None:
                backend = "tesseract"
                print("[OCR] PaddleOCR unavailable; falling back to Tesseract.")
            else:
                raise RuntimeError(
                    "This PDF has no embedded text layer and no OCR engine is installed. "
                    "Install OCR with: pip install -r requirements-ocr.txt "
                    "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/ "
                    f"(original engine error: {e})"
                ) from e

        for pg_num in range(start_page, end_page):
            page = pdf[pg_num]

            mat = fitz.Matrix(1.5, 1.5)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            pil_img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

            if ocr is None:
                page_texts = _run_tesseract(pil_img) or []
            else:
                img = np.array(pil_img)
                try:
                    page_texts = _texts_from_ocr_result(_run_ocr(ocr, img))
                except Exception:
                    fallback_texts = _run_tesseract(pil_img)
                    if fallback_texts is None:
                        raise
                    backend = "tesseract"
                    page_texts = fallback_texts

            extracted_text += f"\n--- Page {pg_num + 1} ---\n"
            for text in page_texts:
                extracted_text += text + "\n"
            pages_processed += 1

            if scan_folder:
                img_path = os.path.join(images_dir, f"page_{pg_num + 1:04d}.png")
                pil_img.save(img_path)
                saved_images.append(img_path)
    finally:
        pdf.close()

    if scan_folder and extracted_text.strip():
        text_file = os.path.join(scan_folder, "raw_text.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

    return {
        "text": extracted_text,
        "backend": backend,
        "pages_processed": pages_processed,
        "total_pages": total_pages,
        "saved_images": saved_images,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scan-folder", default=None, help="Optional: folder to save page images and text")
    args = parser.parse_args()

    _force_cpu()
    try:
        result = extract_from_pdf(args.pdf, args.start, args.end, args.lang, args.scan_folder)
        payload = {"ok": True, **result}
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
