"""
Isolated PaddleOCR worker.

PaddleOCR and PyTorch both register native CUDA-related pybind types. Keeping
PaddleOCR in this separate process prevents those registrations from colliding
inside the Streamlit process that imports PyTorch.
"""

import argparse
import json
import os
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


def extract_from_pdf(pdf_path, page_start=0, page_end=None, lang="en"):
    import fitz
    import numpy as np
    from PIL import Image

    extracted_text = ""
    backend = "embedded"
    pages_processed = 0

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)
        end_page = page_end if page_end is not None else total_pages
        start_page = max(0, page_start)
        end_page = min(end_page, total_pages)

        if start_page >= end_page:
            return {
                "text": "",
                "backend": "none",
                "pages_processed": 0,
                "total_pages": total_pages,
            }

        for pg_num in range(start_page, end_page):
            page_text = pdf[pg_num].get_text("text").strip()
            if page_text:
                extracted_text += f"\n--- Page {pg_num + 1} ---\n{page_text}\n"
                pages_processed += 1

        if extracted_text.strip():
            print("[OCR] Used embedded PDF text; PaddleOCR was not needed.")
            return {
                "text": extracted_text,
                "backend": backend,
                "pages_processed": pages_processed,
                "total_pages": total_pages,
            }

        print("[OCR] Initializing PaddleOCR on CPU... (worker first use may take a moment)")
        ocr = _create_ocr(lang)
        backend = "paddleocr"
        print("[OCR] PaddleOCR initialized successfully in worker.")

        for pg_num in range(start_page, end_page):
            page = pdf[pg_num]

            mat = fitz.Matrix(1.5, 1.5)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            img = np.array(Image.frombytes("RGB", [pm.width, pm.height], pm.samples))

            result = _run_ocr(ocr, img)

            extracted_text += f"\n--- Page {pg_num + 1} ---\n"
            for text in _texts_from_ocr_result(result):
                extracted_text += text + "\n"
            pages_processed += 1

    return {
        "text": extracted_text,
        "backend": backend,
        "pages_processed": pages_processed,
        "total_pages": total_pages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _force_cpu()
    try:
        result = extract_from_pdf(args.pdf, args.start, args.end, args.lang)
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
