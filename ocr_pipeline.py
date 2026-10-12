#!/usr/bin/env python3
"""DecodeLabs AI Project 4 - Basic OCR Recognition Pipeline.

Implements the OCR path from the project brief using pytesseract + OpenCV:
1. grayscale conversion
2. Gaussian blur
3. adaptive thresholding
4. OCR with confidence extraction
5. 80% confidence gate
6. annotated visual output + text/JSON results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import pytesseract
from pytesseract import Output


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return gray, blur, binary


def deskew(binary):
    coords = cv2.findNonZero(255 - binary)
    if coords is None:
        return binary, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.2:
        return binary, float(angle)
    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, float(angle)


def run_ocr(binary, psm: int = 6) -> tuple[str, list[dict[str, Any]], float]:
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(binary, config=config, output_type=Output.DICT)
    words = []
    for i, raw_conf in enumerate(data["conf"]):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = -1.0
        text = data["text"][i].strip()
        if not text or conf < 0:
            continue
        words.append(
            {
                "text": text,
                "confidence": round(conf, 2),
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
            }
        )
    recognized = " ".join(w["text"] for w in words)
    mean_conf = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    return recognized, words, mean_conf


def annotate(image, words, threshold=80.0):
    output = image.copy()
    for w in words:
        x, y, ww, hh = w["x"], w["y"], w["w"], w["h"]
        passed = w["confidence"] >= threshold
        color = (0, 180, 0) if passed else (0, 0, 255)
        cv2.rectangle(output, (x, y), (x + ww, y + hh), color, 2)
        label = f"{w['text']} {w['confidence']:.1f}%"
        cv2.putText(output, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output


def main():
    parser = argparse.ArgumentParser(description="DecodeLabs Project 4 OCR pipeline")
    parser.add_argument("--input", default="input/sample_input.png")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=80.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))
    if image is None:
        raise SystemExit(f"Could not read input image: {input_path}")

    gray, blur, binary = preprocess(image)
    binary, angle = deskew(binary)

    cv2.imwrite(str(out_dir / "01_grayscale.png"), gray)
    cv2.imwrite(str(out_dir / "02_blurred.png"), blur)
    cv2.imwrite(str(out_dir / "03_adaptive_threshold.png"), binary)

    text, words, mean_conf = run_ocr(binary, args.psm)
    passed_words = [w for w in words if w["confidence"] >= args.threshold]
    validation_passed = bool(words) and mean_conf >= args.threshold and len(passed_words) == len(words)

    annotated = annotate(image, words, args.threshold)
    status = "PASS" if validation_passed else "FAIL"
    cv2.putText(annotated, f"Project 4 OCR: {status} | mean confidence {mean_conf:.1f}%", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0) if validation_passed else (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "04_annotated_output.png"), annotated)

    result = {
        "project": "DecodeLabs Artificial Intelligence Project 4",
        "task": "Basic Image/Text Recognition (OCR)",
        "library": "pytesseract + OpenCV",
        "preprocessing": ["grayscale", "Gaussian blur", "deskew", "adaptive thresholding"],
        "psm": args.psm,
        "confidence_threshold_percent": args.threshold,
        "mean_confidence_percent": round(mean_conf, 2),
        "recognized_text": text,
        "word_count": len(words),
        "validation": {
            "library_integration": True,
            "preprocessing_integrity": True,
            "accuracy_benchmarking": validation_passed,
            "visual_confirmation": True,
            "overall": status,
        },
        "deskew_angle_degrees": round(angle, 3),
        "words": words,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "recognized_text.txt").write_text(text + "\n", encoding="utf-8")

    print(f"Status: {status}")
    print(f"Mean confidence: {mean_conf:.2f}%")
    print(f"Recognized text: {text}")
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
