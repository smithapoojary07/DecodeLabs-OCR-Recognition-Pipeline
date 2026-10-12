# DecodeLabs Artificial Intelligence Project 4

## Basic Image/Text Recognition (OCR)

This project implements the OCR execution path described in the Project 4 brief. It uses **pytesseract** (Tesseract OCR) with **OpenCV** preprocessing.

### Pipeline
1. Read a raw image.
2. Convert it to grayscale.
3. Apply Gaussian blur to reduce noise.
4. Deskew the text when a rotation is detected.
5. Apply adaptive thresholding for strong foreground/background separation.
6. Run Tesseract OCR with Page Segmentation Mode 6.
7. Extract per-word confidence scores.
8. Apply the project's **80% confidence gate**.
9. Produce an annotated image, recognized text, and JSON validation report.

### Project validation
The script records four checks:
- Library integration: pytesseract + OpenCV
- Pre-processing integrity: grayscale + adaptive thresholding
- Accuracy benchmark: mean confidence >= 80% and every detected word >= 80%
- Visual confirmation: annotated bounding boxes and confidence labels

## Run

Install Tesseract OCR on the machine and install Python dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python ocr_pipeline.py --input input/sample_input.png --output-dir output
```

A successful run prints `Status: PASS` and creates:

- `output/01_grayscale.png`
- `output/02_blurred.png`
- `output/03_adaptive_threshold.png`
- `output/04_annotated_output.png`
- `output/recognized_text.txt`
- `output/result.json`

### Notes
The confidence score is Tesseract's OCR confidence for detected words. The 80% threshold is used as the project validation gate; it does not claim that OCR is mathematically 80% accurate on all possible images.
