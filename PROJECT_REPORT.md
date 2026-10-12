# Project 4 Submission Report

## Title
**Basic Image/Text Recognition using OCR**

## Objective
Build a basic recognition pipeline that ingests visual data and extracts machine-readable text using an available pre-trained/library-based recognition system.

## Chosen approach
The OCR path was selected using **pytesseract** with **OpenCV**. Tesseract provides the recognition engine while OpenCV performs the required image preprocessing.

## Implementation
The pipeline performs:

1. **Grayscale conversion** – reduces the RGB image to a single intensity channel.
2. **Gaussian blur** – smooths small artifacts and noise.
3. **Deskewing** – estimates and corrects text rotation when necessary.
4. **Adaptive thresholding** – creates a high-contrast binary image despite uneven lighting.
5. **OCR** – Tesseract reads the processed image using PSM 6 for a uniform text block.
6. **Confidence filtering** – the script checks Tesseract's confidence values against an 80% threshold.
7. **Visual confirmation** – detected words are drawn with bounding boxes and confidence labels.

## Validation result
For the included sample image:

- Mean OCR confidence: **94.36%**
- Required minimum: **80%**
- Validation status: **PASS**
- OCR output: `DecodeLabs Artificial Intelligence Project 4 - Basic OCR Recognition Machine perception turns visual data into structured, machine-readable intelligence. CONFIDENCE TARGET: 80% OCR TEST SAMPLE 2026`

## Deliverables
- `ocr_pipeline.py` – complete recognition pipeline
- `input/sample_input.png` – sample input image
- `output/04_annotated_output.png` – visual confirmation with OCR boxes and confidence
- `output/recognized_text.txt` – extracted text
- `output/result.json` – machine-readable validation details
- `output/01_grayscale.png` – preprocessing stage
- `output/02_blurred.png` – preprocessing stage
- `output/03_adaptive_threshold.png` – preprocessing stage
- `requirements.txt` – Python dependencies
- `README.md` – setup and usage instructions

## Conclusion
The submitted pipeline demonstrates library integration, image preprocessing, confidence-based validation, and visual confirmation for basic OCR recognition.
