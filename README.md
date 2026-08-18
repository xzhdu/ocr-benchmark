# OCR Benchmark

A lightweight script that allows you to benchmark and compare different OCR engines (`RapidOCR`, `Tesseract`, `docTR`) on the same files.

## Features

- **Multi-Engine Support**: Easily switch between `RapidOCR`, `Tesseract`, and `docTR`.
- **Bounding Box Visualization**: Displays OCR detection bounding boxes and recognized text on a graphical window (using Matplotlib) to help evaluate quality.
- **Metric Evaluation**:
  - Automatically calculates **Word Error Rate (WER)** and **Character Error Rate (CER)** if a ground truth file is provided.
  - Measures execution time (seconds) and peak memory usage (RSS in MB).
- **Manual Scoring**: Prompts the user to input a manual score (1.0 to 5.0) for each analyzed document.
- **Batch Saving**: Save annotated images with drawn bounding boxes to a directory, and export all results to a JSON file.

## Prerequisites

Install the required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*(Note: Using Tesseract requires the `tesseract` binary to be installed on your system).*

## Usage

Run the script by specifying the input path (an image file or a directory of images) and the OCR engine.

### Basic Run
```bash
python ocr_benchmark.py -i /path/to/images -e rapidocr
```

### With Ground Truth & Output JSON
To compute error rates and save the benchmark results:
```bash
python ocr_benchmark.py -i /path/to/images -e doctr -g ground_truth.json -o results.json
```

### Headless Mode & Saving Visualizations
To run without opening GUI windows and save the annotated bounding box images:
```bash
python ocr_benchmark.py -i /path/to/images -e tesseract --no-display --save-visual-dir ./output_visuals
```

## CLI Options

| Argument | Description |
|---|---|
| `-i`, `--input` | **(Required)** Path to input image file or directory containing images. |
| `-e`, `--engine` | OCR engine to use: `rapidocr` (default), `tesseract`, or `doctr`. |
| `-g`, `--ground-truth` | Path to a ground truth JSON file mapping filenames to expected reference text. |
| `-o`, `--output` | Path to save the benchmark metrics in JSON format. |
| `--no-display` | Disable displaying annotated bounding boxes in a GUI window. |
| `--save-visual-dir` | Directory to save annotated images with bounding boxes. |
