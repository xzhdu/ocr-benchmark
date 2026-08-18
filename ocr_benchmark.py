import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

try:
    import resource
except ImportError:
    resource = None

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except ImportError:
    DocumentFile = None
    ocr_predictor = None


@dataclass
class OCRBox:
    """Bounding box representation with polygon points and recognized text."""

    box: List[List[int]]
    text: str = ""
    confidence: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Data class storing OCR processing results and performance metrics."""

    engine_name: str
    image_path: str
    text: str
    execution_time_sec: float
    peak_rss_mb: float
    manual_evaluation: float
    wer: Optional[float] = None
    cer: Optional[float] = None


class BaseOCREngine(ABC):
    """Abstract base class for OCR engine implementations."""

    name: str = "BaseEngine"

    @abstractmethod
    def process_image(self, image_path: str) -> Tuple[str, List[OCRBox]]:
        """Extract text and bounding boxes from the given image path.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            Tuple[str, List[OCRBox]]: Recognized text and list of bounding boxes.
        """
        pass


class RapidOCREngine(BaseOCREngine):
    """OCR engine implementation using RapidOCR."""

    name: str = "RapidOCR"

    def __init__(self):
        if RapidOCR is None:
            raise ImportError(
                "rapidocr_onnxruntime is not installed. Install it via 'pip install rapidocr_onnxruntime'."
            )
        self.engine = RapidOCR()

    def process_image(self, image_path: str) -> Tuple[str, List[OCRBox]]:
        """Extract text from an image file using RapidOCR.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Tuple[str, List[OCRBox]]: Extracted text and bounding boxes.
        """
        result, _ = self.engine(image_path)
        if not result:
            return "", []

        extracted_lines = []
        boxes = []
        for item in result:
            box_coords = [[int(round(pt[0])), int(round(pt[1]))] for pt in item[0]]
            line_text = str(item[1])
            score = float(item[2]) if len(item) > 2 else None
            extracted_lines.append(line_text)
            boxes.append(OCRBox(box=box_coords, text=line_text, confidence=score))

        return "\n".join(extracted_lines), boxes


class TesseractEngine(BaseOCREngine):
    """OCR engine implementation using Tesseract."""

    name: str = "Tesseract"

    def __init__(self):
        if pytesseract is None or Image is None:
            raise ImportError(
                "pytesseract or Pillow is not installed. "
                "Install them via 'pip install pytesseract pillow'."
            )

    def process_image(self, image_path: str) -> Tuple[str, List[OCRBox]]:
        """Extract text from an image file using Tesseract OCR.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Tuple[str, List[OCRBox]]: Extracted text and bounding boxes.
        """
        with Image.open(image_path) as img:
            full_text = pytesseract.image_to_string(img).strip()
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        boxes = []
        num_boxes = len(data.get("text", []))
        for i in range(num_boxes):
            word_text = data["text"][i].strip()
            if not word_text:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            conf_val = None
            if "conf" in data and str(data["conf"][i]) != "-1":
                try:
                    conf_val = float(data["conf"][i]) / 100.0
                except (ValueError, TypeError):
                    conf_val = None

            poly_box = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
            boxes.append(OCRBox(box=poly_box, text=word_text, confidence=conf_val))

        return full_text, boxes


class DocTREngine(BaseOCREngine):
    """OCR engine implementation using docTR."""

    name: str = "docTR"

    def __init__(self):
        if ocr_predictor is None or DocumentFile is None:
            raise ImportError(
                "python-doctr is not installed. "
                "Install it via 'pip install python-doctr'."
            )
        self.model = ocr_predictor(pretrained=True)

    def process_image(self, image_path: str) -> Tuple[str, List[OCRBox]]:
        """Extract text from an image file using docTR.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Tuple[str, List[OCRBox]]: Extracted text and bounding boxes.
        """
        doc = DocumentFile.from_images(image_path)
        result = self.model(doc)
        export = result.export()

        extracted_lines = []
        boxes = []
        for page in export.get("pages", []):
            page_dims = page.get("dimensions", (1, 1))
            page_h, page_w = page_dims[0], page_dims[1]

            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    line_words = [
                        word["value"] for word in line.get("words", [])
                    ]
                    extracted_lines.append(" ".join(line_words))

                    for word in line.get("words", []):
                        val = word.get("value", "")
                        geom = word.get("geometry")
                        conf = word.get("confidence")
                        if geom:
                            (xmin, ymin), (xmax, ymax) = geom
                            x1 = int(round(float(xmin) * page_w))
                            y1 = int(round(float(ymin) * page_h))
                            x2 = int(round(float(xmax) * page_w))
                            y2 = int(round(float(ymax) * page_h))
                            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            boxes.append(OCRBox(box=poly, text=val, confidence=float(conf) if conf is not None else None))

        return "\n".join(extracted_lines), boxes


ENGINE_REGISTRY: Dict[str, Type[BaseOCREngine]] = {
    "rapidocr": RapidOCREngine,
    "tesseract": TesseractEngine,
    "doctr": DocTREngine,
}


def compute_levenshtein_distance(seq1: Union[str, Sequence[str]], seq2: Union[str, Sequence[str]]) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Args:
        seq1 (Union[str, Sequence[str]]): Source sequence.
        seq2 (Union[str, Sequence[str]]): Target sequence.

    Returns:
        int: Edit distance value.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    return dp[m][n]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER).

    Args:
        reference (str): Ground truth reference text.
        hypothesis (str): Recognized OCR text.

    Returns:
        float: Character Error Rate ratio.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    distance = compute_levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER).

    Args:
        reference (str): Ground truth reference text.
        hypothesis (str): Recognized OCR text.

    Returns:
        float: Word Error Rate ratio.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = compute_levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def get_peak_rss_mb() -> float:
    """Get the process peak Resident Set Size (RSS) memory in Megabytes.

    Returns:
        float: Peak RSS memory in MB.
    """

    # Linux and MacOS
    if resource is not None:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024

    # Windows
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        peak_bytes = getattr(mem_info, "peak_wset", mem_info.rss)
        return peak_bytes / (1024 * 1024)

    return 0.0


def get_image_paths(input_path: str) -> List[str]:
    """Collect valid image paths from a file or directory.

    Args:
        input_path (str): Path to an image file or directory containing images.

    Returns:
        List[str]: List of resolved image file paths.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        image_paths = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    raise ValueError(f"Invalid input path: {input_path}")


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """Load ground truth JSON dictionary mapping filenames to reference text.

    Args:
        gt_path (str): Path to the ground truth JSON file.

    Returns:
        Dict[str, str]: Mapping from image filename to ground truth text.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_manual_evaluation() -> float:
    """Prompt user for manual evaluation score between 1.0 and 5.0.

    Returns:
        float: Validated user evaluation score.
    """
    while True:
        try:
            user_input = input("Enter manual evaluation score (1.0 - 5.0): ")
            score = float(user_input)
            if 1.0 <= score <= 5.0:
                return score
            print("Score must be between 1.0 and 5.0.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def draw_bounding_boxes(
    image_path: str,
    boxes: List[OCRBox],
    draw_labels: bool = True,
) -> Optional["np.ndarray"]:
    """Draw bounding boxes and optional labels on an image.

    Args:
        image_path (str): Path to the image file.
        boxes (List[OCRBox]): Detected bounding boxes.
        draw_labels (bool): Whether to draw text labels.

    Returns:
        Optional[np.ndarray]: Annotated image or None if loading failed.
    """
    if cv2 is None or np is None:
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    annotated = img.copy()
    for item in boxes:
        pts = np.array(item.box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        if draw_labels and item.text:
            label = item.text
            if item.confidence is not None:
                label += f" ({item.confidence:.2f})"

            x, y = item.box[0][0], item.box[0][1]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                annotated,
                (x, max(0, y - h - 4)),
                (x + w, max(0, y)),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x, max(0, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated


def display_bounding_boxes(
    window_title: str,
    annotated_image: "np.ndarray",
) -> Optional[object]:
    """Display annotated image in a window using matplotlib.

    Args:
        window_title (str): Title of the window.
        annotated_image (np.ndarray): Image array with bounding boxes (BGR format from cv2).

    Returns:
        Optional[object]: Matplotlib figure object if displayed, None otherwise.
    """
    if plt is None or annotated_image is None:
        return None

    try:
        if cv2 is not None:
            rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = annotated_image

        plt.ion()
        fig = plt.figure(num=window_title, figsize=(10, 8))
        plt.clf()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_image)
        ax.axis("off")
        ax.set_title(window_title)
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)
        return fig
    except Exception as e:
        print(f"Warning: Unable to display bounding boxes window: {e}")
        return None


def close_bounding_boxes(fig: Optional[object]) -> None:
    """Close the active matplotlib window.

    Args:
        fig (Optional[object]): Matplotlib figure object to close.
    """
    if plt is not None and fig is not None:
        try:
            plt.close(fig)
            plt.pause(0.001)
        except Exception:
            pass


def process_benchmark(
    engine: BaseOCREngine,
    image_paths: List[str],
    ground_truth: Optional[Dict[str, str]] = None,
    show_boxes: bool = True,
    save_visual_dir: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Execute OCR on images and record processing metrics.

    Args:
        engine (BaseOCREngine): OCR engine instance.
        image_paths (List[str]): List of image file paths to process.
        ground_truth (Optional[Dict[str, str]]): Ground truth dictionary.
        show_boxes (bool): Whether to display bounding boxes in a window.
        save_visual_dir (Optional[str]): Directory to save visual output images.

    Returns:
        List[BenchmarkResult]: List of benchmark results for each image.
    """
    results = []
    if save_visual_dir:
        os.makedirs(save_visual_dir, exist_ok=True)

    for path in image_paths:
        file_name = os.path.basename(path)
        print(f"\nProcessing: {path}")

        start_time = time.perf_counter()
        text, boxes = engine.process_image(path)
        elapsed_time = time.perf_counter() - start_time
        peak_rss = get_peak_rss_mb()

        wer, cer = None, None
        if ground_truth and file_name in ground_truth:
            ref_text = ground_truth[file_name]
            wer = calculate_wer(ref_text, text)
            cer = calculate_cer(ref_text, text)
            print(f"WER: {wer:.4f} | CER: {cer:.4f}")

        print(
            f"Time: {elapsed_time:.4f}s | Peak RSS: {peak_rss:.2f} MB | Detected boxes: {len(boxes)}"
        )
        print("--- Recognized Text ---")
        print(text if text else "[No text detected]")
        print("-" * 40)

        active_fig = None
        window_title = f"OCR [{engine.name}] - {file_name}"
        annotated_img = None

        if show_boxes or save_visual_dir:
            annotated_img = draw_bounding_boxes(path, boxes)

        if save_visual_dir and annotated_img is not None and cv2 is not None:
            out_img_path = os.path.join(save_visual_dir, f"bbox_{file_name}")
            cv2.imwrite(out_img_path, annotated_img)
            print(f"Saved visualization to: {out_img_path}")

        if show_boxes and annotated_img is not None:
            active_fig = display_bounding_boxes(window_title, annotated_img)

        score = prompt_manual_evaluation()

        if active_fig is not None:
            close_bounding_boxes(active_fig)

        results.append(
            BenchmarkResult(
                engine_name=engine.name,
                image_path=file_name,
                text=text,
                execution_time_sec=elapsed_time,
                peak_rss_mb=peak_rss,
                manual_evaluation=score,
                wer=wer,
                cer=cer,
            )
        )

    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass

    return results


def save_results_to_json(results: List[BenchmarkResult], output_path: str) -> None:
    """Save benchmark results to a JSON file matching the OCR schema format.

    Args:
        results (List[BenchmarkResult]): List of benchmark result objects.
        output_path (str): Path to the output JSON file.
    """
    existing_data = []

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded_json = json.loads(content)
                    if isinstance(loaded_json, list):
                        existing_data = loaded_json
        except (json.JSONDecodeError, OSError):
            existing_data = []

    for res in results:
        entry = {
            "engine_name": res.engine_name,
            "file_name": res.image_path,
            "recognized_text": res.text,
            "time_seconds": round(res.execution_time_sec, 4),
            "memory_usage_mb": round(res.peak_rss_mb, 2),
            "wer": round(res.wer, 4) if res.wer is not None else None,
            "cer": round(res.cer, 4) if res.cer is not None else None,
            "manual_evaluation": res.manual_evaluation,
        }
        existing_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight script that allows to benchmark different OCR engines on the same files.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input image file or directory",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=list(ENGINE_REGISTRY.keys()),
        default="rapidocr",
        help="OCR engine to use",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable displaying bounding boxes in a window",
    )
    parser.add_argument(
        "--save-visual-dir",
        help="Directory to save annotated images with bounding boxes",
    )
    args = parser.parse_args()

    image_paths = get_image_paths(args.input)
    engine_class = ENGINE_REGISTRY[args.engine]
    engine = engine_class()

    ground_truth_map = None
    if args.ground_truth:
        ground_truth_map = load_ground_truth(args.ground_truth)

    show_boxes = not args.no_display
    results = process_benchmark(
        engine,
        image_paths,
        ground_truth_map,
        show_boxes=show_boxes,
        save_visual_dir=args.save_visual_dir,
    )

    if args.output:
        save_results_to_json(results, args.output)
        print(f"\nResults successfully saved to {args.output}")


if __name__ == "__main__":
    main()

