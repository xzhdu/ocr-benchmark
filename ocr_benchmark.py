import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Type

try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


@dataclass
class BenchmarkResult:
    """Data class storing OCR processing results and performance metrics."""

    engine_name: str
    image_path: str
    text: str
    execution_time_sec: float
    peak_rss_mb: float
    manual_evaluation: float


class BaseOCREngine(ABC):
    """Abstract base class for OCR engine implementations."""

    name: str = "BaseEngine"

    @abstractmethod
    def process_image(self, image_path: str) -> str:
        """Extract text from the given image path.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            str: Recognized text.
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

    def process_image(self, image_path: str) -> str:
        """Extract text from an image file using RapidOCR.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Extracted text separated by lines.
        """
        result, _ = self.engine(image_path)
        if not result:
            return ""
        return "\n".join([line[1] for line in result])


ENGINE_REGISTRY: Dict[str, Type[BaseOCREngine]] = {
    "rapidocr": RapidOCREngine,
}


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


def process_benchmark(engine: BaseOCREngine, image_paths: List[str]) -> List[BenchmarkResult]:
    """Execute OCR on images and record processing metrics.

    Args:
        engine (BaseOCREngine): OCR engine instance.
        image_paths (List[str]): List of image file paths to process.

    Returns:
        List[BenchmarkResult]: List of benchmark results for each image.
    """
    results = []
    for path in image_paths:
        print(f"\nProcessing: {path}")
        start_time = time.perf_counter()
        text = engine.process_image(path)
        elapsed_time = time.perf_counter() - start_time
        peak_rss = get_peak_rss_mb()

        print(f"Time: {elapsed_time:.4f}s | Peak RSS: {peak_rss:.2f} MB")
        print("--- Recognized Text ---")
        print(text if text else "[No text detected]")
        print("-" * 40)

        score = prompt_manual_evaluation()

        results.append(
            BenchmarkResult(
                engine_name=engine.name,
                image_path=path,
                text=text,
                execution_time_sec=elapsed_time,
                peak_rss_mb=peak_rss,
                manual_evaluation=score,
            )
        )
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
        "-o",
        "--output",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    image_paths = get_image_paths(args.input)
    engine_class = ENGINE_REGISTRY[args.engine]
    engine = engine_class()

    results = process_benchmark(engine, image_paths)

    if args.output:
        save_results_to_json(results, args.output)
        print(f"\nResults successfully saved to {args.output}")


if __name__ == "__main__":
    main()
