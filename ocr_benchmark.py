import argparse
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

    image_path: str
    text: str
    execution_time_sec: float
    peak_rss_mb: float


class BaseOCREngine(ABC):
    """Abstract base class for OCR engine implementations."""

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
        start_time = time.perf_counter()
        text = engine.process_image(path)
        elapsed_time = time.perf_counter() - start_time
        peak_rss = get_peak_rss_mb()

        results.append(
            BenchmarkResult(
                image_path=path,
                text=text,
                execution_time_sec=elapsed_time,
                peak_rss_mb=peak_rss,
            )
        )
    return results


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
    args = parser.parse_args()

    image_paths = get_image_paths(args.input)
    engine_class = ENGINE_REGISTRY[args.engine]
    engine = engine_class()

    results = process_benchmark(engine, image_paths)

    for res in results:
        print(f"File: {res.image_path}")
        print(
            f"Time: {res.execution_time_sec:.4f}s | Peak RSS: {res.peak_rss_mb:.2f} MB")
        print("--- Recognized Text ---")
        print(res.text)
        print("-" * 40)


if __name__ == "__main__":
    main()
