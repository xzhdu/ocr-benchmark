import argparse
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Type

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


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

    for path in image_paths:
        print(f"Processing: {path}")
        text = engine.process_image(path)
        print("--- Recognized Text ---")
        print(text)
        print("-" * 24)


if __name__ == "__main__":
    main()
