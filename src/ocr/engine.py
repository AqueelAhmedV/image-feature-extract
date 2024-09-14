import os
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import logging

class OCREngine(ABC):
    def __init__(self, images_base: Union[str, Path] = None):
        self.images_base = Path(images_base).resolve() if images_base else None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def extract_text(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from the given image.

        Args:
            image_path (Union[str, Path]): Path to the image file.

        Returns:
            str: Extracted text from the image.

        Raises:
            FileNotFoundError: If the image file is not found.
            Exception: For any other errors during text extraction.
        """
        pass

    def safe_extract_text(self, image_path: Union[str, Path]) -> str:
        """
        Safely extract text from the given image, catching and logging exceptions.

        Args:
            image_path (Union[str, Path]): Path to the image file.

        Returns:
            str: Extracted text from the image, or an error message if extraction fails.
        """
        try:
            full_path = self._get_full_path(image_path)
            self.logger.info(f"Attempting to extract text from: {full_path}")
            return self.extract_text(full_path)
        except FileNotFoundError:
            error_msg = f"Image file not found: {full_path}"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error extracting text from {full_path}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def _get_full_path(self, image_path: Union[str, Path]) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.images_base:
            return (self.images_base / path).resolve()
        return Path(os.getcwd()) / path
