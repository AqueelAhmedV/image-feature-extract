from pathlib import Path
from typing import Union

import easyocr
from src.ocr.engine import OCREngine

class EasyOCREngine(OCREngine):
    def __init__(self, images_base: str):
        super().__init__(images_base)
        self.reader = easyocr.Reader(['en'])  # Initialize for English language

    def extract_text(self, image_path: Union[str, Path]) -> str:
        if self.images_base:
            full_path = self.images_base / image_path
        else:
            full_path = Path(image_path)

        result = self.reader.readtext(str(full_path))
        extracted_text = '\n'.join([text for _, text, _ in result])
        return extracted_text