from pathlib import Path
from typing import Union

# import paddle
from src.ocr.engine import OCREngine
from paddleocr import PaddleOCR


class PaddleOCREngine(OCREngine):
    def __init__(self, images_base: str):
        super().__init__(images_base)  # This sets up the logger
        self.ocr = PaddleOCR(use_angle_cls=True)

    def extract_text(self, image_path: Union[str, Path]) -> str:
        if self.images_base:
            full_path = self.images_base / image_path
        else:
            full_path = Path(image_path)

        result = self.ocr.ocr(str(full_path), cls=True)
        extracted_text = '\n'.join([line[1][0] for line in result[0]])
        return extracted_text
