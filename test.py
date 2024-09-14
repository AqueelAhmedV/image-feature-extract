import os
import random
import pandas as pd
from src.ocr.paddle_ocr import PaddleOCREngine
from src.ocr.easy_ocr import EasyOCREngine

def process_random_images(csv_path, image_folder, num_images=10):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get a list of image files from the DataFrame, using basename + .jpg
    image_files = [os.path.basename(link) for link in df['image_link']]
    
    # Randomly select 10 images
    selected_images = random.sample(image_files, num_images)
    
    # Initialize OCR engines
    paddle_ocr = PaddleOCREngine(image_folder)
    easy_ocr = EasyOCREngine(image_folder)
    
    for image_file in selected_images:
        image_path = os.path.join(image_folder, image_file)
        print(f"\nProcessing file: {image_file}")
        
        print("\nPaddleOCR results:")
        paddle_text = paddle_ocr.safe_extract_text(image_path)
        print(paddle_text)
        
        print("\nEasyOCR results:")
        easy_text = easy_ocr.safe_extract_text(image_path)
        print(easy_text)

# Usage
csv_path = './dataset/train.csv'
images_folder = os.path.abspath(r'.\train_images')
process_random_images(csv_path, images_folder)
