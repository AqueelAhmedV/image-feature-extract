import os
import pandas as pd
from tqdm import tqdm
from src.ocr.paddle_ocr import PaddleOCREngine
from src.ocr.easy_ocr import EasyOCREngine

def process_ocr_data(input_csv, output_csv, images_folder):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Initialize OCR engines
    paddle_ocr = PaddleOCREngine(images_folder)
    easy_ocr = EasyOCREngine(images_folder)
    
    # Create new columns for OCR results
    df['easy_ocr'] = ''
    df['paddle_ocr'] = ''
    
    # Process each image
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        image_file = os.path.basename(row['image_link'])
        image_path = os.path.join(images_folder, image_file)
        
        # Run EasyOCR
        df.at[index, 'easy_ocr'] = easy_ocr.safe_extract_text(image_path)
        
        # Run PaddleOCR
        df.at[index, 'paddle_ocr'] = paddle_ocr.safe_extract_text(image_path)
    
    # Reorder columns
    columns_order = ['image_link', 'easy_ocr', 'paddle_ocr'] + [col for col in df.columns if col not in ['image_link', 'easy_ocr', 'paddle_ocr']]
    df = df[columns_order]
    
    # Save the results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Paths
train_input = './dataset/ocr_train.csv'
train_output = './dataset/ocr_train_text.csv'
test_input = './dataset/ocr_test.csv'
test_output = './dataset/ocr_test_text.csv'
images_folder = os.path.abspath('./train_images')

# Process train data
print("Processing train data...")
process_ocr_data(train_input, train_output, images_folder)

# Process test data
print("\nProcessing test data...")
process_ocr_data(test_input, test_output, images_folder)

print("Processing complete.")
