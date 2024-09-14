import os
import shutil
import pandas as pd
from tqdm import tqdm

def separate_images(train_csv, test_csv, source_folder, train_dest, test_dest):
    # Create destination folders if they don't exist
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    # Read CSV files
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Function to move images
    def move_images(df, dest_folder):
        moved = 0
        skipped = 0
        for image_link in tqdm(df['image_link'], desc=f"Moving images to {os.path.basename(dest_folder)}"):
            filename = os.path.basename(image_link)
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            
            if os.path.exists(source_path):
                if not os.path.exists(dest_path):
                    shutil.move(source_path, dest_path)
                    moved += 1
                else:
                    tqdm.write(f"Skipped: {filename} already exists in destination")
                    skipped += 1
            else:
                tqdm.write(f"Warning: Image not found in source - {filename}")
                skipped += 1
        return moved, skipped

    # Move train images
    print("Processing train images:")
    train_moved, train_skipped = move_images(train_df, train_dest)
    print(f"Moved {train_moved} train images to {train_dest}")
    print(f"Skipped {train_skipped} train images")

    # Move test images
    print("\nProcessing test images:")
    test_moved, test_skipped = move_images(test_df, test_dest)
    print(f"Moved {test_moved} test images to {test_dest}")
    print(f"Skipped {test_skipped} test images")

if __name__ == "__main__":
    train_csv = './dataset/train.csv'
    test_csv = './dataset/test.csv'
    source_folder = './images'
    train_dest = './train_images'
    test_dest = './test_images'

    separate_images(train_csv, test_csv, source_folder, train_dest, test_dest)
